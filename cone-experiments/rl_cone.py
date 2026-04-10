#!/usr/bin/env python3
"""RL training for cone query ordering.
Trains a permutation-symmetric policy to minimize cumulative MSE.

Usage:
  python3 rl_cone.py --d 8 --lr 3e-4 --hidden 64 --hours 2 --eval-every 100
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import argparse
import json
import os

# Import cone primitives (no numba needed for D=8)
from run_cones import (
    oracle_query_single, compute_F_true,
    _init_layer_vals, _update_layer_vals, _estimate_from_layer0,
    funnel, warmup,
)

# ---------------------------------------------------------------------------
# Vectorized Environment
# ---------------------------------------------------------------------------

class ConeEnvBatch:
    """Batched cone environment for fast RL training."""

    def __init__(self, D, n_envs, seed=42):
        self.D = D
        self.n_envs = n_envs
        self.W = funnel(D)
        self.offsets = [sum(self.W[:i]) for i in range(D)]
        self.total_q = sum(self.W)
        self.W0 = self.W[0]
        self.widths = np.array(self.W, dtype=np.int32)
        self.offsets_arr = np.array(self.offsets, dtype=np.int32)

        # Layer assignment for each flat node
        self.flat_layer = np.repeat(np.arange(D, dtype=np.int32), self.W)
        self.layer_off = np.array([sum(self.W[:i]) for i in range(D)], dtype=np.int32)

        self.rng = np.random.default_rng(seed)
        self.n_features = 7

        # Pre-allocate per-env state
        self._alloc()

    def _alloc(self):
        n, N = self.n_envs, self.total_q
        self.oracle_seeds = np.zeros(n, dtype=np.int64)
        self.F_true = np.zeros(n, dtype=np.float64)
        self.is_known = np.zeros((n, N), dtype=np.bool_)  # qid-level
        self.qid_val = np.zeros((n, N), dtype=np.int32)
        self.g_queried = np.zeros((n, N), dtype=np.bool_)
        self.g_reach = np.zeros((n, N), dtype=np.float64)
        self.est = np.zeros(n, dtype=np.float64)
        self.mse_sum = np.zeros(n, dtype=np.float64)
        self.qi = np.zeros(n, dtype=np.int32)
        self.done = np.zeros(n, dtype=np.bool_)
        # Store layer_vals as flat array per env
        self.layer_vals_flat = np.zeros((n, N), dtype=np.float64)
        # Per-env layer_vals as list of lists (needed for _update_layer_vals)
        self.layer_vals = [[None]*self.D for _ in range(n)]

    def reset_env(self, idx):
        """Reset a single environment."""
        D = self.D; W = self.W; ol = self.offsets
        N = self.total_q; W0 = self.W0

        oracle_seed = int(self.rng.integers(0, 2**62))
        strategy_seed = int(self.rng.integers(0, 2**62))
        self.oracle_seeds[idx] = oracle_seed

        widths = self.widths; offsets_arr = self.offsets_arr
        max_qid = N

        F_true = compute_F_true(widths, offsets_arr, D, oracle_seed)
        self.F_true[idx] = F_true

        self.is_known[idx, :] = False
        self.qid_val[idx, :] = 0
        self.g_queried[idx, :] = False
        self.g_reach[idx, :] = 0.0
        self.g_reach[idx, :W0] = 1.0
        self.mse_sum[idx] = F_true * F_true  # step 0 (before any query)
        self.qi[idx] = 0
        self.done[idx] = False
        self.est[idx] = 0.0

        # Phase 1: one complete trace
        rng_local = np.random.default_rng(strategy_seed)
        v0 = int(rng_local.integers(0, W0))
        cur = v0

        for layer in range(D):
            qid = ol[layer] + cur
            nv = W[layer + 1] if layer < D - 1 else 2
            val = oracle_query_single(oracle_seed, qid, nv)
            self.is_known[idx, qid] = True
            self.qid_val[idx, qid] = val
            flat = self.layer_off[layer] + cur
            self.g_queried[idx, flat] = True
            self.qi[idx] += 1

            # Propagate reach
            if layer < D - 1:
                succ_flat = self.layer_off[layer + 1] + val
                self.g_reach[idx, succ_flat] += self.g_reach[idx, flat]
                self.mse_sum[idx] += F_true * F_true
                cur = val
            else:
                # Terminal — init layer_vals
                is_k = self.is_known[idx]
                q_v = self.qid_val[idx]
                lv = _init_layer_vals(widths, offsets_arr, D, is_k, q_v, max_qid)
                self.layer_vals[idx] = lv
                for l in range(D):
                    a = int(self.layer_off[l])
                    self.layer_vals_flat[idx, a:a+W[l]] = lv[l]
                est = _estimate_from_layer0(lv[0], W0)
                self.est[idx] = est
                self.mse_sum[idx] += (est - F_true) ** 2

    def reset_all(self):
        for i in range(self.n_envs):
            self.reset_env(i)

    def step(self, actions):
        """Execute actions for all envs. actions[i] = flat node index.
        Returns (rewards, dones)."""
        D = self.D; W = self.W; ol = self.offsets
        N = self.total_q; W0 = self.W0
        widths = self.widths; offsets_arr = self.offsets_arr

        rewards = np.zeros(self.n_envs, dtype=np.float64)
        new_dones = np.zeros(self.n_envs, dtype=np.bool_)

        for i in range(self.n_envs):
            if self.done[i]:
                continue

            flat = int(actions[i])
            layer = int(self.flat_layer[flat])
            node = flat - self.layer_off[layer]

            # Execute query
            qid = ol[layer] + node
            nv = W[layer + 1] if layer < D - 1 else 2
            val = oracle_query_single(int(self.oracle_seeds[i]), qid, nv)
            self.is_known[i, qid] = True
            self.qid_val[i, qid] = val
            self.g_queried[i, flat] = True
            self.qi[i] += 1

            # Propagate reach
            if layer < D - 1:
                succ_flat = self.layer_off[layer + 1] + val
                self.g_reach[i, succ_flat] += self.g_reach[i, flat]

            # Update layer_vals
            is_k = self.is_known[i]
            q_v = self.qid_val[i]
            _update_layer_vals(widths, offsets_arr, D, is_k, q_v, N,
                             self.layer_vals[i], layer)
            est = _estimate_from_layer0(self.layer_vals[i][0], W0)
            self.est[i] = est

            mse = (est - self.F_true[i]) ** 2
            self.mse_sum[i] += mse
            rewards[i] = -mse  # per-step reward

            # Update flat layer_vals
            for l in range(min(layer, D-1) + 1):
                a = int(self.layer_off[l])
                self.layer_vals_flat[i, a:a+W[l]] = self.layer_vals[i][l]

            # Check if done
            if self.qi[i] >= N or abs(est - self.F_true[i]) < 1e-15:
                self.done[i] = True
                new_dones[i] = True
                # Fill remaining MSE
                remaining = N - self.qi[i]
                if remaining > 0:
                    self.mse_sum[i] += remaining * mse

        return rewards, new_dones

    def get_obs(self):
        """Get observations for all envs. Returns [n_envs, total_q, n_features]."""
        n, N, D = self.n_envs, self.total_q, self.D

        obs = np.zeros((n, N, self.n_features), dtype=np.float32)

        for i in range(n):
            # Feature 0: layer (normalized)
            obs[i, :, 0] = self.flat_layer / (D - 1)
            # Feature 1: is_queried
            obs[i, :, 1] = self.g_queried[i].astype(np.float32)
            # Feature 2: reach (log-normalized)
            reach = self.g_reach[i]
            obs[i, :, 2] = np.log1p(reach) / max(np.log1p(np.max(reach)), 1e-8)
            # Feature 3: value (layer_vals)
            obs[i, :, 3] = self.layer_vals_flat[i]
            # Feature 4: prop_factor (computed inline)
            pf = self._compute_prop_factors(i)
            pf_max = max(np.max(np.abs(pf)), 1e-8)
            obs[i, :, 4] = pf / pf_max
            # Feature 5: var of next layer
            for l in range(D):
                a = int(self.layer_off[l])
                w = self.W[l]
                if l < D - 1:
                    a_next = int(self.layer_off[l+1])
                    w_next = self.W[l+1]
                    var_next = float(np.var(self.layer_vals_flat[i, a_next:a_next+w_next]))
                else:
                    var_next = 1.0
                obs[i, a:a+w, 5] = var_next
            # Feature 6: valid action mask (as feature)
            valid = (~self.g_queried[i]) & (self.g_reach[i] > 0)
            obs[i, :, 6] = valid.astype(np.float32)

        return obs

    def get_mask(self):
        """Get valid action masks. Returns [n_envs, total_q] bool."""
        mask = (~self.g_queried) & (self.g_reach > 0) & (~self.done[:, None])
        return mask

    def _compute_prop_factors(self, idx):
        """Compute prop_factor for all nodes in env idx."""
        D = self.D; W = self.W
        N = self.total_q
        pf = np.zeros(N, dtype=np.float64)
        W0 = self.W0
        pf[:W0] = 1.0 / W0

        for l in range(1, D):
            a_prev = int(self.layer_off[l-1]); W_prev = W[l-1]
            a_cur = int(self.layer_off[l]); W_cur = W[l]

            unk_sum = 0.0
            known_to = np.zeros(W_cur, dtype=np.float64)
            for v in range(W_prev):
                qid = self.offsets[l-1] + v
                if self.is_known[idx, qid]:
                    dest = self.qid_val[idx, qid]
                    known_to[dest] += pf[a_prev + v]
                else:
                    unk_sum += pf[a_prev + v]
            for u in range(W_cur):
                pf[a_cur + u] = known_to[u] + unk_sum / W_cur

        return pf


# ---------------------------------------------------------------------------
# Model (permutation-symmetric within layers)
# ---------------------------------------------------------------------------

class ConePolicy(nn.Module):
    def __init__(self, n_features=7, hidden=64):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Score: node_emb + layer_context + global_context
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, mask, layer_ids):
        """
        obs: [B, N, F]
        mask: [B, N] bool — valid actions
        layer_ids: [N] int — layer assignment per node
        """
        B, N, F = obs.shape
        H = self.node_enc[0].in_features
        H = self.node_enc[-2].out_features  # hidden dim

        node_emb = self.node_enc(obs)  # [B, N, H]

        # Per-layer mean pool
        D = int(layer_ids.max()) + 1
        layer_ctx = torch.zeros(B, N, node_emb.shape[-1], device=obs.device)
        layer_summaries = []
        for l in range(D):
            lmask = (layer_ids == l)  # [N]
            layer_nodes = node_emb[:, lmask, :]  # [B, W_l, H]
            pool = layer_nodes.mean(dim=1)  # [B, H]
            layer_summaries.append(pool)
            layer_ctx[:, lmask, :] = pool.unsqueeze(1)

        # Global context = mean of layer summaries
        global_ctx = torch.stack(layer_summaries, dim=1).mean(dim=1)  # [B, H]
        global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, N, -1)  # [B, N, H]

        # Score each node
        score_in = torch.cat([node_emb, layer_ctx, global_ctx_exp], dim=-1)
        scores = self.score_head(score_in).squeeze(-1)  # [B, N]
        scores = scores.masked_fill(~mask, -1e9)

        # Value from global context
        value = self.value_head(global_ctx).squeeze(-1)  # [B]

        return scores, value


# ---------------------------------------------------------------------------
# PPO Training
# ---------------------------------------------------------------------------

def train(args):
    D = args.d
    W = funnel(D)
    total_q = sum(W)
    k = D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Warmup numba
    warmup()

    n_envs = args.n_envs
    env = ConeEnvBatch(D, n_envs, seed=args.seed)
    env.reset_all()

    flat_layer_t = torch.tensor(env.flat_layer, dtype=torch.long, device=device)

    policy = ConePolicy(n_features=env.n_features, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"D={D}, n_envs={n_envs}, hidden={args.hidden}, lr={args.lr}")
    print(f"Model params: {n_params}")
    print(f"Total nodes per env: {total_q}, episode steps: ~{total_q}")
    print()

    # PPO hyperparams
    gamma = 1.0
    gae_lambda = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 0.5
    ppo_epochs = 4
    mini_batch_size = min(2048, n_envs * total_q)

    # Training loop
    start_time = time.monotonic()
    max_time = args.hours * 3600
    total_episodes = 0
    epoch = 0

    # Tracking
    episode_returns = []
    best_eval_ratio = float('inf')

    while time.monotonic() - start_time < max_time:
        epoch += 1

        # Collect one rollout per env
        obs_buf = []
        act_buf = []
        rew_buf = []
        val_buf = []
        logp_buf = []
        mask_buf = []
        done_buf = []

        env.reset_all()

        for step in range(total_q):
            obs_np = env.get_obs()
            mask_np = env.get_mask()

            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)

            with torch.no_grad():
                scores, values = policy(obs_t, mask_t, flat_layer_t)
                dist = Categorical(logits=scores)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            obs_buf.append(obs_t)
            mask_buf.append(mask_t)
            act_buf.append(actions)
            val_buf.append(values)
            logp_buf.append(log_probs)

            actions_np = actions.cpu().numpy()
            rewards, dones = env.step(actions_np)
            rew_buf.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(dones, dtype=torch.bool, device=device))

            # Check if all done
            if env.done.all():
                # Pad remaining steps
                for _ in range(step + 1, total_q):
                    rew_buf.append(torch.zeros(n_envs, device=device))
                    done_buf.append(torch.ones(n_envs, dtype=torch.bool, device=device))
                    obs_buf.append(obs_buf[-1])
                    mask_buf.append(mask_buf[-1])
                    act_buf.append(act_buf[-1])
                    val_buf.append(val_buf[-1])
                    logp_buf.append(logp_buf[-1])
                break

        T = len(rew_buf)
        total_episodes += n_envs

        # Track episode returns
        ep_mse = env.mse_sum / k
        episode_returns.extend(ep_mse.tolist())

        # Compute GAE advantages
        rewards = torch.stack(rew_buf)  # [T, B]
        values = torch.stack(val_buf)   # [T, B]
        dones = torch.stack(done_buf)   # [T, B]

        with torch.no_grad():
            # Bootstrap value for last step
            last_val = torch.zeros(n_envs, device=device)
            advantages = torch.zeros(T, n_envs, device=device)
            gae = torch.zeros(n_envs, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = last_val
                else:
                    next_val = values[t + 1]
                delta = rewards[t] + gamma * next_val * (~dones[t]).float() - values[t]
                gae = delta + gamma * gae_lambda * (~dones[t]).float() * gae
                advantages[t] = gae

            returns = advantages + values

        # Flatten for PPO update
        obs_flat = torch.cat(obs_buf, dim=0)    # not right shape...

        # Actually reshape: [T, B, ...] → [T*B, ...]
        obs_all = torch.stack(obs_buf)      # [T, B, N, F]
        mask_all = torch.stack(mask_buf)    # [T, B, N]
        act_all = torch.stack(act_buf)      # [T, B]
        logp_all = torch.stack(logp_buf)    # [T, B]
        adv_all = advantages               # [T, B]
        ret_all = returns                   # [T, B]

        # Remove padded/done steps
        valid = ~torch.stack(done_buf)  # [T, B] — True for valid steps
        # Include the step where done BECOMES true
        for t in range(T):
            if t == 0:
                valid[t] |= torch.stack(done_buf)[t]  # first done step is valid
            else:
                newly_done = torch.stack(done_buf)[t] & ~torch.stack(done_buf)[t-1]
                valid[t] |= newly_done

        # Flatten valid steps
        valid_flat = valid.reshape(-1)
        B_total = T * n_envs

        obs_f = obs_all.reshape(B_total, total_q, -1)[valid_flat]
        mask_f = mask_all.reshape(B_total, total_q)[valid_flat]
        act_f = act_all.reshape(B_total)[valid_flat]
        logp_f = logp_all.reshape(B_total)[valid_flat]
        adv_f = adv_all.reshape(B_total)[valid_flat]
        ret_f = ret_all.reshape(B_total)[valid_flat]

        # Normalize advantages
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        # PPO update
        n_samples = obs_f.shape[0]
        total_loss_sum = 0.0
        n_updates = 0

        for _ in range(ppo_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, mini_batch_size):
                end = min(start + mini_batch_size, n_samples)
                idx = perm[start:end]

                scores, values_pred = policy(obs_f[idx], mask_f[idx], flat_layer_t)
                dist = Categorical(logits=scores)
                new_logp = dist.log_prob(act_f[idx])
                entropy = dist.entropy()

                # PPO clipped objective
                ratio = torch.exp(new_logp - logp_f[idx])
                adv_mb = adv_f[idx]
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_pred, ret_f[idx])
                entropy_loss = -entropy.mean()

                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                total_loss_sum += loss.item()
                n_updates += 1

        # Logging
        elapsed = time.monotonic() - start_time
        recent = episode_returns[-n_envs*10:] if len(episode_returns) >= n_envs*10 else episode_returns
        mean_ratio = np.mean(recent)

        if epoch % args.eval_every == 0 or epoch <= 5:
            print(f"Epoch {epoch:5d} | {elapsed/60:5.1f}m | episodes={total_episodes:7d} | "
                  f"ratio={mean_ratio:.4f} | loss={total_loss_sum/max(n_updates,1):.4f} | "
                  f"T={T}", flush=True)

        # Save periodically
        if epoch % (args.eval_every * 5) == 0:
            torch.save({
                'epoch': epoch,
                'model': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'episode_returns': episode_returns[-10000:],
                'args': vars(args),
            }, f"checkpoint_d{D}_h{args.hidden}_lr{args.lr}_s{args.seed}.pt")

    # Final save
    torch.save({
        'epoch': epoch,
        'model': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode_returns': episode_returns[-10000:],
        'args': vars(args),
    }, f"final_d{D}_h{args.hidden}_lr{args.lr}_s{args.seed}.pt")

    print(f"\nTraining complete: {epoch} epochs, {total_episodes} episodes, "
          f"{(time.monotonic()-start_time)/60:.1f} min")
    print(f"Final ratio (last 1000): {np.mean(episode_returns[-1000:]):.5f}")

    # Quick eval: compare to fwd-merge
    print("\nEvaluating vs fwd-merge (1000 trials)...")
    from run_cones import evaluate_single_trial as eval_fwd

    eval_seeds = [(int(np.random.default_rng(i+99999).integers(0,2**62)),
                   int(np.random.default_rng(i+99999).integers(0,2**62)))
                  for i in range(1000)]

    fwd_results = []
    for s in eval_seeds:
        fwd_results.append(eval_fwd((list(W), list(env.offsets), D, s[0], s[1])) / k)

    # Run policy
    policy.eval()
    eval_env = ConeEnvBatch(D, 1, seed=77777)
    policy_results = []
    for s in eval_seeds:
        eval_env.rng = np.random.default_rng(s[1])
        eval_env.oracle_seeds[0] = s[0]
        eval_env.reset_env(0)
        for step in range(total_q):
            if eval_env.done[0]:
                break
            obs = torch.tensor(eval_env.get_obs(), dtype=torch.float32, device=device)
            mask = torch.tensor(eval_env.get_mask(), dtype=torch.bool, device=device)
            with torch.no_grad():
                scores, _ = policy(obs, mask, flat_layer_t)
            action = scores[0].argmax().item()
            eval_env.step(np.array([action]))
        policy_results.append(float(eval_env.mse_sum[0]) / k)

    fwd_arr = np.array(fwd_results)
    pol_arr = np.array(policy_results)
    diff = pol_arr - fwd_arr
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    sigma = np.mean(diff) / se if se > 0 else 0

    print(f"  fwd-merge: {np.mean(fwd_arr):.5f} ± {np.std(fwd_arr, ddof=1)/np.sqrt(len(fwd_arr)):.4f}")
    print(f"  policy:    {np.mean(pol_arr):.5f} ± {np.std(pol_arr, ddof=1)/np.sqrt(len(pol_arr)):.4f}")
    print(f"  Δ: {np.mean(diff):+.5f} ± {se:.5f} ({sigma:+.1f}σ)")

    result = {
        "D": D, "hidden": args.hidden, "lr": args.lr, "seed": args.seed,
        "epochs": epoch, "episodes": total_episodes,
        "final_ratio": float(np.mean(episode_returns[-1000:])),
        "fwd_ratio": float(np.mean(fwd_arr)),
        "policy_ratio": float(np.mean(pol_arr)),
        "delta": float(np.mean(diff)),
        "sigma": float(sigma),
    }
    with open(f"result_d{D}_h{args.hidden}_lr{args.lr}_s{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()

    train(args)
