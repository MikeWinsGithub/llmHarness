#!/usr/bin/env python3
"""Two-phase RL for cone query ordering.
Phase 1: Imitation learning — learn to mimic fwd-merge via supervised cross-entropy.
Phase 2: RL fine-tuning — PPO on cumulative MSE reward starting from the imitation policy.

Usage:
  python3 rl_cone2.py --d 8 --hidden 32 --imitate-epochs 200 --hours 1.8
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import argparse
import json

from run_cones import (
    oracle_query_single, compute_F_true,
    _init_layer_vals, _update_layer_vals, _estimate_from_layer0,
    funnel, warmup,
    evaluate_single_trial as eval_fwd,
)


# ---------------------------------------------------------------------------
# Permutation-symmetric model
# ---------------------------------------------------------------------------

class ConePolicy(nn.Module):
    def __init__(self, n_features=7, hidden=32):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, mask, layer_ids):
        B, N, _ = obs.shape
        node_emb = self.node_enc(obs)
        D = int(layer_ids.max()) + 1
        layer_ctx = torch.zeros_like(node_emb)
        layer_pools = []
        for l in range(D):
            lm = (layer_ids == l)
            pool = node_emb[:, lm, :].mean(dim=1)
            layer_pools.append(pool)
            layer_ctx[:, lm, :] = pool.unsqueeze(1)
        global_ctx = torch.stack(layer_pools, dim=1).mean(dim=1)
        gc_exp = global_ctx.unsqueeze(1).expand_as(node_emb)
        score_in = torch.cat([node_emb, layer_ctx, gc_exp], dim=-1)
        scores = self.score_head(score_in).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        value = self.value_head(global_ctx).squeeze(-1)
        return scores, value


# ---------------------------------------------------------------------------
# Single-env cone with fwd-merge oracle
# ---------------------------------------------------------------------------

class ConeEnv:
    def __init__(self, D):
        self.D = D
        self.W = funnel(D)
        self.offsets = [sum(self.W[:i]) for i in range(D)]
        self.total_q = sum(self.W)
        self.W0 = self.W[0]
        self.widths = np.array(self.W, dtype=np.int32)
        self.offsets_arr = np.array(self.offsets, dtype=np.int32)
        self.flat_layer = np.repeat(np.arange(D, dtype=np.int32), self.W)
        self.layer_off = np.array([sum(self.W[:i]) for i in range(D)], dtype=np.int32)
        self.n_features = 7

    def reset(self, oracle_seed, strategy_seed):
        D = self.D; W = self.W; ol = self.offsets; W0 = self.W0
        N = self.total_q

        self.oracle_seed = oracle_seed
        self.F_true = compute_F_true(self.widths, self.offsets_arr, D, oracle_seed)
        self.is_known = np.zeros(N, dtype=np.bool_)
        self.qid_val = np.zeros(N, dtype=np.int32)
        self.g_queried = np.zeros(N, dtype=np.bool_)
        self.g_succ = np.full(N, -1, dtype=np.int32)
        self.g_reach = np.zeros(N, dtype=np.float64)
        self.g_reach[:W0] = 1.0
        self.est = 0.0
        self.mse_sum = self.F_true ** 2
        self.qi = 0
        self.done = False

        # fwd-merge state
        rng = np.random.default_rng(strategy_seed)
        self.l0_perm = list(rng.permutation(W0))
        self.l0_idx = 0
        self.fwd_layer = 0
        self.fwd_node = None

        # Phase 1: one complete trace
        v0 = self.l0_perm[self.l0_idx]; self.l0_idx += 1
        cur = v0
        for layer in range(D):
            self._do_query(layer, cur)
            self.qi += 1
            if layer < D - 1:
                self.mse_sum += self.F_true ** 2
                cur = int(self.qid_val[ol[layer] + cur])
            else:
                self.layer_vals = _init_layer_vals(self.widths, self.offsets_arr, D,
                                                   self.is_known, self.qid_val, N)
                self.est = _estimate_from_layer0(self.layer_vals[0], W0)
                self.mse_sum += (self.est - self.F_true) ** 2

        self._advance_fwd()

    def _do_query(self, layer, node):
        D = self.D; W = self.W; ol = self.offsets
        flat = self.layer_off[layer] + node
        qid = ol[layer] + node
        nv = W[layer + 1] if layer < D - 1 else 2
        val = oracle_query_single(self.oracle_seed, qid, nv)
        self.is_known[qid] = True
        self.qid_val[qid] = val
        self.g_queried[flat] = True
        if layer < D - 1:
            self.g_succ[flat] = val
            succ_flat = self.layer_off[layer + 1] + val
            self.g_reach[succ_flat] += self.g_reach[flat]

    def _advance_fwd(self):
        D = self.D; W0 = self.W0
        fl = self.fwd_layer; fn = self.fwd_node
        if fn is not None:
            flat = self.layer_off[fl] + fn
            if not self.g_queried[flat]:
                self.fwd_layer = fl; self.fwd_node = fn; return
            if fl < D - 1 and self.g_succ[flat] >= 0:
                self.fwd_layer = fl + 1; self.fwd_node = int(self.g_succ[flat])
                return self._advance_fwd()
            self.fwd_node = None
        while self.l0_idx < W0:
            v = self.l0_perm[self.l0_idx]
            flat = self.layer_off[0] + v
            if not self.g_queried[flat]:
                self.fwd_layer = 0; self.fwd_node = v; return
            cl, cn = 0, v
            while cl < D:
                cf = self.layer_off[cl] + cn
                if not self.g_queried[cf]:
                    self.fwd_layer = cl; self.fwd_node = cn; return
                if cl < D - 1 and self.g_succ[cf] >= 0:
                    cn = int(self.g_succ[cf]); cl += 1
                else:
                    break
            self.l0_idx += 1
        self.fwd_node = None

    def get_fwd_action(self):
        """Return the flat index of what fwd-merge would query next."""
        if self.fwd_node is None:
            return -1
        return int(self.layer_off[self.fwd_layer] + self.fwd_node)

    def step(self, action_flat):
        """Execute a query. Returns (reward, done)."""
        if self.done:
            return 0.0, True
        layer = int(self.flat_layer[action_flat])
        node = action_flat - self.layer_off[layer]
        self._do_query(layer, node)
        self.qi += 1
        _update_layer_vals(self.widths, self.offsets_arr, self.D,
                          self.is_known, self.qid_val, self.total_q,
                          self.layer_vals, layer)
        self.est = _estimate_from_layer0(self.layer_vals[0], self.W0)
        mse = (self.est - self.F_true) ** 2
        self.mse_sum += mse
        if self.qi >= self.total_q or abs(self.est - self.F_true) < 1e-15:
            remaining = self.total_q - self.qi
            if remaining > 0:
                self.mse_sum += remaining * mse
            self.done = True
        self._advance_fwd()
        return -mse, self.done

    def get_obs(self):
        N = self.total_q; D = self.D
        obs = np.zeros((N, self.n_features), dtype=np.float32)
        obs[:, 0] = self.flat_layer / (D - 1)
        obs[:, 1] = self.g_queried.astype(np.float32)
        reach = self.g_reach
        obs[:, 2] = np.log1p(reach) / max(np.log1p(np.max(reach)), 1e-8)
        # Layer vals
        for l in range(D):
            a = int(self.layer_off[l]); w = self.W[l]
            if self.layer_vals is not None and self.layer_vals[l] is not None:
                obs[a:a+w, 3] = self.layer_vals[l]
        # Prop factors
        pf = self._prop_factors()
        pf_max = max(np.max(np.abs(pf)), 1e-8)
        obs[:, 4] = pf / pf_max
        # Var next
        for l in range(D):
            a = int(self.layer_off[l]); w = self.W[l]
            if l < D - 1 and self.layer_vals is not None:
                a2 = int(self.layer_off[l+1]); w2 = self.W[l+1]
                obs[a:a+w, 5] = float(np.var(self.layer_vals[l+1]))
            else:
                obs[a:a+w, 5] = 1.0
        obs[:, 6] = ((~self.g_queried) & (self.g_reach > 0)).astype(np.float32)
        return obs

    def get_mask(self):
        return (~self.g_queried) & (self.g_reach > 0)

    def _prop_factors(self):
        D = self.D; W = self.W; N = self.total_q
        pf = np.zeros(N, dtype=np.float64)
        pf[:self.W0] = 1.0 / self.W0
        for l in range(1, D):
            ap = int(self.layer_off[l-1]); Wp = W[l-1]
            ac = int(self.layer_off[l]); Wc = W[l]
            unk_sum = 0.0
            known_to = np.zeros(Wc, dtype=np.float64)
            for v in range(Wp):
                qid = self.offsets[l-1] + v
                if self.is_known[qid]:
                    known_to[self.qid_val[qid]] += pf[ap + v]
                else:
                    unk_sum += pf[ap + v]
            for u in range(Wc):
                pf[ac + u] = known_to[u] + unk_sum / Wc
        return pf


# ---------------------------------------------------------------------------
# Data collection for imitation
# ---------------------------------------------------------------------------

def collect_imitation_data(D, n_episodes, seed=42):
    """Run fwd-merge episodes, record (obs, fwd_action) at each step."""
    env = ConeEnv(D)
    rng = np.random.default_rng(seed)
    all_obs = []
    all_actions = []
    all_masks = []

    for ep in range(n_episodes):
        os_ = int(rng.integers(0, 2**62))
        ss_ = int(rng.integers(0, 2**62))
        env.reset(os_, ss_)

        for step in range(env.total_q):
            if env.done:
                break
            obs = env.get_obs()
            mask = env.get_mask()
            fwd_act = env.get_fwd_action()
            if fwd_act < 0 or not mask[fwd_act]:
                break

            all_obs.append(obs)
            all_actions.append(fwd_act)
            all_masks.append(mask)

            env.step(fwd_act)

    return (np.array(all_obs, dtype=np.float32),
            np.array(all_actions, dtype=np.int64),
            np.array(all_masks, dtype=np.bool_))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    D = args.d; W = funnel(D); total_q = sum(W); k = D
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup()

    flat_layer_t = torch.tensor(
        np.repeat(np.arange(D, dtype=np.int32), W), dtype=torch.long, device=device)

    policy = ConePolicy(n_features=7, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"D={D}, hidden={args.hidden}, lr={args.lr}, device={device}, params={n_params}")

    start_time = time.monotonic()

    # ===== PHASE 1: Imitation Learning =====
    print(f"\n{'='*60}")
    print(f"PHASE 1: Imitation Learning ({args.imitate_episodes} episodes)")
    print(f"{'='*60}")

    t0 = time.monotonic()
    obs_data, act_data, mask_data = collect_imitation_data(D, args.imitate_episodes, seed=args.seed)
    print(f"Collected {len(obs_data)} transitions in {time.monotonic()-t0:.1f}s")

    # Convert to tensors
    obs_t = torch.tensor(obs_data, dtype=torch.float32, device=device)
    act_t = torch.tensor(act_data, dtype=torch.long, device=device)
    mask_t = torch.tensor(mask_data, dtype=torch.bool, device=device)

    # Reshape: each transition is [1, N, F] for the model
    N_trans = len(obs_data)
    obs_t = obs_t.unsqueeze(1) if obs_t.dim() == 2 else obs_t  # already [N_trans, total_q, 7]

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    batch_size = min(512, N_trans)

    for epoch in range(args.imitate_epochs):
        perm = torch.randperm(N_trans, device=device)
        total_loss = 0.0; total_correct = 0; n_batches = 0

        for start in range(0, N_trans, batch_size):
            idx = perm[start:start+batch_size]
            b_obs = obs_t[idx]     # [B, total_q, 7]
            b_mask = mask_t[idx]   # [B, total_q]
            b_act = act_t[idx]     # [B]

            scores, _ = policy(b_obs, b_mask, flat_layer_t)
            loss = F.cross_entropy(scores, b_act)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = scores.argmax(dim=-1)
            total_correct += (preds == b_act).sum().item()
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch < 5:
            acc = total_correct / N_trans * 100
            elapsed = time.monotonic() - start_time
            print(f"  Epoch {epoch+1:4d} | loss={total_loss/n_batches:.4f} | "
                  f"acc={acc:.1f}% | {elapsed/60:.1f}m")

    # Quick eval of imitation policy
    print("\nImitation eval (500 trials)...")
    policy.eval()
    im_ratio = eval_policy(policy, D, flat_layer_t, device, n_trials=500)
    fwd_ratio = eval_fwd_merge(D, n_trials=500)
    print(f"  fwd-merge: {fwd_ratio:.5f}")
    print(f"  imitation: {im_ratio:.5f}")
    policy.train()

    # ===== PHASE 2: RL Fine-tuning =====
    print(f"\n{'='*60}")
    print("PHASE 2: RL Fine-tuning (PPO)")
    print(f"{'='*60}")

    # Lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr * 0.3)

    n_envs = args.n_envs
    envs = [ConeEnv(D) for _ in range(n_envs)]
    rng = np.random.default_rng(args.seed + 9999)

    gamma = 1.0; gae_lambda = 0.95; clip_eps = 0.2
    entropy_coef = 0.005; value_coef = 0.5
    max_time = args.hours * 3600
    epoch = 0
    total_episodes = 0
    episode_ratios = []

    while time.monotonic() - start_time < max_time:
        epoch += 1

        # Reset all envs
        for env in envs:
            env.reset(int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))

        obs_buf, act_buf, rew_buf, val_buf, logp_buf, mask_buf, done_buf = \
            [], [], [], [], [], [], []

        for step in range(total_q):
            # Collect obs from all envs
            obs_np = np.stack([env.get_obs() for env in envs])  # [B, N, F]
            mask_np = np.stack([env.get_mask() for env in envs])  # [B, N]

            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(mask_np, dtype=torch.bool, device=device)

            with torch.no_grad():
                scores, values = policy(obs_tensor, mask_tensor, flat_layer_t)
                dist = Categorical(logits=scores)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            obs_buf.append(obs_tensor)
            mask_buf.append(mask_tensor)
            act_buf.append(actions)
            val_buf.append(values)
            logp_buf.append(log_probs)

            # Step all envs
            actions_np = actions.cpu().numpy()
            rews = np.zeros(n_envs)
            dones = np.zeros(n_envs, dtype=bool)
            for i, env in enumerate(envs):
                if not env.done:
                    r, d = env.step(int(actions_np[i]))
                    rews[i] = r
                    dones[i] = d
                else:
                    dones[i] = True

            rew_buf.append(torch.tensor(rews, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(dones, dtype=torch.bool, device=device))

            if all(env.done for env in envs):
                break

        T = len(rew_buf)
        total_episodes += n_envs
        for env in envs:
            episode_ratios.append(env.mse_sum / k)

        # GAE
        rewards = torch.stack(rew_buf)
        values = torch.stack(val_buf)
        dones = torch.stack(done_buf)

        with torch.no_grad():
            advantages = torch.zeros(T, n_envs, device=device)
            gae = torch.zeros(n_envs, device=device)
            for t in reversed(range(T)):
                next_val = values[t + 1] if t < T - 1 else torch.zeros(n_envs, device=device)
                delta = rewards[t] + gamma * next_val * (~dones[t]).float() - values[t]
                gae = delta + gamma * gae_lambda * (~dones[t]).float() * gae
                advantages[t] = gae
            returns = advantages + values

        # Flatten valid steps
        valid = torch.zeros(T, n_envs, dtype=torch.bool, device=device)
        for t in range(T):
            if t == 0:
                valid[t] = True
            else:
                valid[t] = ~dones[t - 1]

        obs_f = torch.stack(obs_buf).reshape(-1, total_q, 7)[valid.reshape(-1)]
        mask_f = torch.stack(mask_buf).reshape(-1, total_q)[valid.reshape(-1)]
        act_f = torch.stack(act_buf).reshape(-1)[valid.reshape(-1)]
        logp_f = torch.stack(logp_buf).reshape(-1)[valid.reshape(-1)]
        adv_f = advantages.reshape(-1)[valid.reshape(-1)]
        ret_f = returns.reshape(-1)[valid.reshape(-1)]

        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        # PPO update
        n_samples = obs_f.shape[0]
        mb_size = min(2048, n_samples)

        for _ in range(4):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, mb_size):
                idx = perm[start:start+mb_size]
                scores, vpred = policy(obs_f[idx], mask_f[idx], flat_layer_t)
                dist = Categorical(logits=scores)
                new_logp = dist.log_prob(act_f[idx])
                entropy = dist.entropy()
                ratio = torch.exp(new_logp - logp_f[idx])
                s1 = ratio * adv_f[idx]
                s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_f[idx]
                loss = (-torch.min(s1, s2).mean()
                        + value_coef * F.mse_loss(vpred, ret_f[idx])
                        - entropy_coef * entropy.mean())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        if epoch % 50 == 0 or epoch <= 3:
            recent = episode_ratios[-n_envs*20:] if len(episode_ratios) > n_envs*20 else episode_ratios
            elapsed = time.monotonic() - start_time
            print(f"  RL Epoch {epoch:5d} | {elapsed/60:.1f}m | ep={total_episodes:7d} | "
                  f"ratio={np.mean(recent):.4f} | T={T}", flush=True)

    # ===== Final Evaluation =====
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (2000 trials)")
    print(f"{'='*60}")

    policy.eval()
    pol_ratio = eval_policy(policy, D, flat_layer_t, device, n_trials=2000, seed=77777)
    fwd_ratio_arr = eval_fwd_paired(D, n_trials=2000, seed=77777)
    pol_ratio_arr = eval_policy_paired(policy, D, flat_layer_t, device, n_trials=2000, seed=77777)

    diff = pol_ratio_arr - fwd_ratio_arr
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    sigma = np.mean(diff) / se if se > 0 else 0

    print(f"  fwd-merge: {np.mean(fwd_ratio_arr):.5f} ± {np.std(fwd_ratio_arr,ddof=1)/np.sqrt(len(fwd_ratio_arr)):.4f}")
    print(f"  policy:    {np.mean(pol_ratio_arr):.5f} ± {np.std(pol_ratio_arr,ddof=1)/np.sqrt(len(pol_ratio_arr)):.4f}")
    print(f"  Δ: {np.mean(diff):+.5f} ± {se:.5f} ({sigma:+.1f}σ)")

    # Save
    torch.save(policy.state_dict(), f"policy_d{D}_h{args.hidden}_s{args.seed}.pt")
    result = {
        "D": D, "hidden": args.hidden, "lr": args.lr, "seed": args.seed,
        "rl_epochs": epoch, "rl_episodes": total_episodes,
        "fwd_ratio": float(np.mean(fwd_ratio_arr)),
        "policy_ratio": float(np.mean(pol_ratio_arr)),
        "delta": float(np.mean(diff)), "sigma": float(sigma),
    }
    with open(f"result_d{D}_h{args.hidden}_s{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to result_d{D}_h{args.hidden}_s{args.seed}.json")


def eval_fwd_merge(D, n_trials=500, seed=42):
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_trials):
        total += eval_fwd((W, ol, D, int(rng.integers(0,2**62)), int(rng.integers(0,2**62)))) / k
    return total / n_trials


def eval_fwd_paired(D, n_trials=2000, seed=77777):
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        results.append(eval_fwd((W, ol, D, int(rng.integers(0,2**62)), int(rng.integers(0,2**62)))) / k)
    return np.array(results)


def eval_policy(policy, D, flat_layer_t, device, n_trials=500, seed=42):
    env = ConeEnv(D)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_trials):
        env.reset(int(rng.integers(0,2**62)), int(rng.integers(0,2**62)))
        for _ in range(env.total_q):
            if env.done: break
            obs = torch.tensor(env.get_obs(), dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(env.get_mask(), dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                scores, _ = policy(obs, mask, flat_layer_t)
            env.step(scores[0].argmax().item())
        total += env.mse_sum / D
    return total / n_trials


def eval_policy_paired(policy, D, flat_layer_t, device, n_trials=2000, seed=77777):
    env = ConeEnv(D)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        os_ = int(rng.integers(0,2**62)); ss_ = int(rng.integers(0,2**62))
        env.reset(os_, ss_)
        for _ in range(env.total_q):
            if env.done: break
            obs = torch.tensor(env.get_obs(), dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(env.get_mask(), dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                scores, _ = policy(obs, mask, flat_layer_t)
            env.step(scores[0].argmax().item())
        results.append(float(env.mse_sum / D))
    return np.array(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--imitate-episodes", type=int, default=5000)
    parser.add_argument("--imitate-epochs", type=int, default=200)
    parser.add_argument("--hours", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
