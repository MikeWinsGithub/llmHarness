#!/usr/bin/env python3
"""2-step lookahead using expected variance reduction (not realized MSE).

At each step, pick q1 to maximize:
  2 * E_o1[(Δest)²]  +  E_o1[max_q2 E_o2[(Δest)²]]  +  λ * heuristic(q1)

The factor of 2 on step-1 is because reducing variance at step i+1
also reduces the starting point for step i+2.

No F_true used — only enumerates possible outcomes under uniform prior.
"""

import numpy as np
from numba import njit
import time
import multiprocessing as mp
import argparse
import os

from run_cones import (
    oracle_query_single, compute_F_true,
    _init_layer_vals, _update_layer_vals, _estimate_from_layer0,
    evaluate_single_trial as evaluate_fwd_trial,
    funnel, warmup,
)
from lookahead2 import _forward_reach_incremental, _compute_layer_vars


def evaluate_lookahead2_var(args):
    widths_list, offsets_list, D, oracle_seed, strategy_seed, K, lam, threshold = args
    widths = np.array(widths_list, dtype=np.int32)
    offsets_arr = np.array(offsets_list, dtype=np.int32)
    W = widths_list
    total_q = sum(W)
    max_qid = offsets_list[-1] + W[-1]
    W0 = W[0]

    F_true = compute_F_true(widths, offsets_arr, D, oracle_seed)
    F_true_sq = F_true * F_true

    layer_off = np.array([sum(W[:i]) for i in range(D)], dtype=np.int32)
    flat_layer = np.repeat(np.arange(D, dtype=np.int32), W)

    is_known = np.zeros(max_qid, dtype=np.bool_)
    qid_val = np.zeros(max_qid, dtype=np.int32)

    g_queried = np.zeros(total_q, dtype=np.bool_)
    g_succ = np.full(total_q, -1, dtype=np.int32)
    g_known = np.zeros(total_q, dtype=np.bool_)
    g_reach = np.zeros(total_q, dtype=np.float64)
    g_preds = [[] for _ in range(total_q)]

    rng = np.random.default_rng(strategy_seed)
    l0_perm = list(rng.permutation(W0))
    l0_idx = 0
    g_reach[:W0] = 1.0
    exp_reach = np.zeros(total_q, dtype=np.float64)
    exp_reach[:W0] = 1.0

    def fi(layer, node):
        return layer_off[layer] + node

    def propagate_known(flat_idx):
        stk = [flat_idx]
        while stk:
            idx = stk.pop()
            if g_known[idx]: continue
            g_known[idx] = True
            if flat_layer[idx] > 0:
                for p in g_preds[idx]:
                    if not g_known[p]: stk.append(p)

    def do_query(layer, node):
        flat = fi(layer, node)
        qid = offsets_list[layer] + node
        nv = W[layer + 1] if layer < D - 1 else 2
        val = oracle_query_single(oracle_seed, qid, nv)
        is_known[qid] = True
        qid_val[qid] = val
        g_queried[flat] = True
        if layer == D - 1:
            g_known[flat] = True
            for p in g_preds[flat]: propagate_known(p)
        else:
            g_succ[flat] = val
            succ_flat = fi(layer + 1, val)
            g_preds[succ_flat].append(flat)
            delta = g_reach[flat]
            cl, cn_flat = layer + 1, succ_flat
            cn = val
            while True:
                g_reach[cn_flat] += delta
                if cl < D - 1 and g_queried[cn_flat]:
                    cn = int(g_succ[cn_flat]); cl += 1; cn_flat = fi(cl, cn)
                else: break
            if g_known[succ_flat]: propagate_known(flat)
        return layer

    # Precomputed per-step: variance at each layer
    layer_var_cache = np.zeros(D, dtype=np.float64)

    def update_layer_var_cache():
        for l in range(D):
            layer_var_cache[l] = float(np.var(layer_vals[l]))

    def compute_prop_factor(layer, node):
        """Compute Δest per unit change in layer_vals[layer][node].
        Backward pass: O(sum of widths from 0 to layer)."""
        # delta[v] = change in layer_vals[l][v] per unit change at (layer, node)
        W_cur = W[layer]
        delta = np.zeros(W_cur, dtype=np.float64)
        delta[node] = 1.0

        for l in range(layer - 1, -1, -1):
            W_next = W[l + 1]  # = len(delta)
            mean_delta = np.sum(delta) / W_next

            W_l = W[l]
            new_delta = np.empty(W_l, dtype=np.float64)
            for v in range(W_l):
                qid = offsets_list[l] + v
                if is_known[qid]:
                    new_delta[v] = delta[qid_val[qid]]
                else:
                    new_delta[v] = mean_delta
            delta = new_delta

        return float(np.sum(delta)) / W0

    def fast_var_reduction(layer, node):
        """E[(Δest)²] = prop_factor² × var(next_layer_values).
        O(D × W_max) via scalar propagation instead of O(nv × D × W_max)."""
        pf = compute_prop_factor(layer, node)
        if layer < D - 1:
            dest_var = layer_var_cache[layer + 1]
        else:
            # Terminal: uniform ±1 outcomes, var = 1 - mean²
            mean_term = float(np.mean(layer_vals[D - 1]))
            dest_var = 1.0 - mean_term * mean_term
        return pf * pf * dest_var

    def two_step_score(q1_flat, q2_candidates):
        """Compute 2*E[R1] + E[max_q2 R2] for q1.
        Uses fast scalar computation for L0 queries."""
        q1_layer = int(flat_layer[q1_flat])
        q1_node = q1_flat - layer_off[q1_layer]

        if q1_layer == 0:
            return two_step_score_L0(q1_node, q2_candidates)

        # Non-L0: use approximate 2-step score
        # Step 1: exact R1 via prop_factor
        R1 = fast_var_reduction(q1_layer, q1_node)

        # Step 2: approximate — use pre-q1 prop_factors for q2 candidates
        # (ignoring the small change q1 makes to the state)
        best_R2 = 0.0
        for q2f in q2_candidates:
            q2f = int(q2f)
            if q2f == q1_flat:
                continue
            q2l = int(flat_layer[q2f]); q2n = q2f - layer_off[q2l]
            R2 = fast_var_reduction(q2l, q2n)
            if R2 > best_R2:
                best_R2 = R2
        # Chain continuation bonus (approximate)
        if q1_layer < D - 1:
            # Average chain continuation R2 over outcomes
            # prop_factor for (q1_layer+1, w) depends on w, approximate with
            # the "generic unknown node" prop factor at that layer
            chain_pf = compute_prop_factor(q1_layer + 1, 0)  # approximate
            if q1_layer + 1 < D - 1:
                chain_R2 = chain_pf * chain_pf * layer_var_cache[q1_layer + 2]
            else:
                mean_term = float(np.mean(layer_vals[D - 1]))
                chain_R2 = chain_pf * chain_pf * (1.0 - mean_term * mean_term)
            if chain_R2 > best_R2:
                best_R2 = chain_R2

        return 2.0 * R1 + best_R2

    def two_step_score_L0(node, q2_candidates):
        """Fast 2-step score for an L0 query using scalar propagation.
        All L0 queries have the same step-1 reduction.
        Step-2 depends on which L1 node we land on (chain continuation)."""
        lv1 = layer_vals[1]
        mean_L1 = float(np.mean(lv1))
        var_L1 = float(np.var(lv1))
        W1 = W[1]

        # Step 1: E[(Δest)²] = var(layer_vals[1]) / W0²
        R1 = var_L1 / (W0 * W0)

        # For each L1 outcome w, compute best q2 reduction
        # Count known L0 transitions to each L1 node
        n_queried_L0 = int(np.sum(g_queried[:W0]))
        n_unk_L0 = W0 - n_queried_L0

        # Pre-compute: R2 for another L0 query = var_L1 / W0² (same as R1)
        R2_L0 = R1

        total_best_R2 = 0.0
        for w in range(W1):
            # After querying L0 node → L1 node w:
            # Chain continuation = query (1, w)
            if not g_queried[fi(1, w)]:
                # How many L0 nodes point to w after this query?
                # Existing known transitions to w + 1 (this query)
                n_to_w = 1  # the query we just made
                # Check if the first trace also points to w
                for v in range(W0):
                    if g_queried[fi(0, v)] and v != node:
                        qid_v = offsets_list[0] + v
                        if is_known[qid_v] and qid_val[qid_v] == w:
                            n_to_w += 1

                if 1 < D - 1:  # layer 1 is non-terminal
                    lv2 = layer_vals[2]
                    var_L2 = float(np.var(lv2))
                    # Propagation factor for (1,w) change to est:
                    # L0 nodes pointing to w: n_to_w, change = Δ
                    # L0 unknown nodes: n_unk_L0 - 1 (one fewer unknown), change = Δ/W1
                    # Actually after querying L0 node, n_unk_L0 decreases by 1
                    n_unk_after = n_unk_L0 - 1
                    prop = (n_to_w + n_unk_after / W1) / W0
                    R2_chain = prop * prop * var_L2
                else:
                    # Layer 1 is terminal (D=2 case)
                    prop = (1 + (n_unk_L0 - 1) / W1) / W0
                    R2_chain = prop * prop * 1.0  # var of ±1

                best_R2 = max(R2_chain, R2_L0)
            else:
                best_R2 = R2_L0

            total_best_R2 += best_R2

        expected_best_R2 = total_best_R2 / W1
        return 2.0 * R1 + expected_best_R2

    # Phase 1: one complete trace
    v0 = int(rng.integers(0, W0))
    cur = v0; qi = 0; mse_sum = F_true_sq

    for layer in range(D):
        do_query(layer, cur); qi += 1
        if layer < D - 1:
            mse_sum += F_true_sq
            cur = int(qid_val[offsets_list[layer] + cur])
        else:
            layer_vals = _init_layer_vals(widths, offsets_arr, D, is_known, qid_val, max_qid)
            est = _estimate_from_layer0(layer_vals[0], W0)
            mse_sum += (est - F_true) ** 2

    if abs(est - F_true) < 1e-15:
        return float(mse_sum)

    _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets_arr,
                                is_known, qid_val, 0)

    lv_flat = np.zeros(total_q, dtype=np.float64)
    for l in range(D):
        a = int(layer_off[l])
        lv_flat[a:a + W[l]] = layer_vals[l]

    # fwd-merge state
    fwd_layer = 0
    fwd_node = None

    def advance_fwd_state():
        nonlocal fwd_layer, fwd_node, l0_idx
        if fwd_node is not None:
            flat = fi(fwd_layer, fwd_node)
            if not g_queried[flat]:
                return
            if fwd_layer < D - 1 and g_succ[flat] >= 0:
                fwd_layer += 1
                fwd_node = int(g_succ[flat])
                return advance_fwd_state()
            else:
                fwd_node = None
        while l0_idx < W0:
            v = l0_perm[l0_idx]
            flat = fi(0, v)
            if not g_queried[flat]:
                fwd_layer = 0
                fwd_node = v
                return
            cur_l, cur_n = 0, v
            while cur_l < D:
                cur_flat = fi(cur_l, cur_n)
                if not g_queried[cur_flat]:
                    fwd_layer = cur_l
                    fwd_node = cur_n
                    return
                if cur_l < D - 1 and g_succ[cur_flat] >= 0:
                    cur_n = int(g_succ[cur_flat])
                    cur_l += 1
                else:
                    break
            l0_idx += 1
        fwd_node = None

    advance_fwd_state()

    # Phase 2: fwd-merge with variance-lookahead deviation gating
    while qi < total_q:
        update_layer_var_cache()

        # Heuristic scores for pre-filtering
        all_vars = _compute_layer_vars(lv_flat, widths, D, layer_off)
        var_score = np.empty(D, dtype=np.float64)
        var_score[:D - 1] = all_vars[1:D]
        var_score[D - 1] = 1.0
        heuristic = exp_reach * var_score[flat_layer]
        heuristic[g_queried] = 0.0
        heuristic[g_reach == 0] = 0.0

        top_k = np.argsort(heuristic)[-K:]
        top_k = top_k[heuristic[top_k] > 0]

        if len(top_k) == 0:
            mse_sum += (total_q - qi) * (est - F_true) ** 2
            break

        # Compute greedy's best pick by 2-step variance reduction
        q2_cands = [int(c) for c in top_k]
        best_greedy_score = -1.0
        best_greedy_flat = int(top_k[-1])
        h_max = max(float(np.max(heuristic[top_k])), 1e-30)

        for q1_flat in top_k:
            q1_flat = int(q1_flat)
            score = two_step_score(q1_flat, q2_cands)
            score += lam * heuristic[q1_flat] / h_max
            if score > best_greedy_score:
                best_greedy_score = score
                best_greedy_flat = q1_flat

        # Compute fwd-merge's score
        if fwd_node is not None:
            fwd_flat = fi(fwd_layer, fwd_node)
            fwd_score = two_step_score(fwd_flat, q2_cands)
            fwd_score += lam * heuristic[fwd_flat] / h_max
        else:
            fwd_score = -1.0

        # Threshold decision: deviate only if greedy is sufficiently better
        if fwd_node is not None and (threshold >= 1e6 or
                best_greedy_score < fwd_score * (1.0 + threshold)):
            best_layer = fwd_layer
            best_node = fwd_node
        else:
            best_layer = int(flat_layer[best_greedy_flat])
            best_node = best_greedy_flat - layer_off[best_layer]

        changed = do_query(best_layer, best_node)
        qi += 1

        _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets_arr,
                                    is_known, qid_val, changed)
        _update_layer_vals(widths, offsets_arr, D, is_known, qid_val, max_qid,
                          layer_vals, changed)
        est = _estimate_from_layer0(layer_vals[0], W0)
        mse_sum += (est - F_true) ** 2

        for l in range(changed + 1):
            a = int(layer_off[l])
            lv_flat[a:a + W[l]] = layer_vals[l]

        if abs(est - F_true) < 1e-15:
            break

        advance_fwd_state()

    return float(mse_sum)


def generate_seeds(n):
    rng = np.random.default_rng(42)
    return [(int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
            for _ in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.01, help="Heuristic bonus weight")
    parser.add_argument("--threshold", type=float, nargs='+', default=[0.0],
                        help="Deviation thresholds (space-separated, runs each)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fwd-file", type=str, default=None)
    args = parser.parse_args()

    D = args.d; n = args.n; K = args.K; lam = args.lam
    thresholds = args.threshold
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))

    print(f"D={D}, n={n}, K={K}, λ={lam}, workers={n_workers}")
    print(f"Thresholds: {thresholds}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    evaluate_lookahead2_var((W, ol, D, 42, 42, K, lam, 0.5))
    print("done.\n")

    seeds = generate_seeds(n)

    # fwd-merge baseline (once)
    if args.fwd_file:
        fwd_raw = np.load(args.fwd_file)[:n]
        fwd = fwd_raw / k
        print(f"Loaded fwd baseline: ratio={np.mean(fwd):.5f}±{se(fwd):.4f}\n")
    else:
        fwd_args = [(W, ol, D, s[0], s[1]) for s in seeds]
        print("Running fwd-merge...", end=" ", flush=True)
        t0 = time.monotonic()
        with mp.Pool(n_workers) as pool:
            fwd_raw = pool.map(evaluate_fwd_trial, fwd_args)
        fwd_dt = time.monotonic() - t0
        fwd = np.array(fwd_raw) / k
        print(f"{fwd_dt:.1f}s  ratio={np.mean(fwd):.5f}±{se(fwd):.4f}\n")

    # Run each threshold
    for threshold in thresholds:
        label = f"t={threshold:.2f}" if threshold < 1e6 else "fwd-merge"
        la_args = [(W, ol, D, s[0], s[1], K, lam, threshold) for s in seeds]
        print(f"Running {label}...", end=" ", flush=True)
        t0 = time.monotonic()
        with mp.Pool(n_workers) as pool:
            la_raw = pool.map(evaluate_lookahead2_var, la_args)
        dt = time.monotonic() - t0
        la = np.array(la_raw) / k

        diff = la - fwd
        mean_diff = np.mean(diff)
        se_diff = se(diff)
        sigma = mean_diff / se_diff if se_diff > 0 else 0
        star = "***" if sigma < -3 else "** " if sigma < -2 else "   "

        print(f"{dt:.0f}s  ratio={np.mean(la):.5f}  "
              f"Δ={mean_diff:+.5f}±{se_diff:.5f} ({sigma:+.1f}σ) {star}")
