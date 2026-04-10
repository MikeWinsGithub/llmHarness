#!/usr/bin/env python3
"""2-step lookahead greedy: at each step, pick the query that minimizes
MSE_after_q1 + min_q2(MSE_after_q1_and_q2).

Pre-filters to top-K candidates using heuristic score, then does exact
2-step evaluation on those K candidates.
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
    _compute_terminal_val, _compute_layer_val,
    evaluate_single_trial as evaluate_fwd_trial,
    funnel, warmup,
)

@njit
def _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets,
                                is_known, qid_val, start_layer):
    for layer in range(start_layer, D - 1):
        a = layer_off[layer]
        W_l = widths[layer]
        W_next = widths[layer + 1]
        qid_start = offsets[layer]
        na = layer_off[layer + 1]
        for u in range(W_next):
            exp_reach[na + u] = 0.0
        unknown_total = 0.0
        for v in range(W_l):
            r = exp_reach[a + v]
            if r == 0.0:
                continue
            qid = qid_start + v
            if is_known[qid]:
                dest = qid_val[qid]
                exp_reach[na + dest] += r
            else:
                unknown_total += r
        if unknown_total > 0.0:
            uniform = unknown_total / W_next
            for u in range(W_next):
                exp_reach[na + u] += uniform

@njit
def _compute_layer_vars(layer_vals_flat, widths, D, layer_off):
    var_buf = np.empty(D, dtype=np.float64)
    for l in range(D):
        a = layer_off[l]
        W_l = widths[l]
        s = 0.0
        s2 = 0.0
        for v in range(W_l):
            val = layer_vals_flat[a + v]
            s += val
            s2 += val * val
        mean = s / W_l
        var_buf[l] = s2 / W_l - mean * mean
    return var_buf


def hypothetical_query_mse(widths, offsets_arr, D, is_known, qid_val, max_qid,
                           layer_vals, layer, node, oracle_seed, offsets_list, W, F_true):
    """Apply a query hypothetically, return (mse, oracle_value).
    Modifies is_known, qid_val, layer_vals in place. Caller must save/restore."""
    W0 = W[0]
    qid = offsets_list[layer] + node
    nv = W[layer + 1] if layer < D - 1 else 2
    val = oracle_query_single(oracle_seed, qid, nv)

    is_known[qid] = True
    qid_val[qid] = val

    _update_layer_vals(widths, offsets_arr, D, is_known, qid_val, max_qid, layer_vals, layer)
    est = _estimate_from_layer0(layer_vals[0], W0)
    return (est - F_true) ** 2, val


def evaluate_lookahead2(args):
    widths_list, offsets_list, D, oracle_seed, strategy_seed, K = args
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

    # Phase 2: 2-step lookahead
    while qi < total_q:
        # Compute heuristic scores for pre-filtering
        all_vars = _compute_layer_vars(lv_flat, widths, D, layer_off)
        var_score = np.empty(D, dtype=np.float64)
        var_score[:D - 1] = all_vars[1:D]
        var_score[D - 1] = 1.0

        scores = exp_reach * var_score[flat_layer]
        scores[g_queried] = 0.0
        scores[g_reach == 0] = 0.0

        # Top-K candidates by heuristic
        top_k = np.argsort(scores)[-K:]
        top_k = top_k[scores[top_k] > 0]

        if len(top_k) == 0:
            mse_sum += (total_q - qi) * (est - F_true) ** 2
            break

        # If only 1 candidate or last query, just pick the best
        if len(top_k) == 1 or qi >= total_q - 1:
            best_flat = top_k[-1]
        else:
            # 2-step lookahead: for each q1, find best q2, minimize mse1 + mse2
            best_total = float('inf')
            best_flat = top_k[-1]  # fallback to 1-step greedy

            for q1_flat in top_k:
                q1_flat = int(q1_flat)
                q1_layer = int(flat_layer[q1_flat])
                q1_node = q1_flat - layer_off[q1_layer]
                qid1 = offsets_list[q1_layer] + q1_node

                # Save state before q1
                saved_ik1 = bool(is_known[qid1])
                saved_qv1 = int(qid_val[qid1])
                saved_lv1 = [layer_vals[l].copy() for l in range(D)]

                # Apply q1
                mse1, q1_val = hypothetical_query_mse(
                    widths, offsets_arr, D, is_known, qid_val, max_qid,
                    layer_vals, q1_layer, q1_node, oracle_seed, offsets_list, W, F_true)

                # Build q2 candidates: original top_k + chain continuation from q1
                q2_candidates = []
                for c in top_k:
                    c = int(c)
                    if c != q1_flat:
                        q2_candidates.append(c)
                # Add chain continuation: q1's successor if it's unqueried
                if q1_layer < D - 1:
                    succ_node = int(q1_val)
                    succ_flat = int(layer_off[q1_layer + 1] + succ_node)
                    if not g_queried[succ_flat] and succ_flat != q1_flat:
                        q2_candidates.append(succ_flat)

                # Find best q2
                best_mse2 = float('inf')
                for q2_flat in q2_candidates:
                    q2_layer = int(flat_layer[q2_flat])
                    q2_node = q2_flat - layer_off[q2_layer]
                    qid2 = offsets_list[q2_layer] + q2_node

                    if qid2 == qid1:
                        continue

                    # Save state before q2 (post-q1 state)
                    saved_ik2 = bool(is_known[qid2])
                    saved_qv2 = int(qid_val[qid2])
                    saved_lv2 = [layer_vals[l].copy() for l in range(D)]

                    # Apply q2
                    mse2, _ = hypothetical_query_mse(
                        widths, offsets_arr, D, is_known, qid_val, max_qid,
                        layer_vals, q2_layer, q2_node, oracle_seed, offsets_list, W, F_true)

                    if mse2 < best_mse2:
                        best_mse2 = mse2

                    # Restore post-q1 state
                    for l in range(D):
                        layer_vals[l] = saved_lv2[l]
                    is_known[qid2] = saved_ik2
                    qid_val[qid2] = saved_qv2

                total = mse1 + best_mse2
                if total < best_total:
                    best_total = total
                    best_flat = q1_flat

                # Restore pre-q1 state
                for l in range(D):
                    layer_vals[l] = saved_lv1[l]
                is_known[qid1] = saved_ik1
                qid_val[qid1] = saved_qv1

        # Actually execute the chosen query
        best_layer = int(flat_layer[best_flat])
        best_node = best_flat - layer_off[best_layer]

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

    return float(mse_sum)


def generate_seeds(n):
    rng = np.random.default_rng(42)
    return [(int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
            for _ in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--K", type=int, default=10, help="Number of lookahead candidates")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fwd-file", type=str, default=None)
    args = parser.parse_args()

    D = args.d; n = args.n; K = args.K
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))

    print(f"D={D}, n={n}, K={K}, workers={n_workers}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    # Warm up lookahead
    t0 = time.monotonic()
    evaluate_lookahead2((W, ol, D, 42, 42, K))
    warmup_time = time.monotonic() - t0
    print(f"done. (1 trial = {warmup_time:.1f}s)\n")

    estimated = warmup_time * n / n_workers
    print(f"Estimated total time: {estimated:.0f}s ({estimated/60:.1f} min)\n")

    seeds = generate_seeds(n)

    # fwd-merge baseline
    if args.fwd_file:
        fwd_raw = np.load(args.fwd_file)[:n]
        fwd = fwd_raw / k
        print(f"Loaded fwd baseline: ratio={np.mean(fwd):.5f}±{se(fwd):.4f}")
    else:
        fwd_args = [(W, ol, D, s[0], s[1]) for s in seeds]
        print("Running fwd-merge...", end=" ", flush=True)
        t0 = time.monotonic()
        with mp.Pool(n_workers) as pool:
            fwd_raw = pool.map(evaluate_fwd_trial, fwd_args)
        fwd_dt = time.monotonic() - t0
        fwd = np.array(fwd_raw) / k
        print(f"{fwd_dt:.1f}s  ratio={np.mean(fwd):.5f}±{se(fwd):.4f}")

    # 2-step lookahead
    la_args = [(W, ol, D, s[0], s[1], K) for s in seeds]
    print(f"\nRunning 2-step lookahead (K={K})...", flush=True)
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        la_raw = list(pool.imap(evaluate_lookahead2, la_args, chunksize=1))
        # Print progress
    dt = time.monotonic() - t0
    la = np.array(la_raw) / k
    print(f"  {dt:.1f}s ({dt/n*1000:.0f}ms/trial)")

    diff = la - fwd
    mean_diff = np.mean(diff)
    se_diff = se(diff)
    sigma = mean_diff / se_diff if se_diff > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: 2-step lookahead (K={K})")
    print(f"{'='*60}")
    print(f"  fwd-merge:     {np.mean(fwd):.5f} ± {se(fwd):.4f}")
    print(f"  2-step:        {np.mean(la):.5f} ± {se(la):.4f}")
    print(f"  Paired diff:   {mean_diff:+.5f} ± {se_diff:.5f}  ({sigma:+.1f}σ)")
    if abs(sigma) >= 2:
        winner = "fwd-merge" if mean_diff > 0 else "2-step lookahead"
        print(f"  ** {winner} is significantly better **")
    else:
        print(f"  Not significant")
