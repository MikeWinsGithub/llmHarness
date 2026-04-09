#!/usr/bin/env python3
"""Optimized reached-weighted-greedy with:
1. Incremental forward DP (only recompute from changed layer onward)
2. Numba-compiled variance computation
3. In-place score buffer"""

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


@njit
def _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets,
                                is_known, qid_val, start_layer):
    """Recompute expected reach from start_layer onward, in place.
    Layers 0..start_layer-1 are assumed correct in exp_reach."""
    for layer in range(start_layer, D - 1):
        a = layer_off[layer]
        W_l = widths[layer]
        W_next = widths[layer + 1]
        qid_start = offsets[layer]
        na = layer_off[layer + 1]

        # Zero out next layer
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
    """Compute variance of node values at each layer. Returns D-element array."""
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


def evaluate_reached_weighted_fast(args):
    widths_list, offsets_list, D, oracle_seed, strategy_seed = args
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
    n_known = np.zeros(D, dtype=np.int64)

    rng = np.random.default_rng(strategy_seed)
    tiebreak = np.zeros(total_q, dtype=np.float64)
    tiebreak[:W0] = rng.permutation(W0).astype(np.float64) * 1e-15
    g_reach[:W0] = 1.0

    # Persistent expected reach buffer (updated incrementally)
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
            n_known[flat_layer[idx]] += 1
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
            n_known[D - 1] += 1
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

    # Phase 1
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

    # Initialize expected reach after phase 1
    _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets_arr,
                                is_known, qid_val, 0)

    # Flatten layer_vals for numba var computation
    lv_flat = np.zeros(total_q, dtype=np.float64)
    for l in range(D):
        a = int(layer_off[l])
        lv_flat[a:a + W[l]] = layer_vals[l]

    _score_buf = np.empty(total_q, dtype=np.float64)

    # Phase 2
    while qi < total_q:
        # Variance per layer (numba, uses shifted index: var for scoring layer l
        # needs var of layer l+1)
        all_vars = _compute_layer_vars(lv_flat, widths, D, layer_off)
        # var_buf[l] = variance to use when scoring nodes at layer l
        #            = var(layer_vals[l+1]) for l < D-1, 1.0 for terminal
        var_score = np.empty(D, dtype=np.float64)
        var_score[:D - 1] = all_vars[1:D]
        var_score[D - 1] = 1.0

        # Score = expected_reach * var_next, filtered by reached + unqueried
        np.multiply(exp_reach, var_score[flat_layer], out=_score_buf)
        _score_buf[g_queried] = 0.0
        _score_buf[g_reach == 0] = 0.0
        _score_buf += tiebreak

        flat_best = int(np.argmax(_score_buf))
        if _score_buf[flat_best] <= 0:
            mse_sum += (total_q - qi) * (est - F_true) ** 2
            break

        best_layer = int(flat_layer[flat_best])
        best_node = flat_best - layer_off[best_layer]

        changed = do_query(best_layer, best_node)
        qi += 1

        # Incremental expected reach update (only from changed layer onward)
        _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets_arr,
                                    is_known, qid_val, changed)

        _update_layer_vals(widths, offsets_arr, D, is_known, qid_val, max_qid,
                          layer_vals, changed)
        est = _estimate_from_layer0(layer_vals[0], W0)
        mse_sum += (est - F_true) ** 2

        # Update flat layer_vals (only changed layers: 0..changed)
        for l in range(changed + 1):
            a = int(layer_off[l])
            lv_flat[a:a + W[l]] = layer_vals[l]

        if abs(est - F_true) < 1e-15:
            break

    return float(mse_sum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    D = args.d; n = args.n
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    print(f"D={D}, n={n}, workers={n_workers}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    evaluate_reached_weighted_fast((W, ol, D, 42, 42))
    print("done.\n")

    rng = np.random.default_rng(42)
    base_args = [(W, ol, D, int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
                 for _ in range(n)]

    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))

    # fwd-merge
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        fwd_raw = pool.map(evaluate_fwd_trial, base_args)
    fwd_dt = time.monotonic() - t0
    fwd = np.array(fwd_raw) / k
    print(f"  fwd-merge:              {np.mean(fwd):.5f} ± {se(fwd):.4f}  ({fwd_dt:.1f}s, {fwd_dt/n*1000:.1f}ms/tr)")

    # fast reached-weighted
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        rw_raw = pool.map(evaluate_reached_weighted_fast, base_args)
    rw_dt = time.monotonic() - t0
    rw = np.array(rw_raw) / k
    diff = rw - fwd
    sigma = np.mean(diff) / se(diff) if se(diff) > 0 else 0
    print(f"  reached-weighted-fast:  {np.mean(rw):.5f} ± {se(rw):.4f}  ({rw_dt:.1f}s, {rw_dt/n*1000:.1f}ms/tr)"
          f"  Δfwd={np.mean(diff):+.4f}±{se(diff):.4f} ({sigma:+.1f}σ)")
