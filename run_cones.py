#!/usr/bin/env python3
"""Standalone cone experiment runner.

Usage:
    python run_cones.py                          # default sweep
    python run_cones.py --dmin 5 --dmax 100      # custom range
    python run_cones.py --d 20                   # single D
    python run_cones.py --minutes 60             # time budget
    python run_cones.py --workers 10             # use 10 cores

Requirements:
    pip install numpy numba
"""

import numpy as np
from numba import njit
import time
import multiprocessing as mp
import argparse
import os

# ---------------------------------------------------------------------------
# Numba-JIT'd core functions
# ---------------------------------------------------------------------------

@njit
def oracle_query_single(seed, qid, num_values):
    h = np.uint64(seed) * np.uint64(6364136223846793005) + np.uint64(qid) * np.uint64(1442695040888963407)
    h = (h >> np.uint64(16)) ^ h
    h *= np.uint64(2685821657736338717)
    return int(h % np.uint64(num_values))

@njit
def compute_F_true(widths, offsets, D, N, seed):
    total = 0.0
    for x_int in range(1 << N):
        v = x_int % widths[0]
        for layer in range(D - 1):
            qid = offsets[layer] + v
            v = oracle_query_single(seed, qid, widths[layer + 1])
        qid = offsets[D - 1] + v
        sign_raw = oracle_query_single(seed, qid, 2)
        total += 1.0 if sign_raw == 1 else -1.0
    return total / (1 << N)

@njit
def _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, layer, next_layer_val):
    W = widths[layer]
    W_next = widths[layer + 1]
    val = np.zeros(W, dtype=np.float64)
    for v in range(W):
        qid = offsets[layer] + v
        if qid < max_qid and is_known[qid]:
            val[v] = next_layer_val[qid_val[qid]]
        else:
            total = 0.0
            for nv in range(W_next):
                total += next_layer_val[nv]
            val[v] = total / W_next
    return val

@njit
def _compute_terminal_val(widths, offsets, D, is_known, qid_val, max_qid):
    W = widths[D - 1]
    val = np.zeros(W, dtype=np.float64)
    for v in range(W):
        qid = offsets[D - 1] + v
        if qid < max_qid and is_known[qid]:
            val[v] = 1.0 if qid_val[qid] == 1 else -1.0
    return val

@njit
def _estimate_from_layer0(layer0_val, W0, N):
    total_F = 0.0
    count_per = (1 << N) // W0
    remainder = (1 << N) % W0
    for v0 in range(W0):
        c = count_per + (1 if v0 < remainder else 0)
        total_F += layer0_val[v0] * c
    return total_F / (1 << N)

# ---------------------------------------------------------------------------
# Layer vals (Python-level, calls njit functions)
# ---------------------------------------------------------------------------

def _init_layer_vals(widths, offsets, D, is_known, qid_val, max_qid):
    layer_vals = [None] * D
    layer_vals[D - 1] = _compute_terminal_val(widths, offsets, D, is_known, qid_val, max_qid)
    for layer in range(D - 2, -1, -1):
        layer_vals[layer] = _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, layer, layer_vals[layer + 1])
    return layer_vals

def _update_layer_vals(widths, offsets, D, is_known, qid_val, max_qid, layer_vals, changed_layer):
    if changed_layer == D - 1:
        layer_vals[D - 1] = _compute_terminal_val(widths, offsets, D, is_known, qid_val, max_qid)
    else:
        layer_vals[changed_layer] = _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, changed_layer, layer_vals[changed_layer + 1])
    for layer in range(changed_layer - 1, -1, -1):
        layer_vals[layer] = _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, layer, layer_vals[layer + 1])

# ---------------------------------------------------------------------------
# Query order: forward sample-merge
# ---------------------------------------------------------------------------

def fwd_merge_queries(widths, offsets, D, N, seed, oracle_seed):
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    for x_int in rng.permutation(1 << N):
        v = int(x_int) % widths[0]
        for layer in range(D - 1):
            qid = offsets[layer] + v
            nv = widths[layer + 1]
            if qid not in known:
                order.append((qid, layer, nv))
            if qid not in known:
                known[qid] = oracle_query_single(oracle_seed, qid, nv)
            v = known[qid]
        qid = offsets[D - 1] + v
        if qid not in known:
            order.append((qid, D - 1, 2))
            known[qid] = oracle_query_single(oracle_seed, qid, 2)
    return order

# ---------------------------------------------------------------------------
# Single trial evaluation
# ---------------------------------------------------------------------------

def evaluate_single_trial(args):
    widths_list, offsets_list, D, N, oracle_seed, strategy_seed = args
    widths = np.array(widths_list, dtype=np.int32)
    offsets = np.array(offsets_list, dtype=np.int32)
    total_q = sum(widths_list)
    max_qid = offsets[-1] + widths[-1]

    F_true = compute_F_true(widths, offsets, D, N, oracle_seed)
    query_order = fwd_merge_queries(widths_list, offsets_list, D, N, strategy_seed, oracle_seed)

    is_known = np.zeros(max_qid, dtype=np.bool_)
    qid_val = np.zeros(max_qid, dtype=np.int32)
    mse_list = np.zeros(total_q + 1)

    F_true_sq = F_true * F_true
    est = 0.0
    layer_vals = None
    terminal_known = False

    qi = 0
    for qid, layer_idx, nv in query_order:
        val = oracle_query_single(oracle_seed, qid, nv)
        is_known[qid] = True
        qid_val[qid] = val
        qi += 1

        if not terminal_known:
            if layer_idx == D - 1:
                terminal_known = True
                layer_vals = _init_layer_vals(widths, offsets, D, is_known, qid_val, max_qid)
                est = _estimate_from_layer0(layer_vals[0], widths[0], N)
            else:
                if qi < len(mse_list):
                    mse_list[qi] = F_true_sq
                continue
        else:
            _update_layer_vals(widths, offsets, D, is_known, qid_val, max_qid, layer_vals, layer_idx)
            est = _estimate_from_layer0(layer_vals[0], widths[0], N)

        mse = (est - F_true) ** 2
        if qi < len(mse_list):
            mse_list[qi] = mse
        if abs(est - F_true) < 1e-15:
            break

    mse_list[0] = F_true_sq
    final_mse = (est - F_true) ** 2 if terminal_known else F_true_sq
    for j in range(qi + 1, total_q + 1):
        mse_list[j] = final_mse

    return float(np.sum(mse_list))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def funnel(D):
    return list(range(2 * D, D, -1))

def warmup():
    w = np.array([4, 4], dtype=np.int32)
    o = np.array([0, 4], dtype=np.int32)
    ik = np.zeros(8, dtype=np.bool_)
    iv = np.zeros(8, dtype=np.int32)
    compute_F_true(w, o, 2, 4, 42)
    _compute_terminal_val(w, o, 2, ik, iv, 8)
    _compute_layer_val(w, o, 2, ik, iv, 8, 0, np.zeros(4))
    _estimate_from_layer0(np.zeros(4), 4, 8)
    oracle_query_single(42, 0, 2)
    # Run one trial to compile everything
    evaluate_single_trial(([4, 4], [0, 4], 2, 4, 42, 42))

def run_D(D, n_samples, n_workers, N=8):
    w = funnel(D)
    ol = [sum(w[:i]) for i in range(D)]
    k = D

    rng = np.random.default_rng(42)
    args = [(w, ol, D, N, int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
            for _ in range(n_samples)]

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(evaluate_single_trial, args)
    else:
        results = [evaluate_single_trial(a) for a in args]

    cumuls = np.array(results) / k
    mean = float(np.mean(cumuls))
    stderr = float(np.std(cumuls, ddof=1)) / np.sqrt(len(cumuls))
    return mean, stderr, len(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cone conjecture experiments")
    parser.add_argument("--dmin", type=int, default=5, help="Minimum D")
    parser.add_argument("--dmax", type=int, default=100, help="Maximum D")
    parser.add_argument("--dstep", type=int, default=5, help="D step size")
    parser.add_argument("--d", type=int, default=None, help="Single D value (overrides dmin/dmax)")
    parser.add_argument("--minutes", type=float, default=20, help="Wall-clock time budget in minutes")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    parser.add_argument("--N", type=int, default=8, help="Input dimension (default: 8)")
    args = parser.parse_args()

    n_workers = args.workers or os.cpu_count() or 4
    wall_limit = args.minutes * 60

    print(f"Cores: {n_workers}, Time budget: {args.minutes} min, N={args.N}")
    print("Warming up JIT...", end=" ", flush=True)
    warmup()
    print("done.\n")

    if args.d is not None:
        D_values = [args.d]
    else:
        D_values = list(range(args.dmin, args.dmax + 1, args.dstep))

    print(f"{'D':>3s}  {'k':>3s}  {'ms/tr':>6s}  {'n':>7s}  {'ratio':>10s}  {'±stderr':>8s}  {'':>2s}  {'wall':>6s}")
    print("-" * 55)

    global_start = time.monotonic()

    for D in D_values:
        elapsed = time.monotonic() - global_start
        remaining = wall_limit - elapsed
        if remaining < 5:
            print("(time limit reached)")
            break

        w = funnel(D)
        ol = [sum(w[:i]) for i in range(D)]

        # Measure ms/trial
        t0 = time.monotonic()
        for _ in range(3):
            evaluate_single_trial((w, ol, D, args.N, 12345, 42))
        ms_per = (time.monotonic() - t0) / 3 * 1000

        # Budget samples: spend proportional time, min 500 max 50000
        time_budget = min(remaining * 0.4, 120)
        n = int(time_budget * n_workers / (ms_per / 1000))
        n = max(500, min(50000, n))

        t0 = time.monotonic()
        mean, stderr, trials = run_D(D, n, n_workers, args.N)
        wall = time.monotonic() - t0
        total_elapsed = time.monotonic() - global_start

        s = '✓' if mean <= 1 + 1e-6 else '✗'
        sigma_over = (mean - 1.0) / stderr if stderr > 0 else 0
        print(f"{D:3d}  {D:3d}  {ms_per:5.1f}  {trials:7d}  {mean:9.5f}  ±{stderr:.4f}  {s:>2s}  {total_elapsed:5.0f}s  ({sigma_over:+.1f}σ from 1)", flush=True)

    print(f"\nTotal: {time.monotonic() - global_start:.0f}s")
