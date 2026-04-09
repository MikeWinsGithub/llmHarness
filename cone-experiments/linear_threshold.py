#!/usr/bin/env python3
"""Linear threshold greedy: threshold = x0 + x1 * t, where t = fraction of L0 inputs traced.
Deviates from fwd-merge only if greedy score >= (1 + threshold) * fwd_score.

Usage:
  python3 linear_threshold.py --d 40 --n 30000 --x0 0.4 --x1 -0.3 --workers 14
"""

import numpy as np
from numba import njit
import time
import multiprocessing as mp
import argparse
import os
import json

from run_cones import (
    oracle_query_single, compute_F_true,
    _init_layer_vals, _update_layer_vals, _estimate_from_layer0,
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


def evaluate_linear_threshold(args):
    widths_list, offsets_list, D, oracle_seed, strategy_seed, x0, x1 = args
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

    tiebreak = np.zeros(total_q, dtype=np.float64)
    tiebreak[:W0] = rng.permutation(W0).astype(np.float64) * 1e-15
    g_reach[:W0] = 1.0

    exp_reach = np.zeros(total_q, dtype=np.float64)
    exp_reach[:W0] = 1.0

    n_l0_traced = 0

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

    # Phase 1: trace the first L0 input
    v0 = l0_perm[l0_idx]; l0_idx += 1; n_l0_traced += 1
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
        return float(mse_sum), 0, 0, 0

    _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets_arr,
                                is_known, qid_val, 0)

    lv_flat = np.zeros(total_q, dtype=np.float64)
    for l in range(D):
        a = int(layer_off[l])
        lv_flat[a:a + W[l]] = layer_vals[l]

    _score_buf = np.empty(total_q, dtype=np.float64)

    fwd_layer = 0
    fwd_node = None
    n_fwd_like = 0
    n_deviated = 0
    n_phase2 = 0

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

    while qi < total_q:
        # Compute current threshold based on L0 progress
        t = n_l0_traced / W0
        threshold = max(x0 + x1 * t, 0.0)  # clamp to non-negative

        all_vars = _compute_layer_vars(lv_flat, widths, D, layer_off)
        var_score = np.empty(D, dtype=np.float64)
        var_score[:D - 1] = all_vars[1:D]
        var_score[D - 1] = 1.0

        np.multiply(exp_reach, var_score[flat_layer], out=_score_buf)
        _score_buf[g_queried] = 0.0
        _score_buf[g_reach == 0] = 0.0
        _score_buf += tiebreak

        flat_best_greedy = int(np.argmax(_score_buf))
        if _score_buf[flat_best_greedy] <= 0:
            mse_sum += (total_q - qi) * (est - F_true) ** 2
            break

        greedy_score = _score_buf[flat_best_greedy]

        if fwd_node is not None:
            fwd_flat = fi(fwd_layer, fwd_node)
            fwd_score = _score_buf[fwd_flat]
        else:
            fwd_score = 0.0
            fwd_flat = -1

        n_phase2 += 1
        if fwd_flat >= 0 and greedy_score < fwd_score * (1.0 + threshold):
            best_layer = fwd_layer
            best_node = fwd_node
            n_fwd_like += 1
        else:
            best_layer = int(flat_layer[flat_best_greedy])
            best_node = flat_best_greedy - layer_off[best_layer]
            n_deviated += 1

        # Track L0 traces
        if best_layer == 0 and not g_queried[fi(0, best_node)]:
            n_l0_traced += 1

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

    return float(mse_sum), n_fwd_like, n_deviated, n_phase2


def _worker(args):
    return evaluate_linear_threshold(args)


def generate_seeds(n):
    """Generate deterministic (oracle_seed, strategy_seed) pairs."""
    rng = np.random.default_rng(42)
    return [(int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
            for _ in range(n)]


def compute_fwd_baseline(D, n, n_workers):
    """Compute fwd-merge baseline and save to fwd_baseline.npy."""
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D
    seeds = generate_seeds(n)
    fwd_args = [(W, ol, D, s[0], s[1]) for s in seeds]

    print(f"Computing fwd-merge baseline: D={D}, n={n}, workers={n_workers}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    print("done.")

    print("Running...", end=" ", flush=True)
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        fwd_raw = pool.map(evaluate_fwd_trial, fwd_args)
    dt = time.monotonic() - t0

    fwd = np.array(fwd_raw) / k
    se_val = np.std(fwd, ddof=1) / np.sqrt(len(fwd))
    print(f"{dt:.1f}s  ratio={np.mean(fwd):.5f}±{se_val:.4f}")

    np.save("fwd_baseline.npy", np.array(fwd_raw, dtype=np.float64))
    print(f"Saved to fwd_baseline.npy ({n} trials)")
    return fwd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--x0", type=float, default=None)
    parser.add_argument("--x1", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="Only compute fwd-merge baseline and save to fwd_baseline.npy")
    parser.add_argument("--fwd-file", type=str, default=None,
                        help="Load precomputed fwd-merge baseline from .npy file")
    args = parser.parse_args()

    D = args.d; n = args.n
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))

    if args.baseline:
        compute_fwd_baseline(D, n, n_workers)
        exit()

    if args.x0 is None or args.x1 is None:
        parser.error("--x0 and --x1 are required (unless using --baseline)")

    x0 = args.x0; x1 = args.x1

    label = f"x0={x0:.2f}, x1={x1:+.2f}"
    print(f"D={D}, n={n}, workers={n_workers}, {label}")
    print(f"  threshold at t=0: {max(x0, 0):.2f}, t=0.5: {max(x0+0.5*x1, 0):.2f}, t=1.0: {max(x0+x1, 0):.2f}")
    print(f"Funnel widths: [{W[0]}, ..., {W[-1]}]  total_q={sum(W)}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    evaluate_linear_threshold((W, ol, D, 42, 42, x0, x1))
    print("done.\n")

    seeds = generate_seeds(n)
    base_args = [(W, ol, D, s[0], s[1], x0, x1) for s in seeds]

    # Load or compute fwd-merge baseline
    if args.fwd_file:
        fwd_raw = np.load(args.fwd_file)
        fwd = fwd_raw / k
        print(f"Loaded fwd baseline from {args.fwd_file}: ratio={np.mean(fwd):.5f}±{se(fwd):.4f}")
    else:
        fwd_args = [(W, ol, D, s[0], s[1]) for s in seeds]
        print("Running fwd-merge baseline...", end=" ", flush=True)
        t0 = time.monotonic()
        with mp.Pool(n_workers) as pool:
            fwd_raw_list = pool.map(evaluate_fwd_trial, fwd_args)
        fwd_dt = time.monotonic() - t0
        fwd = np.array(fwd_raw_list) / k
        print(f"{fwd_dt:.1f}s  ratio={np.mean(fwd):.5f}±{se(fwd):.4f}")

    print(f"Running {label}...", end=" ", flush=True)
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker, base_args)
    dt = time.monotonic() - t0

    mse_vals = np.array([r[0] for r in results]) / k
    total_fwd_like = sum(r[1] for r in results)
    total_deviated = sum(r[2] for r in results)
    total_p2 = sum(r[3] for r in results)
    print(f"{dt:.1f}s")

    diff = mse_vals - fwd
    mean_diff = np.mean(diff)
    se_diff = se(diff)
    sigma = mean_diff / se_diff if se_diff > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    print(f"  ratio:  {np.mean(mse_vals):.5f} ± {se(mse_vals):.4f}")
    print(f"  vs fwd: {mean_diff:+.5f} ± {se_diff:.5f}  ({sigma:+.1f}σ)")

    if total_p2 > 0:
        pct_fwd = 100 * total_fwd_like / total_p2
        pct_dev = 100 * total_deviated / total_p2
        print(f"\n  Phase 2 decisions: {total_p2}")
        print(f"    fwd-like:  {total_fwd_like:>10d}  ({pct_fwd:5.1f}%)")
        print(f"    deviated:  {total_deviated:>10d}  ({pct_dev:5.1f}%)")

    result = {
        "D": D, "n": n, "x0": x0, "x1": x1, "label": label,
        "ratio_mean": float(np.mean(mse_vals)),
        "ratio_se": float(se(mse_vals)),
        "vs_fwd_mean": float(mean_diff),
        "vs_fwd_se": float(se_diff),
        "vs_fwd_sigma": float(sigma),
        "fwd_like_pct": float(pct_fwd) if total_p2 > 0 else None,
        "deviated_pct": float(pct_dev) if total_p2 > 0 else None,
        "threshold_at_0": max(x0, 0),
        "threshold_at_half": max(x0 + 0.5 * x1, 0),
        "threshold_at_1": max(x0 + x1, 0),
    }
    fname = f"result_x0_{x0:.2f}_x1_{x1:+.2f}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {fname}")
