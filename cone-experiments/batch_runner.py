#!/usr/bin/env python3
"""Batch experiment runner for threshold-greedy heuristics.

Usage:
  # Compute fwd-merge baseline
  python3 batch_runner.py --baseline --d 40 --n 30000 --workers 14

  # Run a batch of experiments (reads from experiments.json, outputs to results_N.json)
  python3 batch_runner.py --batch 0 --num-batches 8 --fwd-file fwd_baseline.npy \
      --d 40 --n 30000 --workers 14
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

# ---------------------------------------------------------------------------
# Numba helpers (same as before)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Unified heuristic evaluator
# ---------------------------------------------------------------------------

def evaluate_heuristic(args):
    widths_list, offsets_list, D, oracle_seed, strategy_seed, config = args
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

    # fwd-merge state
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

    # State for chain_complete heuristic
    deviation_chain_layer = -1
    deviation_chain_node = -1

    # State for tracking last query (for no_jump heuristic)
    prev_query_layer = D - 1
    prev_query_node = cur  # last node queried in Phase 1

    htype = config['type']

    # Phase 2
    while qi < total_q:
        # --- Chain-complete override: if in a deviation chain, follow it ---
        if htype == 'chain_complete' and deviation_chain_layer >= 0:
            best_layer = deviation_chain_layer
            best_node = deviation_chain_node
            n_deviated += 1
            n_phase2 += 1

            changed = do_query(best_layer, best_node)
            qi += 1

            # Update chain state
            flat = fi(best_layer, best_node)
            if best_layer < D - 1 and g_succ[flat] >= 0:
                succ = int(g_succ[flat])
                succ_flat = fi(best_layer + 1, succ)
                if not g_queried[succ_flat]:
                    deviation_chain_layer = best_layer + 1
                    deviation_chain_node = succ
                else:
                    deviation_chain_layer = -1  # merged, chain done
            else:
                deviation_chain_layer = -1  # terminal, chain done

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
            prev_query_layer = best_layer
            prev_query_node = best_node
            advance_fwd_state()
            continue

        # --- Compute greedy scores ---
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

        greedy_layer = int(flat_layer[flat_best_greedy])
        greedy_node = flat_best_greedy - layer_off[greedy_layer]
        greedy_score = _score_buf[flat_best_greedy]

        if fwd_node is not None:
            fwd_flat = fi(fwd_layer, fwd_node)
            fwd_score = _score_buf[fwd_flat]
        else:
            fwd_score = 0.0
            fwd_flat = -1

        # --- Apply heuristic decision rule ---
        t = n_l0_traced / W0
        deviate = False

        if fwd_flat < 0:
            deviate = True  # no fwd option, must use greedy
        elif htype == 'fixed':
            threshold = config['threshold']
            deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'linear':
            threshold = max(config['x0'] + config['x1'] * t, 0.0)
            deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'layer_gated':
            if greedy_layer <= config['l_max']:
                threshold = config['threshold']
                deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'l0_only':
            if greedy_layer == 0:
                threshold = config['threshold']
                deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'no_jump':
            # Allow if greedy picks L0 (new chain) or chain continuation
            prev_flat = fi(prev_query_layer, prev_query_node)
            is_chain_cont = (prev_query_layer < D - 1
                             and g_succ[prev_flat] >= 0
                             and greedy_layer == prev_query_layer + 1
                             and greedy_node == int(g_succ[prev_flat]))
            if greedy_layer == 0 or is_chain_cont:
                threshold = config['threshold']
                deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'depth_weighted':
            threshold = config['base'] + config['depth_weight'] * (greedy_layer / (D - 1))
            deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'switch':
            deviate = n_l0_traced >= config['switch_k']

        elif htype == 'inverse':
            threshold = config['c'] / (1.0 + config['alpha'] * t)
            deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'exp_decay':
            import math
            threshold = config['c'] * math.exp(-config['alpha'] * t)
            deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'chain_complete':
            threshold = config['threshold']
            deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'step':
            threshold = config['high'] if t < config['cutoff'] else config['low']
            deviate = greedy_score >= fwd_score * (1.0 + threshold)

        elif htype == 'quadratic':
            threshold = max(config['x0'] + config['x1'] * t + config['x2'] * t * t, 0.0)
            deviate = greedy_score >= fwd_score * (1.0 + threshold)

        # --- Execute decision ---
        n_phase2 += 1
        if deviate:
            best_layer = greedy_layer
            best_node = greedy_node
            n_deviated += 1

            # Start deviation chain if chain_complete
            if htype == 'chain_complete' and best_layer < D - 1:
                # Chain will be followed in subsequent iterations
                pass  # chain state set after do_query below
        else:
            best_layer = fwd_layer
            best_node = fwd_node
            n_fwd_like += 1

        if best_layer == 0 and not g_queried[fi(0, best_node)]:
            n_l0_traced += 1

        changed = do_query(best_layer, best_node)
        qi += 1

        # Set chain_complete state after query
        if htype == 'chain_complete' and deviate and best_layer < D - 1:
            flat = fi(best_layer, best_node)
            if g_succ[flat] >= 0:
                succ = int(g_succ[flat])
                succ_flat = fi(best_layer + 1, succ)
                if not g_queried[succ_flat]:
                    deviation_chain_layer = best_layer + 1
                    deviation_chain_node = succ

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

        prev_query_layer = best_layer
        prev_query_node = best_node
        advance_fwd_state()

    return float(mse_sum), n_fwd_like, n_deviated, n_phase2


def _worker(args):
    return evaluate_heuristic(args)


# ---------------------------------------------------------------------------
# Experiment generation
# ---------------------------------------------------------------------------

def generate_experiments():
    exps = []

    # 1. Fixed threshold (21 experiments)
    for c in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
              0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0]:
        exps.append({"name": f"fixed_{c:.2f}", "type": "fixed", "threshold": c})

    # 2. Linear in progress (35 experiments)
    for x0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for x1 in [-0.6, -0.4, -0.2, 0.0, 0.2]:
            exps.append({"name": f"linear_{x0:.1f}_{x1:+.1f}",
                         "type": "linear", "x0": x0, "x1": x1})

    # 3. Layer-gated (24 experiments)
    for l_max in [0, 1, 2, 3, 5, 10, 20, 39]:
        for threshold in [0.0, 0.2, 0.4]:
            exps.append({"name": f"layer_gated_L{l_max}_t{threshold:.1f}",
                         "type": "layer_gated", "l_max": l_max, "threshold": threshold})

    # 4. L0-only (8 experiments)
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        exps.append({"name": f"l0_only_{threshold:.1f}",
                     "type": "l0_only", "threshold": threshold})

    # 5. No-jump (8 experiments)
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        exps.append({"name": f"no_jump_{threshold:.1f}",
                     "type": "no_jump", "threshold": threshold})

    # 6. Depth-weighted (16 experiments)
    for base in [0.0, 0.2, 0.4, 0.6]:
        for dw in [0.2, 0.5, 1.0, 2.0]:
            exps.append({"name": f"depth_wt_b{base:.1f}_d{dw:.1f}",
                         "type": "depth_weighted", "base": base, "depth_weight": dw})

    # 7. Switch-at-K (10 experiments)
    for k in [1, 2, 3, 5, 10, 15, 20, 30, 50, 70]:
        exps.append({"name": f"switch_{k}", "type": "switch", "switch_k": k})

    # 8. Inverse decay (12 experiments)
    for c in [0.3, 0.5, 0.7]:
        for alpha in [1, 2, 5, 10]:
            exps.append({"name": f"inverse_c{c:.1f}_a{alpha}",
                         "type": "inverse", "c": c, "alpha": alpha})

    # 9. Exponential decay (9 experiments)
    for c in [0.3, 0.5, 0.7]:
        for alpha in [1, 3, 10]:
            exps.append({"name": f"exp_decay_c{c:.1f}_a{alpha}",
                         "type": "exp_decay", "c": c, "alpha": alpha})

    # 10. Chain-complete (8 experiments)
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        exps.append({"name": f"chain_complete_{threshold:.1f}",
                     "type": "chain_complete", "threshold": threshold})

    # 11. Step function (12 experiments)
    for high, low in [(0.5, 0.0), (0.5, 0.2), (1.0, 0.0), (1.0, 0.3)]:
        for cutoff in [0.1, 0.25, 0.5]:
            exps.append({"name": f"step_h{high:.1f}_l{low:.1f}_c{cutoff:.2f}",
                         "type": "step", "high": high, "low": low, "cutoff": cutoff})

    # 12. Quadratic (15 experiments)
    for x0, x1, x2 in [
        (0.3, -0.3, 0.5), (0.3, 0.0, 0.5), (0.3, 0.0, 1.0),
        (0.4, -0.5, 0.5), (0.4, -0.5, 1.0), (0.4, -0.3, 0.5),
        (0.4, 0.0, 0.5), (0.4, 0.0, 1.0), (0.4, 0.3, -0.3),
        (0.5, -0.5, 0.5), (0.5, -1.0, 1.0), (0.5, 0.0, 0.5),
        (0.2, 0.0, 0.5), (0.2, 0.5, -0.5), (0.6, -0.5, 0.5),
    ]:
        exps.append({"name": f"quad_{x0:.1f}_{x1:+.1f}_{x2:+.1f}",
                     "type": "quadratic", "x0": x0, "x1": x1, "x2": x2})

    return exps


# ---------------------------------------------------------------------------
# Seed generation (deterministic, shared across all experiments)
# ---------------------------------------------------------------------------

def generate_seeds(n):
    rng = np.random.default_rng(42)
    return [(int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
            for _ in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="Compute fwd-merge baseline only")
    parser.add_argument("--fwd-file", type=str, default=None,
                        help="Precomputed fwd baseline .npy")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch index (0-indexed)")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    args = parser.parse_args()

    D = args.d; n = args.n
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))

    all_exps = generate_experiments()

    if args.list:
        for i, e in enumerate(all_exps):
            print(f"  {i:3d}. {e['name']:40s}  type={e['type']}")
        print(f"\nTotal: {len(all_exps)} experiments")
        # Show batch sizes
        for b in range(args.num_batches):
            batch = [e for i, e in enumerate(all_exps) if i % args.num_batches == b]
            print(f"  Batch {b}: {len(batch)} experiments")
        exit()

    if args.baseline:
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
        print(f"{dt:.1f}s  ratio={np.mean(fwd):.5f}±{se(fwd):.4f}")
        np.save("fwd_baseline.npy", np.array(fwd_raw, dtype=np.float64))
        print(f"Saved fwd_baseline.npy ({n} trials)")
        exit()

    if args.batch is None:
        parser.error("--batch is required (or use --baseline / --list)")

    # Select batch
    batch_exps = [e for i, e in enumerate(all_exps) if i % args.num_batches == args.batch]
    print(f"Batch {args.batch}/{args.num_batches}: {len(batch_exps)} experiments")
    print(f"D={D}, n={n}, workers={n_workers}\n")

    # Load fwd baseline
    if args.fwd_file:
        fwd_raw = np.load(args.fwd_file)
        fwd = fwd_raw / k
        print(f"Loaded fwd baseline: ratio={np.mean(fwd):.5f}±{se(fwd):.4f}\n")
    else:
        print("Computing fwd baseline...", end=" ", flush=True)
        seeds = generate_seeds(n)
        fwd_args = [(W, ol, D, s[0], s[1]) for s in seeds]
        warmup()
        with mp.Pool(n_workers) as pool:
            fwd_raw_list = pool.map(evaluate_fwd_trial, fwd_args)
        fwd = np.array(fwd_raw_list) / k
        print(f"ratio={np.mean(fwd):.5f}±{se(fwd):.4f}\n")

    # Warm up
    print("Warming up...", end=" ", flush=True)
    warmup()
    evaluate_heuristic((W, ol, D, 42, 42, batch_exps[0]))
    print("done.\n")

    seeds = generate_seeds(n)
    results = []

    for exp_i, config in enumerate(batch_exps):
        trial_args = [(W, ol, D, s[0], s[1], config) for s in seeds]

        t0 = time.monotonic()
        with mp.Pool(n_workers) as pool:
            raw = pool.map(_worker, trial_args)
        dt = time.monotonic() - t0

        mse_vals = np.array([r[0] for r in raw]) / k
        total_fwd_like = sum(r[1] for r in raw)
        total_deviated = sum(r[2] for r in raw)
        total_p2 = sum(r[3] for r in raw)

        diff = mse_vals - fwd
        mean_diff = np.mean(diff)
        se_diff = se(diff)
        sigma = mean_diff / se_diff if se_diff > 0 else 0

        pct_dev = 100 * total_deviated / total_p2 if total_p2 > 0 else 0

        result = {
            "name": config['name'],
            "config": config,
            "ratio_mean": float(np.mean(mse_vals)),
            "ratio_se": float(se(mse_vals)),
            "vs_fwd_mean": float(mean_diff),
            "vs_fwd_se": float(se_diff),
            "vs_fwd_sigma": float(sigma),
            "deviated_pct": float(pct_dev),
            "elapsed": float(dt),
        }
        results.append(result)

        status = "***" if sigma < -3 else "** " if sigma < -2 else "   "
        print(f"  [{exp_i+1:2d}/{len(batch_exps)}] {config['name']:40s}  "
              f"Δ={mean_diff:+.5f}  ({sigma:+5.1f}σ)  dev={pct_dev:5.1f}%  "
              f"{dt:.0f}s  {status}", flush=True)

    # Save results
    fname = f"results_batch{args.batch}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {fname}")

    # Summary: top 10 by sigma
    results.sort(key=lambda r: r['vs_fwd_sigma'])
    print(f"\n{'='*70}")
    print("TOP 10 (most improvement over fwd-merge)")
    print(f"{'='*70}")
    for r in results[:10]:
        print(f"  {r['name']:40s}  Δ={r['vs_fwd_mean']:+.5f}  ({r['vs_fwd_sigma']:+.1f}σ)  dev={r['deviated_pct']:.1f}%")
