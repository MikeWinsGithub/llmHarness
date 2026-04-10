#!/usr/bin/env python3
"""Exact 1-step variance reduction greedy with fwd-merge threshold gating.

Precomputes prop_factor[l][u] = ∂est/∂layer_vals[l][u] for ALL nodes in O(total_q)
via a forward pass. Then:
  R(l, u) = prop_factor[l][u]² × var(layer_vals[l+1])
is the exact 1-step variance reduction, computed in O(1) per candidate.

Also supports reach^a × var^b scoring for comparison.
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

@njit
def _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets,
                                is_known, qid_val, start_layer):
    for layer in range(start_layer, D - 1):
        a = layer_off[layer]; W_l = widths[layer]; W_next = widths[layer + 1]
        qid_start = offsets[layer]; na = layer_off[layer + 1]
        for u in range(W_next): exp_reach[na + u] = 0.0
        unknown_total = 0.0
        for v in range(W_l):
            r = exp_reach[a + v]
            if r == 0.0: continue
            qid = qid_start + v
            if is_known[qid]: exp_reach[na + qid_val[qid]] += r
            else: unknown_total += r
        if unknown_total > 0.0:
            uniform = unknown_total / W_next
            for u in range(W_next): exp_reach[na + u] += uniform


def evaluate_exact_greedy(args):
    widths_list, offsets_list, D, oracle_seed, strategy_seed, threshold, score_a, schedule = args
    # schedule: dict with optional keys:
    #   "type": "fixed" (default), "linear", "switch", "step"
    #   "x1": slope for linear (threshold = threshold + x1 * t)
    #   "switch_k": for switch (fwd until K L0 traced, then greedy)
    #   "high","low","cutoff": for step function
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
        is_known[qid] = True; qid_val[qid] = val; g_queried[flat] = True
        if layer == D - 1:
            g_known[flat] = True
            for p in g_preds[flat]: propagate_known(p)
        else:
            g_succ[flat] = val
            succ_flat = fi(layer + 1, val)
            g_preds[succ_flat].append(flat)
            delta = g_reach[flat]
            cl, cn_flat = layer + 1, succ_flat; cn = val
            while True:
                g_reach[cn_flat] += delta
                if cl < D - 1 and g_queried[cn_flat]:
                    cn = int(g_succ[cn_flat]); cl += 1; cn_flat = fi(cl, cn)
                else: break
            if g_known[succ_flat]: propagate_known(flat)
        return layer

    def compute_all_prop_factors():
        """Compute prop_factor for ALL nodes in O(total_q) via forward pass.
        prop_factor[flat] = ∂est / ∂layer_vals[layer][node] for node at flat index."""
        pf = np.zeros(total_q, dtype=np.float64)
        # Layer 0: each node contributes 1/W0 to est
        pf[:W0] = 1.0 / W0

        for l in range(1, D):
            W_prev = W[l - 1]; W_cur = W[l]
            a_prev = layer_off[l - 1]; a_cur = layer_off[l]

            # Accumulate influence from layer l-1 to layer l
            unk_sum = 0.0
            known_to = np.zeros(W_cur, dtype=np.float64)

            for v in range(W_prev):
                qid = offsets_list[l - 1] + v
                if is_known[qid]:
                    dest = qid_val[qid]
                    known_to[dest] += pf[a_prev + v]
                else:
                    unk_sum += pf[a_prev + v]

            for u in range(W_cur):
                pf[a_cur + u] = known_to[u] + unk_sum / W_cur

        return pf

    def compute_scores(pf):
        """Score each unqueried reachable node by exact variance reduction
        or reach^a × var^b."""
        scores = np.zeros(total_q, dtype=np.float64)

        if score_a < 0:
            # Exact variance reduction: R = pf² × var_next
            for l in range(D):
                if l < D - 1:
                    var_next = float(np.var(layer_vals[l + 1]))
                else:
                    var_next = 1.0 - float(np.mean(layer_vals[D - 1])) ** 2
                a = layer_off[l]
                for u in range(W[l]):
                    flat = a + u
                    if not g_queried[flat] and g_reach[flat] > 0:
                        scores[flat] = pf[flat] ** 2 * var_next
        else:
            # reach^a × var scoring
            for l in range(D):
                if l < D - 1:
                    var_next = float(np.var(layer_vals[l + 1]))
                else:
                    var_next = 1.0
                a = layer_off[l]
                for u in range(W[l]):
                    flat = a + u
                    if not g_queried[flat] and g_reach[flat] > 0:
                        if score_a == 1.0:
                            scores[flat] = exp_reach[flat] * var_next
                        else:
                            scores[flat] = (exp_reach[flat] ** score_a) * var_next

        return scores

    # fwd-merge state
    fwd_layer = 0; fwd_node = None

    def advance_fwd_state():
        nonlocal fwd_layer, fwd_node, l0_idx
        if fwd_node is not None:
            flat = fi(fwd_layer, fwd_node)
            if not g_queried[flat]: return
            if fwd_layer < D - 1 and g_succ[flat] >= 0:
                fwd_layer += 1; fwd_node = int(g_succ[flat])
                return advance_fwd_state()
            else: fwd_node = None
        while l0_idx < W0:
            v = l0_perm[l0_idx]
            flat = fi(0, v)
            if not g_queried[flat]:
                fwd_layer = 0; fwd_node = v; return
            cur_l, cur_n = 0, v
            while cur_l < D:
                cur_flat = fi(cur_l, cur_n)
                if not g_queried[cur_flat]:
                    fwd_layer = cur_l; fwd_node = cur_n; return
                if cur_l < D - 1 and g_succ[cur_flat] >= 0:
                    cur_n = int(g_succ[cur_flat]); cur_l += 1
                else: break
            l0_idx += 1
        fwd_node = None

    # Phase 1: one complete trace
    v0 = l0_perm[l0_idx]; l0_idx += 1
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
        return float(mse_sum), 0, 0

    _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets_arr,
                                is_known, qid_val, 0)

    advance_fwd_state()
    n_fwd = 0; n_dev = 0; n_l0_traced = 1

    sched_type = schedule.get("type", "fixed") if schedule else "fixed"

    # Phase 2
    while qi < total_q:
        # Compute dynamic threshold
        t = n_l0_traced / W0
        if sched_type == "linear":
            cur_threshold = max(threshold + schedule.get("x1", 0) * t, 0.0)
        elif sched_type == "switch":
            cur_threshold = 0.0 if n_l0_traced >= schedule.get("switch_k", W0) else 1e9
        elif sched_type == "step":
            cur_threshold = schedule.get("high", 1.0) if t < schedule.get("cutoff", 0.5) else schedule.get("low", 0.0)
        else:
            cur_threshold = threshold

        pf = compute_all_prop_factors()
        scores = compute_scores(pf)

        greedy_flat = int(np.argmax(scores))
        if scores[greedy_flat] <= 0:
            mse_sum += (total_q - qi) * (est - F_true) ** 2
            break

        greedy_score = scores[greedy_flat]

        # fwd-merge comparison
        if fwd_node is not None:
            fwd_flat = fi(fwd_layer, fwd_node)
            fwd_score = scores[fwd_flat]
        else:
            fwd_score = 0.0

        # Threshold gating
        if fwd_node is not None and greedy_score < fwd_score * (1.0 + cur_threshold):
            best_layer = fwd_layer; best_node = fwd_node; n_fwd += 1
        else:
            best_layer = int(flat_layer[greedy_flat])
            best_node = greedy_flat - layer_off[best_layer]; n_dev += 1

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

        if abs(est - F_true) < 1e-15:
            break

        advance_fwd_state()

    return float(mse_sum), n_fwd, n_dev


def _worker(args):
    return evaluate_exact_greedy(args)


def generate_seeds(n):
    rng = np.random.default_rng(42)
    return [(int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
            for _ in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--threshold", type=float, nargs='+', default=[0.0, 0.3, 0.5, 1.0])
    parser.add_argument("--score-a", type=float, default=-1.0,
                        help="Score type: -1 = exact var reduction, >0 = reach^a × var")
    parser.add_argument("--schedule", type=str, default="fixed",
                        choices=["fixed", "linear", "switch", "step", "all"],
                        help="Threshold schedule type")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fwd-file", type=str, default=None)
    args = parser.parse_args()

    D = args.d; n = args.n; score_a = args.score_a
    thresholds = args.threshold
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))

    scoring = "exact_var_reduction" if score_a < 0 else f"reach^{score_a:.1f}×var"

    # Build experiment list
    experiments = []
    if args.schedule == "all":
        # Comprehensive sweep
        for t in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            experiments.append((f"fixed_t{t:.1f}", t, {}))
        for x1 in [-0.3, 0.0, 0.3, 0.5]:
            experiments.append((f"linear_0.4{x1:+.1f}", 0.4, {"type": "linear", "x1": x1}))
        for x1 in [0.0, 0.3, 0.5]:
            experiments.append((f"linear_0.5{x1:+.1f}", 0.5, {"type": "linear", "x1": x1}))
        for sk in [30, 40, 50, 60, 70]:
            experiments.append((f"switch_{sk}", 0.0, {"type": "switch", "switch_k": sk}))
        for hi, lo, co in [(1.0, 0.3, 0.5), (0.8, 0.2, 0.5), (1.0, 0.0, 0.5),
                           (0.5, 0.0, 0.3), (0.5, 0.0, 0.5)]:
            experiments.append((f"step_h{hi}_l{lo}_c{co}", 0.0,
                              {"type": "step", "high": hi, "low": lo, "cutoff": co}))
    elif args.schedule == "fixed":
        for t in thresholds:
            experiments.append((f"fixed_t{t:.2f}", t, {}))
    elif args.schedule == "linear":
        for t in thresholds:
            experiments.append((f"linear_{t:+.1f}", 0.4, {"type": "linear", "x1": t}))
    elif args.schedule == "switch":
        for t in thresholds:
            experiments.append((f"switch_{int(t)}", 0.0, {"type": "switch", "switch_k": int(t)}))
    elif args.schedule == "step":
        for t in thresholds:
            experiments.append((f"step_c{t:.1f}", 0.0,
                              {"type": "step", "high": 1.0, "low": 0.3, "cutoff": t}))

    print(f"D={D}, n={n}, scoring={scoring}, schedule={args.schedule}, workers={n_workers}")
    print(f"Experiments: {len(experiments)}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    evaluate_exact_greedy((W, ol, D, 42, 42, 0.5, score_a, {}))
    print("done.\n")

    seeds = generate_seeds(n)

    # fwd-merge baseline
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

    for name, threshold, schedule in experiments:
        trial_args = [(W, ol, D, s[0], s[1], threshold, score_a, schedule) for s in seeds]
        print(f"Running {name:30s}...", end=" ", flush=True)
        t0 = time.monotonic()
        with mp.Pool(n_workers) as pool:
            results = pool.map(_worker, trial_args)
        dt = time.monotonic() - t0

        mse_vals = np.array([r[0] for r in results]) / k
        total_fwd = sum(r[1] for r in results)
        total_dev = sum(r[2] for r in results)
        total_p2 = total_fwd + total_dev
        pct_dev = 100 * total_dev / total_p2 if total_p2 > 0 else 0

        diff = mse_vals - fwd
        mean_diff = np.mean(diff)
        se_diff = se(diff)
        sigma = mean_diff / se_diff if se_diff > 0 else 0
        star = "***" if sigma < -3 else "** " if sigma < -2 else "   "

        print(f"{dt:.0f}s  ratio={np.mean(mse_vals):.5f}  "
              f"Δ={mean_diff:+.5f}±{se_diff:.5f} ({sigma:+.1f}σ)  "
              f"dev={pct_dev:.1f}% {star}")
