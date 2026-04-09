#!/usr/bin/env python3
"""Counterfactual analysis of greedy deviations from fwd-merge.
At each deviation, compute MSE from both choices to see which was better."""

import numpy as np
from numba import njit
import time
import multiprocessing as mp
import argparse
import os
from collections import defaultdict

from run_cones import (
    oracle_query_single, compute_F_true,
    _init_layer_vals, _update_layer_vals, _estimate_from_layer0,
    _compute_terminal_val, _compute_layer_val,
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


def hypothetical_mse(widths, offsets_arr, D, is_known, qid_val, max_qid,
                     layer_vals, layer, node, oracle_seed, offsets_list, W, F_true):
    """Temporarily apply a query and compute the resulting estimate, then undo."""
    W0 = W[0]
    qid = offsets_list[layer] + node
    nv = W[layer + 1] if layer < D - 1 else 2
    val = oracle_query_single(oracle_seed, qid, nv)

    # Temporarily apply
    is_known[qid] = True
    qid_val[qid] = val

    # Recompute layer vals from changed layer up
    # Make copies of affected layers
    old_layers = {}
    if layer == D - 1:
        old_layers[D - 1] = layer_vals[D - 1].copy()
        layer_vals[D - 1] = _compute_terminal_val(widths, offsets_arr, D, is_known, qid_val, max_qid)
    else:
        old_layers[layer] = layer_vals[layer].copy()
        layer_vals[layer] = _compute_layer_val(widths, offsets_arr, D, is_known, qid_val, max_qid,
                                                layer, layer_vals[layer + 1])
    for l in range(min(layer, D - 2), -1, -1):
        old_layers[l] = layer_vals[l].copy()
        layer_vals[l] = _compute_layer_val(widths, offsets_arr, D, is_known, qid_val, max_qid,
                                            l, layer_vals[l + 1])

    est = _estimate_from_layer0(layer_vals[0], W0)
    mse = (est - F_true) ** 2

    # Undo
    is_known[qid] = False
    qid_val[qid] = 0
    for l, old in old_layers.items():
        layer_vals[l] = old

    return mse


def evaluate_with_deviations(args):
    """Run threshold-greedy at 40% and record counterfactual at each deviation."""
    widths_list, offsets_list, D, oracle_seed, strategy_seed = args
    threshold = 0.4

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

    # Phase 1
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
        return []

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

    # Deviation records
    deviations = []
    n_l0_traced = 1  # Phase 1 traced one

    # Phase 2
    while qi < total_q:
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

        # Decide
        use_greedy = (fwd_flat < 0 or greedy_score >= fwd_score * (1.0 + threshold))

        if use_greedy and fwd_flat >= 0:
            # This is a deviation — record counterfactual
            mse_greedy = hypothetical_mse(widths, offsets_arr, D, is_known, qid_val, max_qid,
                                           layer_vals, greedy_layer, greedy_node,
                                           oracle_seed, offsets_list, W, F_true)
            mse_fwd = hypothetical_mse(widths, offsets_arr, D, is_known, qid_val, max_qid,
                                        layer_vals, fwd_layer, fwd_node,
                                        oracle_seed, offsets_list, W, F_true)

            # Classify greedy pick type
            if greedy_layer == 0:
                greedy_type = "new_L0"
            else:
                greedy_type = f"jump_L{greedy_layer}"

            # What is fwd doing?
            if fwd_layer == 0:
                fwd_type = "start_L0"
            else:
                fwd_type = f"chain_L{fwd_layer}"

            progress = qi / total_q
            score_ratio = greedy_score / fwd_score if fwd_score > 0 else float('inf')

            deviations.append({
                'mse_greedy': mse_greedy,
                'mse_fwd': mse_fwd,
                'delta': mse_fwd - mse_greedy,  # positive = greedy better
                'greedy_layer': greedy_layer,
                'fwd_layer': fwd_layer,
                'greedy_type': greedy_type,
                'fwd_type': fwd_type,
                'score_ratio': score_ratio,
                'progress': progress,
                'n_l0_traced': n_l0_traced,
                'qi': qi,
            })

            best_layer, best_node = greedy_layer, greedy_node
        elif use_greedy:
            best_layer, best_node = greedy_layer, greedy_node
        else:
            best_layer, best_node = fwd_layer, fwd_node

        # Track L0 traces
        if best_layer == 0:
            n_l0_traced += 1

        changed = do_query(best_layer, best_node)
        qi += 1

        _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets_arr,
                                    is_known, qid_val, changed)
        _update_layer_vals(widths, offsets_arr, D, is_known, qid_val, max_qid,
                          layer_vals, changed)
        est = _estimate_from_layer0(layer_vals[0], W0)

        for l in range(changed + 1):
            a = int(layer_off[l])
            lv_flat[a:a + W[l]] = layer_vals[l]

        if abs(est - F_true) < 1e-15:
            break

        advance_fwd_state()

    return deviations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    D = args.d; n = args.n
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    print(f"D={D}, n={n}, workers={n_workers}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    evaluate_with_deviations((W, ol, D, 42, 42))
    print("done.\n")

    rng = np.random.default_rng(42)
    base_args = [(W, ol, D, int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
                 for _ in range(n)]

    print("Running...", end=" ", flush=True)
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        all_results = pool.map(evaluate_with_deviations, base_args)
    dt = time.monotonic() - t0
    print(f"{dt:.1f}s\n")

    # Flatten all deviations
    all_devs = []
    for trial_devs in all_results:
        all_devs.extend(trial_devs)

    n_devs = len(all_devs)
    print(f"Total deviations: {n_devs} across {n} trials ({n_devs/n:.1f} per trial)\n")

    if n_devs == 0:
        print("No deviations to analyze.")
        exit()

    deltas = np.array([d['delta'] for d in all_devs])
    n_better = np.sum(deltas > 1e-15)
    n_worse = np.sum(deltas < -1e-15)
    n_same = n_devs - n_better - n_worse

    print("=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"  Greedy better:  {n_better:>7d}  ({100*n_better/n_devs:5.1f}%)")
    print(f"  Greedy worse:   {n_worse:>7d}  ({100*n_worse/n_devs:5.1f}%)")
    print(f"  Same:           {n_same:>7d}  ({100*n_same/n_devs:5.1f}%)")
    print(f"  Mean delta:     {np.mean(deltas):+.6f}  (positive = greedy better)")
    print(f"  Median delta:   {np.median(deltas):+.6f}")

    # By greedy pick type
    print(f"\n{'=' * 70}")
    print("BY GREEDY PICK TYPE")
    print("=" * 70)
    types = defaultdict(list)
    for d in all_devs:
        types[d['greedy_type']].append(d['delta'])

    for t in sorted(types.keys()):
        vals = np.array(types[t])
        n_t = len(vals)
        n_b = np.sum(vals > 1e-15)
        n_w = np.sum(vals < -1e-15)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        print(f"  {t:>12s}: n={n_t:>6d}  mean_delta={np.mean(vals):+.6f}±{se:.6f}"
              f"  better={100*n_b/n_t:5.1f}%  worse={100*n_w/n_t:5.1f}%")

    # By fwd pick type
    print(f"\n{'=' * 70}")
    print("BY FWD-MERGE PICK TYPE (what greedy overrode)")
    print("=" * 70)
    types2 = defaultdict(list)
    for d in all_devs:
        types2[d['fwd_type']].append(d['delta'])

    for t in sorted(types2.keys()):
        vals = np.array(types2[t])
        n_t = len(vals)
        n_b = np.sum(vals > 1e-15)
        n_w = np.sum(vals < -1e-15)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        print(f"  {t:>12s}: n={n_t:>6d}  mean_delta={np.mean(vals):+.6f}±{se:.6f}"
              f"  better={100*n_b/n_t:5.1f}%  worse={100*n_w/n_t:5.1f}%")

    # By score ratio bucket
    print(f"\n{'=' * 70}")
    print("BY SCORE RATIO (greedy_score / fwd_score)")
    print("=" * 70)
    ratios = np.array([d['score_ratio'] for d in all_devs])
    buckets = [(1.4, 1.6), (1.6, 1.8), (1.8, 2.0), (2.0, 2.5), (2.5, 3.0),
               (3.0, 5.0), (5.0, 10.0), (10.0, float('inf'))]
    for lo, hi in buckets:
        mask = (ratios >= lo) & (ratios < hi)
        if np.sum(mask) == 0:
            continue
        vals = deltas[mask]
        n_t = len(vals)
        n_b = np.sum(vals > 1e-15)
        n_w = np.sum(vals < -1e-15)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        hi_s = f"{hi:.1f}" if hi < 100 else "inf"
        print(f"  [{lo:.1f}, {hi_s:>4s}): n={n_t:>6d}  mean_delta={np.mean(vals):+.6f}±{se:.6f}"
              f"  better={100*n_b/n_t:5.1f}%  worse={100*n_w/n_t:5.1f}%")

    # By progress bucket
    print(f"\n{'=' * 70}")
    print("BY PROGRESS (fraction of queries completed)")
    print("=" * 70)
    progress = np.array([d['progress'] for d in all_devs])
    p_buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                 (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for lo, hi in p_buckets:
        mask = (progress >= lo) & (progress < hi)
        if np.sum(mask) == 0:
            continue
        vals = deltas[mask]
        n_t = len(vals)
        n_b = np.sum(vals > 1e-15)
        n_w = np.sum(vals < -1e-15)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        print(f"  [{lo:.0%}, {hi:.0%}): n={n_t:>6d}  mean_delta={np.mean(vals):+.6f}±{se:.6f}"
              f"  better={100*n_b/n_t:5.1f}%  worse={100*n_w/n_t:5.1f}%")

    # By greedy layer + fwd layer combination
    print(f"\n{'=' * 70}")
    print("BY DEVIATION PATTERN (greedy_type → fwd_type)")
    print("=" * 70)
    combos = defaultdict(list)
    for d in all_devs:
        key = f"{d['greedy_type']} over {d['fwd_type']}"
        combos[key].append(d['delta'])

    for key in sorted(combos.keys(), key=lambda k: -len(combos[k])):
        vals = np.array(combos[key])
        n_t = len(vals)
        if n_t < 10:
            continue
        n_b = np.sum(vals > 1e-15)
        n_w = np.sum(vals < -1e-15)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        print(f"  {key:>35s}: n={n_t:>6d}  mean_delta={np.mean(vals):+.6f}±{se:.6f}"
              f"  better={100*n_b/n_t:5.1f}%  worse={100*n_w/n_t:5.1f}%")
