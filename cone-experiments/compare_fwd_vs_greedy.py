#!/usr/bin/env python3
"""Compare fwd-merge vs reached-weighted-fast at D=40.
Paired test + decision classification for the greedy."""

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
from reached_weighted_fast import (
    _forward_reach_incremental, _compute_layer_vars,
)


def evaluate_greedy_instrumented(args):
    """Reached-weighted-fast with decision tracking.
    Returns (mse_sum, counts_dict) where counts_dict classifies each Phase 2 pick."""
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
        return float(mse_sum), {
            'chain_continue': 0, 'new_L0_complete': 0,
            'new_L0_incomplete': 0, 'jump': 0, 'phase2_total': 0,
        }

    _forward_reach_incremental(exp_reach, widths, D, layer_off, offsets_arr,
                                is_known, qid_val, 0)

    lv_flat = np.zeros(total_q, dtype=np.float64)
    for l in range(D):
        a = int(layer_off[l])
        lv_flat[a:a + W[l]] = layer_vals[l]

    _score_buf = np.empty(total_q, dtype=np.float64)

    # Decision counters
    chain_continue = 0    # picked the successor of prev query
    new_L0_complete = 0   # picked L0, prev chain was fully resolved
    new_L0_incomplete = 0 # picked L0, prev chain had unqueried successors
    jump = 0              # picked something else (mid-layer, not successor)

    prev_layer = D - 1  # last query of Phase 1 was terminal
    prev_node = cur
    prev_chain_complete = True  # Phase 1 chain is complete

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

        flat_best = int(np.argmax(_score_buf))
        if _score_buf[flat_best] <= 0:
            mse_sum += (total_q - qi) * (est - F_true) ** 2
            break

        best_layer = int(flat_layer[flat_best])
        best_node = flat_best - layer_off[best_layer]

        # Classify this decision
        # What would "continuing the chain" mean?
        # If prev_layer < D-1 and prev node has a known successor, the chain
        # continuation is querying that successor.
        prev_flat = fi(prev_layer, prev_node)
        if prev_layer < D - 1 and g_succ[prev_flat] >= 0:
            expected_next_layer = prev_layer + 1
            expected_next_node = int(g_succ[prev_flat])
            if best_layer == expected_next_layer and best_node == expected_next_node:
                chain_continue += 1
            elif best_layer == 0:
                # Check if the chain from prev was complete
                # Walk down from prev to see if all successors are queried
                if prev_chain_complete:
                    new_L0_complete += 1
                else:
                    new_L0_incomplete += 1
            else:
                jump += 1
        elif best_layer == 0:
            # prev was terminal or had no successor — starting new chain is natural
            new_L0_complete += 1
        else:
            jump += 1

        # Update prev tracking
        prev_layer = best_layer
        prev_node = best_node

        # Check if current chain is complete (walk successors to see if all queried)
        prev_chain_complete = False
        cl, cn = best_layer, best_node
        while cl < D - 1:
            cf = fi(cl, cn)
            if g_succ[cf] >= 0:
                cn = int(g_succ[cf]); cl += 1
                if g_queried[fi(cl, cn)]:
                    continue
                else:
                    break
            else:
                break
        else:
            # Reached terminal layer and it's queried
            if g_queried[fi(cl, cn)]:
                prev_chain_complete = True

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

    phase2_total = chain_continue + new_L0_complete + new_L0_incomplete + jump
    return float(mse_sum), {
        'chain_continue': chain_continue,
        'new_L0_complete': new_L0_complete,
        'new_L0_incomplete': new_L0_incomplete,
        'jump': jump,
        'phase2_total': phase2_total,
    }


def _unpack_greedy(args):
    """Wrapper for mp.Pool.map that returns (mse, counts)."""
    return evaluate_greedy_instrumented(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    D = args.d; n = args.n
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    print(f"D={D}, n={n}, workers={n_workers}")
    print(f"Funnel widths: [{W[0]}, {W[1]}, ..., {W[-1]}]  total_q={sum(W)}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    # Warm up greedy too
    evaluate_greedy_instrumented((W, ol, D, 42, 42))
    print("done.\n")

    rng = np.random.default_rng(42)
    base_args = [(W, ol, D, int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
                 for _ in range(n)]

    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))

    # fwd-merge
    print("Running fwd-merge...", end=" ", flush=True)
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        fwd_raw = pool.map(evaluate_fwd_trial, base_args)
    fwd_dt = time.monotonic() - t0
    fwd = np.array(fwd_raw) / k
    print(f"{fwd_dt:.1f}s")

    # reached-weighted-fast (instrumented)
    print("Running reached-weighted-fast...", end=" ", flush=True)
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        greedy_results = pool.map(_unpack_greedy, base_args)
    greedy_dt = time.monotonic() - t0
    print(f"{greedy_dt:.1f}s")

    greedy_mse = np.array([r[0] for r in greedy_results]) / k
    counts = [r[1] for r in greedy_results]

    # --- Results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n  fwd-merge:              {np.mean(fwd):.5f} ± {se(fwd):.4f}")
    print(f"  reached-weighted-fast:  {np.mean(greedy_mse):.5f} ± {se(greedy_mse):.4f}")

    diff = greedy_mse - fwd
    mean_diff = np.mean(diff)
    se_diff = se(diff)
    sigma = mean_diff / se_diff if se_diff > 0 else 0
    print(f"\n  Paired difference:      {mean_diff:+.5f} ± {se_diff:.5f}  ({sigma:+.1f}σ)")
    if abs(sigma) >= 2:
        winner = "fwd-merge" if mean_diff > 0 else "reached-weighted-fast"
        print(f"  ** Statistically significant at 2σ: {winner} is better **")
    else:
        print(f"  Not statistically significant (|{sigma:.1f}σ| < 2σ)")

    # --- Decision classification ---
    total_cc = sum(c['chain_continue'] for c in counts)
    total_l0c = sum(c['new_L0_complete'] for c in counts)
    total_l0i = sum(c['new_L0_incomplete'] for c in counts)
    total_j = sum(c['jump'] for c in counts)
    total_p2 = sum(c['phase2_total'] for c in counts)

    print(f"\n" + "=" * 60)
    print("GREEDY DECISION CLASSIFICATION (Phase 2)")
    print("=" * 60)
    print(f"  Total Phase 2 decisions:  {total_p2}")
    if total_p2 > 0:
        def pct(x): return f"{x:>7d}  ({100*x/total_p2:5.1f}%)"
        print(f"\n  Matches fwd-merge behavior:")
        print(f"    Chain continue (next in path):     {pct(total_cc)}")
        print(f"    New L0 (prev chain complete):       {pct(total_l0c)}")
        fwd_like = total_cc + total_l0c
        print(f"    --- subtotal fwd-like:             {pct(fwd_like)}")
        print(f"\n  Differs from fwd-merge:")
        print(f"    New L0 (prev chain incomplete):     {pct(total_l0i)}")
        print(f"    Jump (mid-layer, not successor):    {pct(total_j)}")
        non_fwd = total_l0i + total_j
        print(f"    --- subtotal non-fwd:              {pct(non_fwd)}")

    # Per-trial averages
    avg_p2 = np.mean([c['phase2_total'] for c in counts])
    avg_cc = np.mean([c['chain_continue'] for c in counts])
    avg_l0c = np.mean([c['new_L0_complete'] for c in counts])
    avg_l0i = np.mean([c['new_L0_incomplete'] for c in counts])
    avg_j = np.mean([c['jump'] for c in counts])
    print(f"\n  Per-trial averages (of {avg_p2:.0f} Phase 2 decisions):")
    print(f"    chain_continue:  {avg_cc:.1f}")
    print(f"    new_L0_complete: {avg_l0c:.1f}")
    print(f"    new_L0_inc:      {avg_l0i:.1f}")
    print(f"    jump:            {avg_j:.1f}")
