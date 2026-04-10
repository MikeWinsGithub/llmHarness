#!/usr/bin/env python3
"""Parameterized strategy family for cone query ordering.

The strategy is fwd-merge with tunable deviations:

Parameters:
  a             Score exponent: score = prop_factor^a × var_next  (default 1.0)
  threshold     Only deviate if alt_score > fwd_score × (1+threshold)  (default 0.5)
  chain_bonus   Multiply chain continuation score by (1+bonus)  (default 0.0)
  slope         Threshold += slope × (fraction L0 traced)  (default 0.0)
  depth_pen     Threshold += depth_pen × (alt_layer / D)  (default 0.0)
  n_warmup      Complete N fwd-merge traces before allowing deviations  (default 1)

Special cases:
  threshold=∞           → pure fwd-merge
  threshold=0, bonus=0  → pure greedy (prop_factor scoring)
  n_warmup=50           → switch-at-50
  depth_pen=∞           → L0-only deviations
  chain_bonus=∞         → always complete chains
  a=1.3                 → reach^1.3 scoring (approximately)
"""

import numpy as np
import time
import multiprocessing as mp
import argparse
import json

from run_cones import (
    oracle_query_single, compute_F_true,
    _init_layer_vals, _update_layer_vals, _estimate_from_layer0,
    evaluate_single_trial as evaluate_fwd_trial,
    funnel, warmup,
)


def evaluate_param_strategy(args):
    (widths_list, offsets_list, D, oracle_seed, strategy_seed, params) = args
    a           = params.get('a', 1.0)         # prop_factor exponent
    b           = params.get('b', 1.0)         # var_next exponent
    threshold   = params.get('threshold', 0.5) # base deviation threshold
    chain_bonus = params.get('chain_bonus', 0.0) # bonus for chain continuation
    slope       = params.get('slope', 0.0)     # threshold += slope * progress
    depth_pen   = params.get('depth_pen', 0.0) # threshold += depth_pen * (layer/D)
    n_warmup    = max(1, int(params.get('n_warmup', 1))) # warmup traces
    jump_pen    = params.get('jump_pen', 0.0)  # extra threshold for non-L0 deviations
    reach_min   = params.get('reach_min', 0.0) # min reach to consider deviation
    progress_pow = params.get('progress_pow', 1.0) # threshold += slope * t^pow

    widths = np.array(widths_list, dtype=np.int32)
    offsets_arr = np.array(offsets_list, dtype=np.int32)
    W = widths_list
    total_q = sum(W); max_qid = offsets_list[-1] + W[-1]; W0 = W[0]

    F_true = compute_F_true(widths, offsets_arr, D, oracle_seed)
    F_true_sq = F_true * F_true
    layer_off = np.array([sum(W[:i]) for i in range(D)], dtype=np.int32)
    flat_layer = np.repeat(np.arange(D, dtype=np.int32), W)

    is_known = np.zeros(max_qid, dtype=np.bool_)
    qid_val = np.zeros(max_qid, dtype=np.int32)
    g_queried = np.zeros(total_q, dtype=np.bool_)
    g_succ = np.full(total_q, -1, dtype=np.int32)
    g_reach = np.zeros(total_q, dtype=np.float64)
    g_known = np.zeros(total_q, dtype=np.bool_)
    g_preds = [[] for _ in range(total_q)]

    rng = np.random.default_rng(strategy_seed)
    l0_perm = list(rng.permutation(W0))
    l0_idx = 0
    g_reach[:W0] = 1.0

    def fi(ly, nd): return layer_off[ly] + nd

    def propagate_known(flat_idx):
        stk = [flat_idx]
        while stk:
            idx = stk.pop()
            if g_known[idx]: continue
            g_known[idx] = True
            if flat_layer[idx] > 0:
                for p in g_preds[idx]:
                    if not g_known[p]: stk.append(p)

    def do_query(ly, nd):
        flat = fi(ly, nd); qid = offsets_list[ly] + nd
        nv = W[ly + 1] if ly < D - 1 else 2
        val = oracle_query_single(oracle_seed, qid, nv)
        is_known[qid] = True; qid_val[qid] = val; g_queried[flat] = True
        if ly == D - 1:
            g_known[flat] = True
            for p in g_preds[flat]: propagate_known(p)
        else:
            g_succ[flat] = val
            sf = fi(ly + 1, val); g_preds[sf].append(flat)
            d = g_reach[flat]; cl = ly + 1; cf = sf
            while True:
                g_reach[cf] += d
                if cl < D - 1 and g_queried[cf]:
                    cl += 1; cf = fi(cl, int(g_succ[cf]))
                else: break
            if g_known[sf]: propagate_known(flat)
        return ly

    def compute_prop_factors():
        pf = np.zeros(total_q, dtype=np.float64)
        pf[:W0] = 1.0 / W0
        for l in range(1, D):
            ap = int(layer_off[l-1]); Wp = W[l-1]
            ac = int(layer_off[l]); Wc = W[l]
            unk_sum = 0.0
            known_to = np.zeros(Wc, dtype=np.float64)
            for v in range(Wp):
                qid = offsets_list[l-1] + v
                if is_known[qid]:
                    known_to[qid_val[qid]] += pf[ap + v]
                else:
                    unk_sum += pf[ap + v]
            for u in range(Wc):
                pf[ac + u] = known_to[u] + unk_sum / Wc
        return pf

    def compute_scores(pf):
        scores = np.zeros(total_q, dtype=np.float64)
        for l in range(D):
            if l < D - 1:
                var_next = float(np.var(layer_vals[l + 1]))
            else:
                mn = float(np.mean(layer_vals[D - 1]))
                var_next = max(1.0 - mn * mn, 0.0)
            vn = var_next ** b if b != 1.0 else var_next
            al = int(layer_off[l])
            for u in range(W[l]):
                flat = al + u
                if not g_queried[flat] and g_reach[flat] > 0:
                    pf_val = pf[flat] ** a if a != 1.0 else pf[flat]
                    scores[flat] = pf_val * vn
        return scores

    # fwd-merge state
    fwd_layer = 0; fwd_node = None; n_l0_traced = 0

    def advance_fwd():
        nonlocal fwd_layer, fwd_node, l0_idx
        if fwd_node is not None:
            flat = fi(fwd_layer, fwd_node)
            if not g_queried[flat]: return
            if fwd_layer < D - 1 and g_succ[flat] >= 0:
                fwd_layer += 1; fwd_node = int(g_succ[flat])
                return advance_fwd()
            fwd_node = None
        while l0_idx < W0:
            v = l0_perm[l0_idx]
            if not g_queried[fi(0, v)]:
                fwd_layer = 0; fwd_node = v; return
            cl, cn = 0, v
            while cl < D:
                if not g_queried[fi(cl, cn)]:
                    fwd_layer = cl; fwd_node = cn; return
                if cl < D - 1 and g_succ[fi(cl, cn)] >= 0:
                    cn = int(g_succ[fi(cl, cn)]); cl += 1
                else: break
            l0_idx += 1
        fwd_node = None

    # Phase 1: n_warmup complete traces
    qi = 0; mse_sum = F_true_sq; layer_vals = None; est = 0.0

    for trace in range(n_warmup):
        if l0_idx >= W0: break
        v0 = l0_perm[l0_idx]; l0_idx += 1; n_l0_traced += 1
        cur = v0
        for ly in range(D):
            do_query(ly, cur); qi += 1
            if ly < D - 1:
                if layer_vals is None:
                    mse_sum += F_true_sq
                else:
                    mse_sum += (est - F_true) ** 2
                cur = int(qid_val[offsets_list[ly] + cur])
            else:
                if layer_vals is None:
                    layer_vals = _init_layer_vals(widths, offsets_arr, D, is_known, qid_val, max_qid)
                else:
                    _update_layer_vals(widths, offsets_arr, D, is_known, qid_val, max_qid,
                                     layer_vals, ly)
                est = _estimate_from_layer0(layer_vals[0], W0)
                mse_sum += (est - F_true) ** 2

        if abs(est - F_true) < 1e-15:
            return float(mse_sum), 0, 0

    advance_fwd()

    n_fwd = 0; n_dev = 0

    # Phase 2: parameterized deviations
    while qi < total_q:
        pf = compute_prop_factors()
        scores = compute_scores(pf)

        # Best alternative
        best_alt = int(np.argmax(scores))
        if scores[best_alt] <= 0:
            mse_sum += (total_q - qi) * (est - F_true) ** 2
            break

        alt_score = scores[best_alt]
        alt_layer = int(flat_layer[best_alt])

        # fwd-merge score (chain bonus only when continuing, not starting new L0)
        if fwd_node is not None:
            fwd_flat = fi(fwd_layer, fwd_node)
            is_chain_cont = (fwd_layer > 0)  # layer>0 means continuing, layer=0 means new input
            bonus = chain_bonus if is_chain_cont else 0.0
            fwd_score = scores[fwd_flat] * (1.0 + bonus)
        else:
            fwd_score = 0.0

        # Dynamic threshold
        t = n_l0_traced / W0
        tp = t ** progress_pow if progress_pow != 1.0 else t
        is_jump = (alt_layer > 0)
        eff_threshold = (threshold
                         + slope * tp
                         + depth_pen * (alt_layer / (D - 1))
                         + jump_pen * (1.0 if is_jump else 0.0))

        # Reach gate
        alt_reach = g_reach[best_alt]

        # Decision
        if (fwd_node is not None
            and (alt_score < fwd_score * (1.0 + eff_threshold)
                 or alt_reach < reach_min)):
            best_ly = fwd_layer; best_nd = fwd_node; n_fwd += 1
        else:
            best_ly = alt_layer
            best_nd = best_alt - layer_off[alt_layer]; n_dev += 1

        if best_ly == 0 and not g_queried[fi(0, best_nd)]:
            n_l0_traced += 1

        changed = do_query(best_ly, best_nd); qi += 1
        _update_layer_vals(widths, offsets_arr, D, is_known, qid_val, max_qid,
                          layer_vals, changed)
        est = _estimate_from_layer0(layer_vals[0], W0)
        mse_sum += (est - F_true) ** 2

        if abs(est - F_true) < 1e-15:
            break
        advance_fwd()

    return float(mse_sum), n_fwd, n_dev


def _worker(args):
    return evaluate_param_strategy(args)


def generate_seeds(n):
    rng = np.random.default_rng(42)
    return [(int(rng.integers(0, 2**62)), int(rng.integers(0, 2**62)))
            for _ in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=None)
    # Strategy parameters
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chain-bonus", type=float, default=0.0)
    parser.add_argument("--slope", type=float, default=0.0)
    parser.add_argument("--depth-pen", type=float, default=0.0)
    parser.add_argument("--n-warmup", type=int, default=1)
    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Run a preset sweep of parameter combinations")
    args = parser.parse_args()

    D = args.d; n = args.n
    n_workers = args.workers or os.cpu_count() or 4
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
    seeds = generate_seeds(n)

    print(f"D={D}, n={n}, workers={n_workers}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    evaluate_param_strategy((W, ol, D, 42, 42, 1.0, 0.5, 0.0, 0.0, 0.0, 1))
    print("done.\n")

    # fwd-merge baseline
    fwd_args = [(W, ol, D, s[0], s[1]) for s in seeds]
    print("Running fwd-merge...", end=" ", flush=True)
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        fwd_raw = pool.map(evaluate_fwd_trial, fwd_args)
    fwd = np.array(fwd_raw) / k
    print(f"{time.monotonic()-t0:.1f}s  ratio={np.mean(fwd):.5f}±{se(fwd):.4f}\n")

    if args.sweep:
        experiments = []
        def add(name, **kw):
            experiments.append((name, kw))

        # Baselines
        add("fwd-merge", threshold=1e6)
        add("pure greedy", threshold=0.0)

        # Score exponents
        for av in [0.8, 1.0, 1.3, 2.0]:
            add(f"a={av}", a=av, threshold=0.5)
        for bv in [0.5, 1.0, 1.5]:
            add(f"b={bv}", b=bv, threshold=0.5)

        # Chain bonus (now only applies mid-chain)
        for cb in [0.5, 1.0, 2.0, 5.0]:
            add(f"chain={cb} t=0", chain_bonus=cb, threshold=0.0)
        for cb in [0.5, 1.0, 2.0]:
            add(f"chain={cb} t=0.3", chain_bonus=cb, threshold=0.3)

        # Jump penalty
        for jp in [0.3, 0.5, 1.0, 2.0]:
            add(f"jump_pen={jp} t=0", jump_pen=jp, threshold=0.0)
        for jp in [0.5, 1.0]:
            add(f"jump_pen={jp} t=0.3", jump_pen=jp, threshold=0.3)

        # Depth penalty
        for dp in [0.5, 1.0, 2.0]:
            add(f"depth={dp} t=0.3", depth_pen=dp, threshold=0.3)

        # Slope
        for sl in [-0.3, 0.3, 0.5]:
            add(f"slope={sl} t=0.3", slope=sl, threshold=0.3)

        # Warmup
        for w in [3, 10, 30, 60]:
            add(f"warmup={w} t=0", n_warmup=w, threshold=0.0)

        # Combos
        add("chain=2 jump=1 t=0", chain_bonus=2, jump_pen=1, threshold=0.0)
        add("chain=3 jump=1 t=0", chain_bonus=3, jump_pen=1, threshold=0.0)
        add("chain=5 jump=2 t=0", chain_bonus=5, jump_pen=2, threshold=0.0)
        add("a=1.3 chain=2 jp=1", a=1.3, chain_bonus=2, jump_pen=1, threshold=0.0)
        add("a=1.3 ch=2 jp=1 dp=0.5", a=1.3, chain_bonus=2, jump_pen=1, depth_pen=0.5, threshold=0.0)
        add("ch=2 jp=1 slope=0.3", chain_bonus=2, jump_pen=1, slope=0.3, threshold=0.0)
        add("ch=3 jp=2 w=5", chain_bonus=3, jump_pen=2, n_warmup=5, threshold=0.0)
    else:
        experiments = [(
            "custom",
            dict(a=args.a, b=1.0, threshold=args.threshold,
                 chain_bonus=args.chain_bonus, slope=args.slope,
                 depth_pen=args.depth_pen, n_warmup=args.n_warmup),
        )]

    print(f"{'Name':40s}  {'ratio':>8s}  {'Δ':>10s}  {'σ':>6s}  {'dev%':>6s}")
    print("-" * 78)

    for name, params in experiments:
        trial_args = [(W, ol, D, s[0], s[1], params) for s in seeds]
        t0 = time.monotonic()
        with mp.Pool(n_workers) as pool:
            results = pool.map(_worker, trial_args)
        dt = time.monotonic() - t0

        mse_vals = np.array([r[0] for r in results]) / k
        total_dev = sum(r[2] for r in results)
        total_p2 = sum(r[1] + r[2] for r in results)
        pct_dev = 100 * total_dev / total_p2 if total_p2 > 0 else 0

        diff = mse_vals - fwd
        md = np.mean(diff); sd = se(diff)
        sigma = md / sd if sd > 0 else 0
        star = "***" if sigma < -3 else "** " if sigma < -2 else "   "

        print(f"{name:40s}  {np.mean(mse_vals):8.5f}  {md:+.5f}  {sigma:+5.1f}  {pct_dev:5.1f}% {star}")


import os  # needed for cpu_count
