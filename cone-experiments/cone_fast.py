"""Fast cone experiment harness using numba-JIT'd DP.

~100x faster than cone_experiments.py for the Bayesian estimate.
"""

import numpy as np
from numba import njit
import numba
import time
import multiprocessing as mp
from functools import partial


# ---------------------------------------------------------------------------
# Numba-accelerated core
# ---------------------------------------------------------------------------

@njit
def oracle_query_single(seed, qid, num_values):
    """Deterministic hash oracle using splitmix64-style mixing."""
    h = np.uint64(seed) * np.uint64(6364136223846793005) + np.uint64(qid) * np.uint64(1442695040888963407)
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xbf58476d1ce4e5b9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94d049bb133111eb)
    h ^= h >> np.uint64(31)
    return int(h % np.uint64(num_values))


@njit
def compute_F_true(widths, offsets, D, seed):
    total = 0.0
    W0 = widths[0]
    for v in range(W0):
        cur = v
        for layer in range(D - 1):
            qid = offsets[layer] + cur
            cur = oracle_query_single(seed, qid, widths[layer + 1])
        qid = offsets[D - 1] + cur
        sign_raw = oracle_query_single(seed, qid, 2)
        total += 1.0 if sign_raw == 1 else -1.0
    return total / W0


@njit
def bayesian_estimate(widths, offsets, D, is_known, qid_val, max_qid):
    """Exact Bayesian DP using pre-built arrays."""
    val = np.zeros(widths[D - 1], dtype=np.float64)
    for v in range(widths[D - 1]):
        qid = offsets[D - 1] + v
        if qid < max_qid and is_known[qid]:
            val[v] = 1.0 if qid_val[qid] == 1 else -1.0

    for layer in range(D - 2, -1, -1):
        new_val = np.zeros(widths[layer], dtype=np.float64)
        W_next = widths[layer + 1]
        # Precompute mean — same for all unknown vertices
        val_mean = 0.0
        for nv in range(W_next):
            val_mean += val[nv]
        val_mean /= W_next
        for v in range(widths[layer]):
            qid = offsets[layer] + v
            if qid < max_qid and is_known[qid]:
                new_val[v] = val[qid_val[qid]]
            else:
                new_val[v] = val_mean
        val = new_val

    W0 = widths[0]
    total = 0.0
    for v in range(W0):
        total += val[v]
    return total / W0


# ---------------------------------------------------------------------------
# Incremental DP
# ---------------------------------------------------------------------------

@njit
def _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, layer, next_layer_val):
    """Compute val array for a single layer given the next layer's val array."""
    W = widths[layer]
    W_next = widths[layer + 1]
    # Precompute mean of next layer — same for all unknown vertices
    next_mean = 0.0
    for nv in range(W_next):
        next_mean += next_layer_val[nv]
    next_mean /= W_next
    val = np.zeros(W, dtype=np.float64)
    for v in range(W):
        qid = offsets[layer] + v
        if qid < max_qid and is_known[qid]:
            val[v] = next_layer_val[qid_val[qid]]
        else:
            val[v] = next_mean
    return val


@njit
def _compute_terminal_val(widths, offsets, D, is_known, qid_val, max_qid):
    """Compute val array for the terminal layer."""
    W = widths[D - 1]
    val = np.zeros(W, dtype=np.float64)
    for v in range(W):
        qid = offsets[D - 1] + v
        if qid < max_qid and is_known[qid]:
            val[v] = 1.0 if qid_val[qid] == 1 else -1.0
    return val


def _init_layer_vals(widths, offsets, D, is_known, qid_val, max_qid):
    """Build all layer val arrays from scratch."""
    layer_vals = [None] * D
    layer_vals[D - 1] = _compute_terminal_val(widths, offsets, D, is_known, qid_val, max_qid)
    for layer in range(D - 2, -1, -1):
        layer_vals[layer] = _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, layer, layer_vals[layer + 1])
    return layer_vals


def _update_layer_vals(widths, offsets, D, is_known, qid_val, max_qid, layer_vals, changed_layer):
    """Recompute layer vals from changed_layer up to layer 0."""
    if changed_layer == D - 1:
        layer_vals[D - 1] = _compute_terminal_val(widths, offsets, D, is_known, qid_val, max_qid)
    else:
        layer_vals[changed_layer] = _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, changed_layer, layer_vals[changed_layer + 1])
    for layer in range(changed_layer - 1, -1, -1):
        layer_vals[layer] = _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, layer, layer_vals[layer + 1])


@njit
def _estimate_from_layer0(layer0_val, W0):
    """Compute F estimate from layer 0 values."""
    total = 0.0
    for v in range(W0):
        total += layer0_val[v]
    return total / W0


@njit
def _all_equal(arr):
    """Check if all elements of an array are equal."""
    if len(arr) <= 1:
        return True
    v0 = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != v0:
            return False
    return True


# ---------------------------------------------------------------------------
# Query allocation strategies (pure Python, feeds into numba DP)
# ---------------------------------------------------------------------------

def _oq_helper(known, oracle_seed, qid, nv):
    if qid not in known:
        known[qid] = oracle_query_single(oracle_seed, qid, nv)
    return known[qid]


def _fwd_merge_from(widths, offsets, D, rng, oracle_seed, known, order):
    """Trace each starting vertex forward. Appends to order list."""
    W0 = widths[0]
    for v in rng.permutation(W0):
        v = int(v)
        for layer in range(D - 1):
            qid = offsets[layer] + v
            nv = widths[layer + 1]
            if qid not in known:
                order.append((qid, layer, nv))
            v = _oq_helper(known, oracle_seed, qid, nv)
        qid = offsets[D - 1] + v
        if qid not in known:
            order.append((qid, D - 1, 2))
            _oq_helper(known, oracle_seed, qid, 2)


def fwd_merge_queries(widths, offsets, D, seed, oracle_seed):
    """Forward merge: trace each starting vertex forward."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    _fwd_merge_from(widths, offsets, D, rng, oracle_seed, known, order)
    return order


def blind_quarter_then_fwd_queries(widths, offsets, D, seed, oracle_seed):
    """Reveal all vertices at layer D//4, then fwd-merge the rest."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    target_layer = max(0, D // 4)
    for v in range(widths[target_layer]):
        qid = offsets[target_layer] + v
        nv = widths[target_layer + 1] if target_layer < D - 1 else 2
        if qid not in known:
            order.append((qid, target_layer, nv))
            _oq_helper(known, oracle_seed, qid, nv)
    _fwd_merge_from(widths, offsets, D, rng, oracle_seed, known, order)
    return order


def sample_third_then_fwd_queries(widths, offsets, D, seed, oracle_seed):
    """Trace ONE random vertex starting from layer D//3 to the terminal,
    then fwd-merge everything from the start."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    start_layer = max(0, D // 3)

    v = int(rng.integers(0, widths[start_layer]))
    for layer in range(start_layer, D - 1):
        qid = offsets[layer] + v
        nv = widths[layer + 1]
        if qid not in known:
            order.append((qid, layer, nv))
        v = _oq_helper(known, oracle_seed, qid, nv)
    qid = offsets[D - 1] + v
    if qid not in known:
        order.append((qid, D - 1, 2))
        _oq_helper(known, oracle_seed, qid, 2)

    _fwd_merge_from(widths, offsets, D, rng, oracle_seed, known, order)
    return order


def sample_then_fwd_queries(widths, offsets, D, seed, oracle_seed, frac=0.33, n_probes=1):
    """Trace n_probes random vertices from layer D*frac to terminal, then fwd-merge."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    start_layer = max(0, min(D - 1, int(D * frac)))

    for _ in range(n_probes):
        v = int(rng.integers(0, widths[start_layer]))
        for layer in range(start_layer, D - 1):
            qid = offsets[layer] + v
            nv = widths[layer + 1]
            if qid not in known:
                order.append((qid, layer, nv))
            v = _oq_helper(known, oracle_seed, qid, nv)
        qid = offsets[D - 1] + v
        if qid not in known:
            order.append((qid, D - 1, 2))
            _oq_helper(known, oracle_seed, qid, 2)

    _fwd_merge_from(widths, offsets, D, rng, oracle_seed, known, order)
    return order


# Wrapper functions for each (frac, n_probes) combo so they're picklable
def _s25x1(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.25, 1)
def _s25x3(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.25, 3)
def _s25x5(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.25, 5)
def _s33x1(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.33, 1)
def _s33x3(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.33, 3)
def _s33x5(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.33, 5)
def _s50x1(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.50, 1)
def _s50x3(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.50, 3)
def _s50x5(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.50, 5)
def _s67x1(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.67, 1)
def _s67x3(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.67, 3)
def _s67x5(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.67, 5)
def _s75x1(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.75, 1)
def _s75x3(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.75, 3)
def _s75x5(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.75, 5)
def _s90x1(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.90, 1)
def _s90x3(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.90, 3)
def _s90x5(w, o, D, s, os): return sample_then_fwd_queries(w, o, D, s, os, 0.90, 5)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

def multi_pass_fwd_queries(widths, offsets, D, seed, oracle_seed):
    """Multiple passes: each pass traces all vertices, one query per vertex per pass."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    W0 = widths[0]
    for _pass in range(D + 1):
        made_progress = False
        for v0 in rng.permutation(W0):
            v = int(v0)
            for layer in range(D - 1):
                qid = offsets[layer] + v
                nv = widths[layer + 1]
                if qid not in known:
                    order.append((qid, layer, nv))
                    _oq_helper(known, oracle_seed, qid, nv)
                    made_progress = True
                    break
                v = known[qid]
            else:
                qid = offsets[D - 1] + v
                if qid not in known:
                    order.append((qid, D - 1, 2))
                    _oq_helper(known, oracle_seed, qid, 2)
                    made_progress = True
        if not made_progress:
            break
    return order


def breadth_first_queries(widths, offsets, D, seed, oracle_seed):
    """Cycle through all starting vertices, reveal one query per vertex per round."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    W0 = widths[0]
    v0_order = list(rng.permutation(W0))
    for _round in range(D + 1):
        for v0 in v0_order:
            v = int(v0)
            for layer in range(D - 1):
                qid = offsets[layer] + v
                nv = widths[layer + 1]
                if qid not in known:
                    order.append((qid, layer, nv))
                    _oq_helper(known, oracle_seed, qid, nv)
                    break
                v = known[qid]
            else:
                qid = offsets[D - 1] + v
                if qid not in known:
                    order.append((qid, D - 1, 2))
                    _oq_helper(known, oracle_seed, qid, 2)
    return order


def popular_first_queries(widths, offsets, D, seed, oracle_seed):
    """Fwd-merge but explore starting vertices in order. (All equal weight now.)"""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    W0 = widths[0]
    for v0 in rng.permutation(W0):
        v = int(v0)
        for layer in range(D - 1):
            qid = offsets[layer] + v
            nv = widths[layer + 1]
            if qid not in known:
                order.append((qid, layer, nv))
            v = _oq_helper(known, oracle_seed, qid, nv)
        qid = offsets[D - 1] + v
        if qid not in known:
            order.append((qid, D - 1, 2))
            _oq_helper(known, oracle_seed, qid, 2)
    return order


def rare_first_queries(widths, offsets, D, seed, oracle_seed):
    """Fwd-merge but explore starting vertices in order. (All equal weight now.)"""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    W0 = widths[0]
    for v0 in rng.permutation(W0):
        v = int(v0)
        for layer in range(D - 1):
            qid = offsets[layer] + v
            nv = widths[layer + 1]
            if qid not in known:
                order.append((qid, layer, nv))
            v = _oq_helper(known, oracle_seed, qid, nv)
        qid = offsets[D - 1] + v
        if qid not in known:
            order.append((qid, D - 1, 2))
            _oq_helper(known, oracle_seed, qid, 2)
    return order


def interleaved_queries(widths, offsets, D, seed, oracle_seed):
    """Alternate: trace one starting vertex forward, then probe one random terminal vertex."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    W0 = widths[0]
    v0_list = list(rng.permutation(W0))
    back_verts = list(rng.permutation(widths[D - 1]))
    ii, bi = 0, 0
    while ii < len(v0_list) or bi < len(back_verts):
        if ii < len(v0_list):
            v = int(v0_list[ii]); ii += 1
            for layer in range(D - 1):
                qid = offsets[layer] + v
                nv = widths[layer + 1]
                if qid not in known:
                    order.append((qid, layer, nv))
                v = _oq_helper(known, oracle_seed, qid, nv)
            qid = offsets[D - 1] + v
            if qid not in known:
                order.append((qid, D - 1, 2))
                _oq_helper(known, oracle_seed, qid, 2)
        if bi < len(back_verts):
            v = int(back_verts[bi]); bi += 1
            qid = offsets[D - 1] + v
            if qid not in known:
                order.append((qid, D - 1, 2))
                _oq_helper(known, oracle_seed, qid, 2)
    return order


def two_pass_queries(widths, offsets, D, seed, oracle_seed):
    """Pass 1: reveal only layer 0. Pass 2: full fwd-merge (layer 0 already cached)."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    W0 = widths[0]
    for v0 in rng.permutation(W0):
        v = int(v0)
        qid = offsets[0] + v
        nv = widths[1]
        if qid not in known:
            order.append((qid, 0, nv))
            _oq_helper(known, oracle_seed, qid, nv)
    _fwd_merge_from(widths, offsets, D, rng, oracle_seed, known, order)
    return order


STRATEGIES = {
    "fwd-merge": fwd_merge_queries,
    "blind-1/4+fwd": blind_quarter_then_fwd_queries,
    "sample-1/3+fwd": sample_third_then_fwd_queries,
    "s25%x1+fwd": _s25x1, "s25%x3+fwd": _s25x3, "s25%x5+fwd": _s25x5,
    "s33%x1+fwd": _s33x1, "s33%x3+fwd": _s33x3, "s33%x5+fwd": _s33x5,
    "s50%x1+fwd": _s50x1, "s50%x3+fwd": _s50x3, "s50%x5+fwd": _s50x5,
    "s67%x1+fwd": _s67x1, "s67%x3+fwd": _s67x3, "s67%x5+fwd": _s67x5,
    "s75%x1+fwd": _s75x1, "s75%x3+fwd": _s75x3, "s75%x5+fwd": _s75x5,
    "s90%x1+fwd": _s90x1, "s90%x3+fwd": _s90x3, "s90%x5+fwd": _s90x5,
    "multi-pass": multi_pass_fwd_queries,
    "breadth-1st": breadth_first_queries,
    "popular-1st": popular_first_queries,
    "rare-1st": rare_first_queries,
    "interleaved": interleaved_queries,
    "two-pass": two_pass_queries,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def probe_then_fwd_queries(widths, offsets, D, seed, oracle_seed, probe_spec):
    """General probe strategy: probe_spec is list of (layer_index, n_probes).
    Do probes first, then fwd-merge."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    for layer_idx, n in probe_spec:
        layer = max(0, min(D - 1, layer_idx))
        for _ in range(n):
            v = int(rng.integers(0, widths[layer]))
            for l in range(layer, D - 1):
                qid = offsets[l] + v
                nv = widths[l + 1]
                if qid not in known:
                    order.append((qid, l, nv))
                v = _oq_helper(known, oracle_seed, qid, nv)
            qid = offsets[D - 1] + v
            if qid not in known:
                order.append((qid, D - 1, 2))
                _oq_helper(known, oracle_seed, qid, 2)
    _fwd_merge_from(widths, offsets, D, rng, oracle_seed, known, order)
    return order


def evaluate_single_trial(args):
    """Run a single trial. Designed for multiprocessing."""
    widths_list, offsets_list, D, oracle_seed, strategy_seed, strategy_spec = args
    widths = np.array(widths_list, dtype=np.int32)
    offsets = np.array(offsets_list, dtype=np.int32)
    total_q = sum(widths_list)
    max_qid = offsets[-1] + widths[-1]
    W0 = widths[0]

    F_true = compute_F_true(widths, offsets, D, oracle_seed)

    if isinstance(strategy_spec, str):
        strat_fn = STRATEGIES[strategy_spec]
        query_order = strat_fn(widths_list, offsets_list, D, strategy_seed, oracle_seed)
    else:
        query_order = probe_then_fwd_queries(widths_list, offsets_list, D, strategy_seed, oracle_seed, strategy_spec)

    is_known = np.zeros(max_qid, dtype=np.bool_)
    qid_val = np.zeros(max_qid, dtype=np.int32)
    mse_list = np.zeros(total_q + 1)

    F_true_sq = F_true * F_true
    est = 0.0
    layer_vals = None
    terminal_known = False
    deepest_pending = -1

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
                est = _estimate_from_layer0(layer_vals[0], W0)
                deepest_pending = -1
            else:
                if qi < len(mse_list):
                    mse_list[qi] = F_true_sq
                continue
        else:
            if layer_idx == D - 1:
                update_from = D - 1
                if deepest_pending >= 0:
                    update_from = max(update_from, deepest_pending)
                _update_layer_vals(widths, offsets, D, is_known, qid_val, max_qid, layer_vals, update_from)
                est = _estimate_from_layer0(layer_vals[0], W0)
                deepest_pending = -1
            else:
                if deepest_pending < 0 or layer_idx > deepest_pending:
                    deepest_pending = layer_idx
                _update_layer_vals(widths, offsets, D, is_known, qid_val, max_qid, layer_vals, layer_idx)
                est = _estimate_from_layer0(layer_vals[0], W0)

        mse = (est - F_true) ** 2
        if qi < len(mse_list):
            mse_list[qi] = mse
        if abs(est - F_true) < 1e-15:
            break

    mse_list[0] = F_true_sq
    final_mse = (est - F_true) ** 2 if terminal_known else F_true_sq
    for j in range(qi + 1, total_q + 1):
        mse_list[j] = final_mse

    return mse_list


def evaluate_cone_fast(widths_list, num_samples=500, seed=42, n_workers=4, strategy="fwd-merge"):
    """Evaluate a strategy on a cone using numba + multiprocessing."""
    D = len(widths_list)
    offsets_list = [sum(widths_list[:i]) for i in range(D)]
    total_q = sum(widths_list)

    rng = np.random.default_rng(seed)
    args_list = []
    for _ in range(num_samples):
        oracle_seed = int(rng.integers(0, 2**62))
        strategy_seed = int(rng.integers(0, 2**62))
        args_list.append((widths_list, offsets_list, D, oracle_seed, strategy_seed, strategy))

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(evaluate_single_trial, args_list)
    else:
        results = [evaluate_single_trial(a) for a in args_list]

    mse_accum = np.zeros(total_q + 1)
    for r in results:
        mse_accum += r
    mse_accum /= len(results)

    cumul = float(np.sum(mse_accum))
    k = D
    return cumul, k, len(results)


def funnel(D):
    return list(range(2 * D, D, -1))


# ---------------------------------------------------------------------------
# Warmup & main
# ---------------------------------------------------------------------------

def warmup():
    """JIT compile numba functions."""
    w = np.array([4, 4], dtype=np.int32)
    o = np.array([0, 4], dtype=np.int32)
    ik = np.zeros(8, dtype=np.bool_)
    iv = np.zeros(8, dtype=np.int32)
    compute_F_true(w, o, 2, 42)
    bayesian_estimate(w, o, 2, ik, iv, 8)
    oracle_query_single(42, 0, 2)


if __name__ == "__main__":
    print("Warming up JIT...", flush=True)
    warmup()
    print("Done.\n")

    strat_names = list(STRATEGIES.keys())
    header = f"{'Funnel':15s}  {'k':>3s}  " + "  ".join(f"{s:>14s}" for s in strat_names)
    print(header)
    print("-" * len(header))

    for D in range(5, 31):
        w = funnel(D)
        k = D
        tq = sum(w)
        n = 1000 if tq < 100 else 500 if tq < 500 else 200 if tq < 2000 else 50
        results = []
        for sname in strat_names:
            t0 = time.monotonic()
            cumul, _, trials = evaluate_cone_fast(w, num_samples=n, n_workers=4, strategy=sname)
            elapsed = time.monotonic() - t0
            ratio = cumul / k
            s = '✓' if ratio <= 1 + 1e-6 else '✗'
            results.append(f"{ratio:.4f}{s}({trials:3d})")
        print(f"D={D:<3d} [{w[0]:>3d}..{w[-1]:>2d}]  {k:3d}  " + "  ".join(results), flush=True)
