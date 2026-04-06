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
    """Deterministic hash oracle."""
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
def bayesian_estimate(widths, offsets, D, N, is_known, qid_val, max_qid):
    """Exact Bayesian DP using pre-built arrays."""
    val = np.zeros(widths[D - 1], dtype=np.float64)
    for v in range(widths[D - 1]):
        qid = offsets[D - 1] + v
        if qid < max_qid and is_known[qid]:
            val[v] = 1.0 if qid_val[qid] == 1 else -1.0

    for layer in range(D - 2, -1, -1):
        new_val = np.zeros(widths[layer], dtype=np.float64)
        W_next = widths[layer + 1]
        for v in range(widths[layer]):
            qid = offsets[layer] + v
            if qid < max_qid and is_known[qid]:
                next_v = qid_val[qid]
                new_val[v] = val[next_v]
            else:
                total = 0.0
                for nv in range(W_next):
                    total += val[nv]
                new_val[v] = total / W_next
        val = new_val

    W0 = widths[0]
    total_F = 0.0
    count_per = (1 << N) // W0
    remainder = (1 << N) % W0
    for v0 in range(W0):
        c = count_per + (1 if v0 < remainder else 0)
        total_F += val[v0] * c
    return total_F / (1 << N)


# ---------------------------------------------------------------------------
# Incremental DP
# ---------------------------------------------------------------------------

@njit
def _compute_layer_val(widths, offsets, D, is_known, qid_val, max_qid, layer, next_layer_val):
    """Compute val array for a single layer given the next layer's val array."""
    W = widths[layer]
    W_next = widths[layer + 1]
    val = np.zeros(W, dtype=np.float64)
    for v in range(W):
        qid = offsets[layer] + v
        if qid < max_qid and is_known[qid]:
            next_v = qid_val[qid]
            val[v] = next_layer_val[next_v]
        else:
            total = 0.0
            for nv in range(W_next):
                total += next_layer_val[nv]
            val[v] = total / W_next
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
def _estimate_from_layer0(layer0_val, W0, N):
    """Compute F estimate from layer 0 values."""
    total_F = 0.0
    count_per = (1 << N) // W0
    remainder = (1 << N) % W0
    for v0 in range(W0):
        c = count_per + (1 if v0 < remainder else 0)
        total_F += layer0_val[v0] * c
    return total_F / (1 << N)


# ---------------------------------------------------------------------------
# Query allocation strategies (pure Python, feeds into numba DP)
# ---------------------------------------------------------------------------

def _oq_helper(known, oracle_seed, qid, nv):
    if qid not in known:
        known[qid] = oracle_query_single(oracle_seed, qid, nv)
    return known[qid]


def _fwd_merge_from(widths, offsets, D, N, rng, oracle_seed, known, order):
    """Trace random inputs from layer 0 forward. Appends to order list."""
    for x_int in rng.permutation(1 << N):
        v = int(x_int) % widths[0]
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


def fwd_merge_queries(widths, offsets, D, N, seed, oracle_seed):
    """Forward merge: trace random inputs layer 0 -> terminal."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    _fwd_merge_from(widths, offsets, D, N, rng, oracle_seed, known, order)
    return order


def blind_quarter_then_fwd_queries(widths, offsets, D, N, seed, oracle_seed):
    """Reveal all vertices at layer D//4, then fwd-merge the rest.
    The D//4 layer is near the start (narrow end for widening cones, wide for narrowing)."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    target_layer = max(0, D // 4)
    # Reveal all vertices at target layer
    for v in range(widths[target_layer]):
        qid = offsets[target_layer] + v
        nv = widths[target_layer + 1] if target_layer < D - 1 else 2
        if qid not in known:
            order.append((qid, target_layer, nv))
            _oq_helper(known, oracle_seed, qid, nv)
    # Then fwd-merge
    _fwd_merge_from(widths, offsets, D, N, rng, oracle_seed, known, order)
    return order


def sample_third_then_fwd_queries(widths, offsets, D, N, seed, oracle_seed):
    """Trace ONE random input starting from layer D//3 to the terminal,
    then fwd-merge everything from the start."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    start_layer = max(0, D // 3)

    # Pick one random vertex at start_layer and trace forward
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

    # Then fwd-merge from the start
    _fwd_merge_from(widths, offsets, D, N, rng, oracle_seed, known, order)
    return order


def sample_then_fwd_queries(widths, offsets, D, N, seed, oracle_seed, frac=0.33, n_probes=1):
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

    _fwd_merge_from(widths, offsets, D, N, rng, oracle_seed, known, order)
    return order


# Wrapper functions for each (frac, n_probes) combo so they're picklable
def _s25x1(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.25, 1)
def _s25x3(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.25, 3)
def _s25x5(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.25, 5)
def _s33x1(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.33, 1)
def _s33x3(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.33, 3)
def _s33x5(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.33, 5)
def _s50x1(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.50, 1)
def _s50x3(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.50, 3)
def _s50x5(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.50, 5)
def _s67x1(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.67, 1)
def _s67x3(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.67, 3)
def _s67x5(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.67, 5)
def _s75x1(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.75, 1)
def _s75x3(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.75, 3)
def _s75x5(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.75, 5)
def _s90x1(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.90, 1)
def _s90x3(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.90, 3)
def _s90x5(w, o, D, N, s, os): return sample_then_fwd_queries(w, o, D, N, s, os, 0.90, 5)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

def multi_pass_fwd_queries(widths, offsets, D, N, seed, oracle_seed):
    """Multiple passes: each pass traces all inputs, one query per input per pass."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    for _pass in range(D + 1):
        made_progress = False
        for x_int in rng.permutation(1 << N):
            v = int(x_int) % widths[0]
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


def breadth_first_queries(widths, offsets, D, N, seed, oracle_seed):
    """Cycle through all inputs, reveal one query per input per round."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    input_order = list(rng.permutation(1 << N))
    for _round in range(D + 1):
        for x_int in input_order:
            v = int(x_int) % widths[0]
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


def popular_first_queries(widths, offsets, D, N, seed, oracle_seed):
    """Fwd-merge but explore most common starting vertices first."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    counts = {}
    for x_int in range(1 << N):
        v = x_int % widths[0]
        counts[v] = counts.get(v, 0) + 1
    inputs = list(range(1 << N))
    rng.shuffle(inputs)
    inputs.sort(key=lambda x: -counts[x % widths[0]])
    for x_int in inputs:
        v = int(x_int) % widths[0]
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


def rare_first_queries(widths, offsets, D, N, seed, oracle_seed):
    """Fwd-merge but explore rarest starting vertices first."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    counts = {}
    for x_int in range(1 << N):
        v = x_int % widths[0]
        counts[v] = counts.get(v, 0) + 1
    inputs = list(range(1 << N))
    rng.shuffle(inputs)
    inputs.sort(key=lambda x: counts[x % widths[0]])
    for x_int in inputs:
        v = int(x_int) % widths[0]
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


def interleaved_queries(widths, offsets, D, N, seed, oracle_seed):
    """Alternate: trace one input forward, then probe one random terminal vertex."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    inputs = list(rng.permutation(1 << N))
    back_verts = list(rng.permutation(widths[D - 1]))
    ii, bi = 0, 0
    while ii < len(inputs) or bi < len(back_verts):
        if ii < len(inputs):
            x_int = int(inputs[ii]); ii += 1
            v = x_int % widths[0]
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


def two_pass_queries(widths, offsets, D, N, seed, oracle_seed):
    """Pass 1: reveal only layer 0. Pass 2: full fwd-merge (layer 0 already cached)."""
    rng = np.random.default_rng(seed)
    known = {}
    order = []
    for x_int in rng.permutation(1 << N):
        v = int(x_int) % widths[0]
        qid = offsets[0] + v
        nv = widths[1]
        if qid not in known:
            order.append((qid, 0, nv))
            _oq_helper(known, oracle_seed, qid, nv)
    _fwd_merge_from(widths, offsets, D, N, rng, oracle_seed, known, order)
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

def probe_then_fwd_queries(widths, offsets, D, N, seed, oracle_seed, probe_spec):
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
    _fwd_merge_from(widths, offsets, D, N, rng, oracle_seed, known, order)
    return order


def evaluate_single_trial(args):
    """Run a single trial. Designed for multiprocessing."""
    widths_list, offsets_list, D, N, oracle_seed, strategy_seed, strategy_spec = args
    widths = np.array(widths_list, dtype=np.int32)
    offsets = np.array(offsets_list, dtype=np.int32)
    total_q = sum(widths_list)
    max_qid = offsets[-1] + widths[-1]

    F_true = compute_F_true(widths, offsets, D, N, oracle_seed)

    # Get query order: strategy_spec is either a string (named strategy) or a list (probe spec)
    if isinstance(strategy_spec, str):
        strat_fn = STRATEGIES[strategy_spec]
        query_order = strat_fn(widths_list, offsets_list, D, N, strategy_seed, oracle_seed)
    else:
        # It's a probe spec: list of (layer_index, n_probes)
        query_order = probe_then_fwd_queries(widths_list, offsets_list, D, N, strategy_seed, oracle_seed, strategy_spec)

    # Track MSE using incremental DP
    is_known = np.zeros(max_qid, dtype=np.bool_)
    qid_val = np.zeros(max_qid, dtype=np.int32)

    mse_list = np.zeros(total_q + 1)

    # Initialize layer_vals: val[layer] = array of per-vertex values
    # Build initial DP (nothing known)
    layer_vals = _init_layer_vals(widths, offsets, D, is_known, qid_val, max_qid)
    est = _estimate_from_layer0(layer_vals[0], widths[0], N)
    mse_list[0] = (est - F_true) ** 2

    qi = 0
    for qid, layer_idx, nv in query_order:
        val = oracle_query_single(oracle_seed, qid, nv)
        is_known[qid] = True
        qid_val[qid] = val
        qi += 1
        # Incremental update: only recompute layers layer_idx down to 0
        _update_layer_vals(widths, offsets, D, is_known, qid_val, max_qid, layer_vals, layer_idx)
        est = _estimate_from_layer0(layer_vals[0], widths[0], N)
        mse = (est - F_true) ** 2
        if qi < len(mse_list):
            mse_list[qi] = mse
        if abs(est - F_true) < 1e-15:
            break

    # Fill tail with final MSE
    final_mse = (est - F_true) ** 2
    for j in range(qi + 1, total_q + 1):
        mse_list[j] = final_mse

    return mse_list


def evaluate_cone_fast(widths_list, N=8, num_samples=500, seed=42, n_workers=4, strategy="fwd-merge"):
    """Evaluate a strategy on a cone using numba + multiprocessing."""
    D = len(widths_list)
    offsets_list = [sum(widths_list[:i]) for i in range(D)]
    total_q = sum(widths_list)

    rng = np.random.default_rng(seed)
    args_list = []
    for _ in range(num_samples):
        oracle_seed = int(rng.integers(0, 2**62))
        strategy_seed = int(rng.integers(0, 2**62))
        args_list.append((widths_list, offsets_list, D, N, oracle_seed, strategy_seed, strategy))

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
    compute_F_true(w, o, 2, 4, 42)
    bayesian_estimate(w, o, 2, 4, ik, iv, 8)
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
            cumul, _, trials = evaluate_cone_fast(w, N=8, num_samples=n, n_workers=4, strategy=sname)
            elapsed = time.monotonic() - t0
            ratio = cumul / k
            s = '✓' if ratio <= 1 + 1e-6 else '✗'
            results.append(f"{ratio:.4f}{s}({trials:3d})")
        print(f"D={D:<3d} [{w[0]:>3d}..{w[-1]:>2d}]  {k:3d}  " + "  ".join(results), flush=True)
