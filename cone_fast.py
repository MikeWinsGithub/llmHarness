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


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "fwd-merge": fwd_merge_queries,
    "blind-1/4+fwd": blind_quarter_then_fwd_queries,
    "sample-1/3+fwd": sample_third_then_fwd_queries,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_single_trial(args):
    """Run a single trial. Designed for multiprocessing."""
    widths_list, offsets_list, D, N, oracle_seed, strategy_seed, strategy_name = args
    widths = np.array(widths_list, dtype=np.int32)
    offsets = np.array(offsets_list, dtype=np.int32)
    total_q = sum(widths_list)
    max_qid = offsets[-1] + widths[-1]

    F_true = compute_F_true(widths, offsets, D, N, oracle_seed)

    # Get query order based on strategy
    strat_fn = STRATEGIES[strategy_name]
    query_order = strat_fn(widths_list, offsets_list, D, N, strategy_seed, oracle_seed)

    # Track MSE
    is_known = np.zeros(max_qid, dtype=np.bool_)
    qid_val = np.zeros(max_qid, dtype=np.int32)

    mse_list = np.zeros(total_q + 1)

    # MSE before any queries
    est = bayesian_estimate(widths, offsets, D, N, is_known, qid_val, max_qid)
    mse_list[0] = (est - F_true) ** 2

    qi = 0
    for qid, layer, nv in query_order:
        val = oracle_query_single(oracle_seed, qid, nv)
        is_known[qid] = True
        qid_val[qid] = val
        qi += 1
        est = bayesian_estimate(widths, offsets, D, N, is_known, qid_val, max_qid)
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
