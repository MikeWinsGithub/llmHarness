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

def fwd_merge_queries(widths, offsets, D, N, seed, oracle_seed, budget=None):
    """Forward merge: trace random inputs, yield (qid, layer, num_values) tuples in order."""
    rng = np.random.default_rng(seed)
    # Build oracle values on demand
    known = {}

    def oq(qid, nv):
        if qid not in known:
            known[qid] = oracle_query_single(oracle_seed, qid, nv)
        return known[qid]

    order = []
    for x_int in rng.permutation(1 << N):
        v = int(x_int) % widths[0]
        for layer in range(D - 1):
            qid = offsets[layer] + v
            nv = widths[layer + 1]
            if qid not in known:
                order.append((qid, layer, nv))
            v = oq(qid, nv)
        qid = offsets[D - 1] + v
        if qid not in known:
            order.append((qid, D - 1, 2))
            oq(qid, 2)
    return order


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_single_trial(args):
    """Run a single trial. Designed for multiprocessing."""
    widths_list, offsets_list, D, N, oracle_seed, strategy_seed = args
    widths = np.array(widths_list, dtype=np.int32)
    offsets = np.array(offsets_list, dtype=np.int32)
    total_q = sum(widths_list)
    max_qid = offsets[-1] + widths[-1]

    F_true = compute_F_true(widths, offsets, D, N, oracle_seed)

    # Get query order
    query_order = fwd_merge_queries(widths_list, offsets_list, D, N, strategy_seed, oracle_seed)

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


def evaluate_cone_fast(widths_list, N=8, num_samples=500, seed=42, n_workers=4):
    """Evaluate fwd-merge on a cone using numba + multiprocessing."""
    D = len(widths_list)
    offsets_list = [sum(widths_list[:i]) for i in range(D)]
    total_q = sum(widths_list)

    rng = np.random.default_rng(seed)
    args_list = []
    for _ in range(num_samples):
        oracle_seed = int(rng.integers(0, 2**62))
        strategy_seed = int(rng.integers(0, 2**62))
        args_list.append((widths_list, offsets_list, D, N, oracle_seed, strategy_seed))

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

    print(f"{'Funnel':15s}  {'k':>3s}  {'ratio':>8s}  {'cumul':>8s}  {'trials':>6s}  {'time':>6s}")
    print("-" * 55)

    for D in range(5, 51):
        w = funnel(D)
        k = D
        n = 1000 if sum(w) < 100 else 500 if sum(w) < 500 else 200 if sum(w) < 2000 else 50
        t0 = time.monotonic()
        cumul, k, trials = evaluate_cone_fast(w, N=8, num_samples=n, n_workers=4)
        elapsed = time.monotonic() - t0
        ratio = cumul / k
        s = '✓' if ratio <= 1 + 1e-6 else '✗'
        print(f"D={D:<3d} [{w[0]}..{w[-1]}]  {k:3d}  {ratio:8.5f}{s}  {cumul:8.3f}  {trials:6d}  {elapsed:5.1f}s", flush=True)
