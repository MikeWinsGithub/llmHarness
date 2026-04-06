"""Fast experiment harness for cone sequential conjecture.

Usage:
    python cone_experiments.py                    # run default experiments
    python cone_experiments.py --cone "10,9,8,7"  # specific cone
"""

from cone_study import _exact_bayesian_estimate, Oracle, run_cone_on_input, cone_k, cone_total_queries
import numpy as np
import time
import sys
import json


def evaluate(widths, strategy_fn, N=8, num_samples=300, seed=42, timeout=120):
    """Evaluate a strategy. Returns (cumul_mse, k, trials)."""
    k = cone_k(widths)
    D = len(widths)
    total_q = sum(widths)
    offsets = [sum(widths[:i]) for i in range(D)]
    rng = np.random.default_rng(seed)
    mse_accum = np.zeros(total_q + 1)
    trials = 0
    deadline = time.monotonic() + timeout
    for _ in range(num_samples):
        if time.monotonic() >= deadline:
            break
        oseed = int(rng.integers(0, 2**62))
        to = Oracle(seed=oseed)
        F_true = sum(run_cone_on_input(
            np.array([(xi >> b) & 1 for b in range(N)], dtype=np.int8),
            widths, to) for xi in range(2**N)) / 2**N
        so = Oracle(seed=oseed)
        queried = set()
        est = _exact_bayesian_estimate(widths, so, queried, N)
        mse_accum[0] += (est - F_true)**2
        tr = np.random.default_rng(rng.integers(0, 2**62))
        qi = 0
        for qid, layer in strategy_fn(widths, so, queried, N, tr, offsets):
            nv = widths[layer + 1] if layer < D - 1 else 2
            so.query(qid, num_values=nv)
            queried.add(qid)
            qi += 1
            est = _exact_bayesian_estimate(widths, so, queried, N)
            if qi < len(mse_accum):
                mse_accum[qi] += (est - F_true)**2
            if abs(est - F_true) < 1e-15:
                break
        trials += 1
    mse = mse_accum / max(trials, 1)
    return float(np.sum(mse)), k, trials


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def fwd_merge(widths, oracle, queried, N, rng, offsets):
    """Standard: trace random inputs layer 0 -> terminal."""
    D = len(widths)
    for x_int in rng.permutation(2**N):
        v = int(x_int) % widths[0]
        for layer in range(D - 1):
            qid = offsets[layer] + v
            if qid not in queried:
                yield (qid, layer)
            if qid not in queried:
                break
            v = oracle._values[qid]
        else:
            qid = offsets[D - 1] + v
            if qid not in queried:
                yield (qid, D - 1)


def blind_schedule(layer_schedule):
    """Blind trace: for each layer L in schedule, sample random vertices at L
    and trace forward to terminal."""
    def strat(widths, oracle, queried, N, rng, offsets):
        D = len(widths)
        for start_layer in layer_schedule:
            if start_layer >= D:
                continue
            for v in rng.permutation(widths[start_layer]):
                v = int(v)
                cur_v = v
                for layer in range(start_layer, D - 1):
                    qid = offsets[layer] + cur_v
                    if qid not in queried:
                        yield (qid, layer)
                    if qid in queried:
                        cur_v = oracle._values[qid]
                    else:
                        break
                else:
                    qid = offsets[D - 1] + cur_v
                    if qid not in queried:
                        yield (qid, D - 1)
    return strat


def blind_last(widths, oracle, queried, N, rng, offsets):
    """Blind from terminal layer only."""
    return blind_schedule([len(widths) - 1])(widths, oracle, queried, N, rng, offsets)


def blind_narrow_first(widths, oracle, queried, N, rng, offsets):
    """Layers sorted by width, narrowest first."""
    D = len(widths)
    order = sorted(range(D), key=lambda l: widths[l])
    return blind_schedule(order)(widths, oracle, queried, N, rng, offsets)


def blind_reverse(widths, oracle, queried, N, rng, offsets):
    """Layers from last to first."""
    D = len(widths)
    return blind_schedule(list(range(D - 1, -1, -1)))(widths, oracle, queried, N, rng, offsets)


# ---------------------------------------------------------------------------
# Schedule generators
# ---------------------------------------------------------------------------

def all_single_layer_schedules(D):
    """One schedule per starting layer."""
    return {f"blind[{L}]": blind_schedule([L]) for L in range(D)}


def standard_schedules(D, widths):
    """Standard set of schedules to compare."""
    s = {
        "fwd-merge": fwd_merge,
        "blind-last": blind_last,
        "blind-narrow": blind_narrow_first,
        "blind-rev": blind_reverse,
    }
    # Single layers
    for L in range(D):
        s[f"blind[{L}]"] = blind_schedule([L])
    # Last + first
    s[f"blind[{D-1},0]"] = blind_schedule([D - 1, 0])
    # Last two + first
    if D >= 3:
        s[f"blind[{D-1},{D-2},0]"] = blind_schedule([D - 1, D - 2, 0])
    # Reverse then forward
    s["blind-rev+fwd"] = blind_schedule(list(range(D - 1, -1, -1)) + list(range(D)))
    return s


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_experiment(widths, schedules, N=8, num_samples=300, timeout_per=60):
    """Run all schedules on a cone, return sorted results."""
    k = cone_k(widths)
    results = []
    for name, sfn in schedules.items():
        t0 = time.monotonic()
        cumul, _, trials = evaluate(widths, sfn, N=N, num_samples=num_samples, timeout=timeout_per)
        elapsed = time.monotonic() - t0
        ratio = cumul / k
        results.append({
            "name": name,
            "ratio": round(ratio, 5),
            "cumul": round(cumul, 4),
            "k": k,
            "trials": trials,
            "time": round(elapsed, 1),
            "holds": ratio <= 1.0 + 1e-6,
        })
    results.sort(key=lambda r: r["ratio"])
    return results


def print_results(widths, results, top_n=10):
    """Print results table."""
    k = results[0]["k"] if results else cone_k(widths)
    D = len(widths)
    w_str = str(widths) if len(str(widths)) < 40 else f"[{widths[0]}..{widths[-1]}] D={D}"
    print(f"\n{'='*65}")
    print(f"  {w_str}  k={k}  total_q={sum(widths)}")
    print(f"{'='*65}")
    for r in results[:top_n]:
        s = '✓' if r["holds"] else '✗'
        print(f"  {r['ratio']:.5f}  {s}  {r['name']:25s}  n={r['trials']:3d}  {r['time']:.1f}s")
    if len(results) > top_n:
        print(f"  ... ({len(results) - top_n} more)")
        worst = results[-1]
        s = '✓' if worst["holds"] else '✗'
        print(f"  {worst['ratio']:.5f}  {s}  {worst['name']:25s}  (worst)")


def funnel(D):
    """Funnel widths [2D, 2D-1, ..., D+1]."""
    return list(range(2 * D, D, -1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cones = [
        funnel(5), funnel(8), funnel(10), funnel(15),
        funnel(20), funnel(30), funnel(40), funnel(50),
        [3, 3, 3, 3], [5, 5, 5], [2] * 10, [2] * 20,
        [16, 8, 4, 2], [2, 4, 8, 16],
        [100, 50, 25, 10],
    ]

    # Allow CLI override
    if "--cone" in sys.argv:
        idx = sys.argv.index("--cone")
        cone_str = sys.argv[idx + 1]
        cones = [list(map(int, cone_str.split(",")))]

    for widths in cones:
        D = len(widths)
        k = cone_k(widths)
        tq = sum(widths)

        # Adjust samples based on size
        if tq < 50:
            n = 500
        elif tq < 200:
            n = 300
        elif tq < 500:
            n = 150
        else:
            n = 50

        schedules = standard_schedules(D, widths)
        results = run_experiment(widths, schedules, num_samples=n, timeout_per=60)
        print_results(widths, results)
        best = results[0]
        print(f"  BEST: {best['name']}  ratio={best['ratio']}")
