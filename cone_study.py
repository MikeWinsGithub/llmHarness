"""Generalized Cone study module.

A generalized cone has widths [W_0, W_1, ..., W_{D-1}].
- Input x maps to starting vertex v_0 = x_int % W_0.
- At layer i, vertex v_i transitions to layer i+1 using ceil(log2(W_{i+1}))
  oracle bits to pick a destination in [W_{i+1}].
- At the terminal layer (D-1), a single oracle query gives the ±1 output.

The key property: given known oracle values, the exact Bayesian estimate
of F(O) can be computed efficiently via dynamic programming on the DAG.
"""

import hashlib
import math
import time
import numpy as np


# ---------------------------------------------------------------------------
# Oracle (same as in oracle_averaging.py)
# ---------------------------------------------------------------------------

class Oracle:
    """Random oracle O: query_id -> {-1, +1}, lazily sampled.
    Deterministic given the seed."""
    def __init__(self, seed):
        self._seed = seed
        self._values = {}

    def query(self, q):
        q = int(q)
        if q not in self._values:
            h = hashlib.md5(f"{self._seed}:{q}".encode()).digest()
            self._values[q] = 1 if (h[0] & 1) else -1
        return self._values[q]


# ---------------------------------------------------------------------------
# Cone geometry helpers
# ---------------------------------------------------------------------------

def cone_k(widths):
    """Compute k (queries per input) for a generalized cone."""
    k = 0
    for i in range(len(widths) - 1):
        k += max(1, math.ceil(math.log2(widths[i + 1]))) if widths[i + 1] > 1 else 0
    k += 1  # terminal sign query
    return k


def cone_total_queries(widths):
    """Total unique oracle query locations in the cone."""
    total = 0
    for i in range(len(widths) - 1):
        bits = max(1, math.ceil(math.log2(widths[i + 1]))) if widths[i + 1] > 1 else 0
        total += widths[i] * bits
    total += widths[-1]  # terminal sign queries
    return total


def _layer_offsets(widths):
    """Compute the query offset for each layer transition and the terminal."""
    offsets = []
    offset = 0
    for i in range(len(widths) - 1):
        bits = max(1, math.ceil(math.log2(widths[i + 1]))) if widths[i + 1] > 1 else 0
        offsets.append((offset, bits))
        offset += widths[i] * bits
    offsets.append((offset, 0))  # terminal layer offset
    return offsets


# ---------------------------------------------------------------------------
# Cone instance (oracle algorithm)
# ---------------------------------------------------------------------------

def make_cone_instance(widths):
    """Create an oracle_algorithm function for a generalized cone."""
    offsets = _layer_offsets(widths)

    def oracle_algorithm(x, query):
        # Map input to starting vertex
        x_int = 0
        for i in range(len(x)):
            x_int = x_int * 2 + int(x[i])
        v = x_int % widths[0]

        # Traverse layers
        for i in range(len(widths) - 1):
            layer_offset, bits = offsets[i]
            if bits == 0:
                v = 0
                continue
            next_v = 0
            for b in range(bits):
                qid = layer_offset + v * bits + b
                next_v = next_v * 2 + ((query(qid) + 1) // 2)
            v = next_v % widths[i + 1]

        # Terminal sign query
        terminal_offset = offsets[-1][0]
        return float(query(terminal_offset + v))

    return oracle_algorithm


# ---------------------------------------------------------------------------
# Merge-aware query allocation
# ---------------------------------------------------------------------------

def _allocate_queries(widths, oracle, N, budget, rng):
    """Explore random inputs using cached queries (merge-aware).
    Returns dict of {query_id: value} for all revealed oracle locations."""
    offsets = _layer_offsets(widths)
    known = {}
    queries_used = 0

    for idx in rng.permutation(2 ** N):
        v = idx % widths[0]

        for i in range(len(widths) - 1):
            layer_offset, bits = offsets[i]
            if bits == 0:
                v = 0
                continue
            next_v = 0
            for b in range(bits):
                qid = layer_offset + v * bits + b
                if qid not in known:
                    if queries_used >= budget:
                        return known
                    known[qid] = oracle.query(qid)
                    queries_used += 1
                next_v = next_v * 2 + ((known[qid] + 1) // 2)
            v = next_v % widths[i + 1]

        # Terminal sign query
        terminal_offset = offsets[-1][0]
        qid = terminal_offset + v
        if qid not in known:
            if queries_used >= budget:
                return known
            known[qid] = oracle.query(qid)
            queries_used += 1

    return known


# ---------------------------------------------------------------------------
# Exact Bayesian estimate via DP
# ---------------------------------------------------------------------------

def _exact_bayesian_estimate(widths, known, N):
    """Compute E[F(O) | known] exactly using dynamic programming.

    Works backwards from terminal layer:
    - val[v] at terminal = known_sign if known, else 0
    - val[v] at layer i = average over possible next vertices
      (averaging over unknown oracle bits)
    """
    D = len(widths)
    offsets = _layer_offsets(widths)

    # Terminal layer values
    terminal_offset = offsets[-1][0]
    val = {}
    for v in range(widths[-1]):
        qid = terminal_offset + v
        val[v] = float(known[qid]) if qid in known else 0.0

    # Work backwards through layers
    for layer in range(D - 2, -1, -1):
        layer_offset, bits = offsets[layer]
        new_val = {}

        for v in range(widths[layer]):
            if bits == 0:
                new_val[v] = val.get(0, 0.0)
                continue

            # Determine which bits are known/unknown for this vertex
            bit_info = []
            for b in range(bits):
                qid = layer_offset + v * bits + b
                if qid in known:
                    bit_info.append((known[qid] + 1) // 2)  # 0 or 1
                else:
                    bit_info.append(None)  # unknown

            # Enumerate all possible next vertices
            num_unknown = sum(1 for b in bit_info if b is None)
            total = 0.0
            for combo in range(2 ** num_unknown):
                # Fill in unknown bits
                bits_val = list(bit_info)
                ui = 0
                for j in range(len(bits_val)):
                    if bits_val[j] is None:
                        bits_val[j] = (combo >> ui) & 1
                        ui += 1
                # Compute next vertex
                next_v = 0
                for b in bits_val:
                    next_v = next_v * 2 + b
                next_v = next_v % widths[layer + 1]
                total += val.get(next_v, 0.0)

            new_val[v] = total / (2 ** num_unknown)

        val = new_val

    # Average over starting vertices (weighted by input count)
    W_0 = widths[0]
    # Each vertex v_0 has floor(2^N / W_0) or ceil(2^N / W_0) inputs
    total_F = 0.0
    for v_0 in range(W_0):
        # Count inputs mapping to this vertex
        count = 0
        for x_int in range(2 ** N):
            if x_int % W_0 == v_0:
                count += 1
        total_F += val.get(v_0, 0.0) * count

    return total_F / (2 ** N)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_cone(
    widths,
    N=8,
    max_d=20,
    num_oracle_samples=50,
    seed=42,
    timeout=600,
    progress_callback=None,
):
    """Evaluate the exact Bayesian estimator on a generalized cone.

    Returns dict with Ed curve, cumulative risk, and metadata.
    """
    k = cone_k(widths)
    total_q = cone_total_queries(widths)

    rng = np.random.default_rng(seed)
    Ed_accum = np.zeros(max_d)
    trials_completed = 0
    deadline = time.monotonic() + timeout

    instance_fn = make_cone_instance(widths)

    for trial in range(num_oracle_samples):
        if time.monotonic() >= deadline:
            break

        oracle_seed = int(rng.integers(0, 2 ** 62))

        # Compute F_true exactly
        truth_oracle = Oracle(seed=oracle_seed)
        F_true = 0.0
        for x_int in range(2 ** N):
            x = np.array([(x_int >> bit) & 1 for bit in range(N)], dtype=np.int8)
            F_true += instance_fn(x, truth_oracle.query)
        F_true /= 2 ** N

        for d_idx in range(max_d):
            if time.monotonic() >= deadline:
                break
            d = d_idx + 1
            budget = d * k

            # Fresh oracle with same seed
            stage_oracle = Oracle(seed=oracle_seed)
            stage_rng = np.random.default_rng(rng.integers(0, 2 ** 62))

            # Allocate queries (merge-aware)
            known = _allocate_queries(widths, stage_oracle, N, budget, stage_rng)

            # Exact Bayesian estimate
            estimate = _exact_bayesian_estimate(widths, known, N)

            Ed_accum[d_idx] += (estimate - F_true) ** 2

        trials_completed += 1
        if progress_callback:
            progress_callback(trials_completed, num_oracle_samples)

    if trials_completed == 0:
        return {
            "widths": widths,
            "k": k,
            "total_queries": total_q,
            "error": f"Timed out (0 trials completed)",
        }

    Ed = (Ed_accum / trials_completed).tolist()
    cumulative_risk = sum(Ed)

    return {
        "widths": widths,
        "k": k,
        "total_queries": total_q,
        "N": N,
        "max_d": max_d,
        "num_oracle_samples": num_oracle_samples,
        "trials_completed": trials_completed,
        "Ed": Ed,
        "cumulative_risk": round(cumulative_risk, 6),
        "conjecture_holds": cumulative_risk <= 1.0 + 1e-6,
    }


# ---------------------------------------------------------------------------
# Preset width sequences
# ---------------------------------------------------------------------------

PRESETS = [
    {
        "name": "Constant W=4, D=5",
        "widths": [4, 4, 4, 4, 4],
        "description": "Standard layered graph with constant width. k=9.",
    },
    {
        "name": "Narrowing [16,8,4,2]",
        "widths": [16, 8, 4, 2],
        "description": "Classic cone: exponential narrowing, paths merge aggressively. k=7.",
    },
    {
        "name": "Widening [2,4,8,16]",
        "widths": [2, 4, 8, 16],
        "description": "Inverse cone: paths diverge, harder to learn shared structure. k=10.",
    },
    {
        "name": "Hourglass [16,4,16]",
        "widths": [16, 4, 16],
        "description": "Narrows then widens: bottleneck forces merging then re-expands. k=7.",
    },
    {
        "name": "Diamond [4,16,4]",
        "widths": [4, 16, 4],
        "description": "Widens then narrows: expands then merges. k=7.",
    },
    {
        "name": "Deep narrow [2,2,2,2,2,2,2,2]",
        "widths": [2, 2, 2, 2, 2, 2, 2, 2],
        "description": "Many layers at minimum width. k=8. Long adaptive chains.",
    },
    {
        "name": "Sharp funnel [64,2]",
        "widths": [64, 2],
        "description": "Wide input layer collapsing to 2 terminals. k=2. Extreme merging.",
    },
    {
        "name": "Flat wide [16,16,16]",
        "widths": [16, 16, 16],
        "description": "Wide constant-width graph. k=9. Many unique query locations.",
    },
    {
        "name": "Staircase down [32,16,8,4,2]",
        "widths": [32, 16, 8, 4, 2],
        "description": "Gradual narrowing over 5 layers. k=9.",
    },
    {
        "name": "Staircase up [2,4,8,16,32]",
        "widths": [2, 4, 8, 16, 32],
        "description": "Gradual widening over 5 layers. k=14.",
    },
]
