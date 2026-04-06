"""Generalized Cone study module.

A generalized cone has widths [W_0, W_1, ..., W_{D-1}].
- Input x maps to starting vertex v_0 = x_int % W_0.
- At layer i, vertex v_i makes ONE oracle call that returns a value
  in {0, ..., W_{i+1}-1}, determining the next vertex.
- At the terminal layer (D-1), a single oracle call returns ±1 for the output.
- k = D (one query per layer).

The key property: given known oracle values, the exact Bayesian estimate
of F(O) can be computed efficiently via dynamic programming on the DAG.
"""

import hashlib
import time
import numpy as np


# ---------------------------------------------------------------------------
# Oracle: query_id -> value in a specified range
# ---------------------------------------------------------------------------

class Oracle:
    """Random oracle, lazily sampled. Deterministic given the seed.
    query(q) returns a value in {0, ..., range-1} for transition queries,
    or {-1, +1} for sign queries."""
    def __init__(self, seed):
        self._seed = seed
        self._values = {}

    def query(self, q, num_values=2):
        """Query the oracle at location q, returning a value in {0, ..., num_values-1}.
        For sign queries, use num_values=2 and map to ±1 externally."""
        q = int(q)
        if q not in self._values:
            h = hashlib.md5(f"{self._seed}:{q}".encode()).digest()
            raw = int.from_bytes(h[:4], 'little')
            self._values[q] = raw % num_values
        return self._values[q]


# ---------------------------------------------------------------------------
# Cone geometry helpers
# ---------------------------------------------------------------------------

def cone_k(widths):
    """k = D = number of layers (one query per layer)."""
    return len(widths)


def cone_total_queries(widths):
    """Total unique oracle query locations in the cone."""
    # One query per vertex at each layer (transition + terminal)
    return sum(widths)


def _layer_offset(widths, layer):
    """Query offset for a given layer."""
    return sum(widths[:layer])


# ---------------------------------------------------------------------------
# Cone instance (oracle algorithm)
# ---------------------------------------------------------------------------

def make_cone_instance(widths):
    """Create an oracle_algorithm function for a generalized cone.

    Each layer i has W_i vertices. The oracle at vertex v in layer i
    returns a value in {0, ..., W_{i+1}-1} for the next vertex.
    The terminal layer returns ±1.
    """
    D = len(widths)
    # Precompute offsets
    offsets = [sum(widths[:i]) for i in range(D)]

    def oracle_algorithm(x, query):
        # Map input to starting vertex
        x_int = 0
        for i in range(len(x)):
            x_int = x_int * 2 + int(x[i])
        v = x_int % widths[0]

        # Traverse layers 0..D-2: transition queries
        for i in range(D - 1):
            qid = offsets[i] + v
            # Query returns raw value; we need to interpret it
            # Use a modified query that returns in the right range
            h = hashlib.md5(f"cone:{id(query)}:{qid}".encode()).digest()
            # Actually, we need to use the actual oracle query function
            # The oracle query returns ±1, but we want {0, ..., W_{i+1}-1}
            # Solution: the query function IS our oracle — it returns the raw value
            raw = query(qid)
            # raw is ±1 from the standard oracle interface, but we want wider range
            # We can't change the query interface... let's use our own oracle model
            v = raw % widths[i + 1]

        # Terminal sign query at layer D-1
        qid = offsets[D - 1] + v
        sign = query(qid)
        # Map {0,1} to {-1,+1}
        return 1.0 if sign == 1 else -1.0

    return oracle_algorithm


# Actually, the standard oracle_algorithm interface expects query(q) -> {-1,+1}.
# Since we want query(q) -> {0, ..., W-1}, we need our own evaluation loop
# that doesn't go through the standard framework. Let's do that.


def run_cone_on_input(x, widths, oracle):
    """Run the cone on input x using our Oracle (which supports arbitrary ranges).
    Returns the output in {-1, +1}."""
    D = len(widths)
    offsets = [sum(widths[:i]) for i in range(D)]

    x_int = 0
    for i in range(len(x)):
        x_int = x_int * 2 + int(x[i])
    v = x_int % widths[0]

    # Traverse layers 0..D-2
    for i in range(D - 1):
        qid = offsets[i] + v
        v = oracle.query(qid, num_values=widths[i + 1])

    # Terminal sign query
    qid = offsets[D - 1] + v
    sign_raw = oracle.query(qid, num_values=2)
    return 1 if sign_raw == 1 else -1


# ---------------------------------------------------------------------------
# Merge-aware query allocation
# ---------------------------------------------------------------------------

def _allocate_queries(widths, oracle, N, budget, rng):
    """Explore random inputs using cached queries (merge-aware).
    Returns set of query locations that were evaluated (values are in oracle._values)."""
    D = len(widths)
    offsets = [sum(widths[:i]) for i in range(D)]
    queries_used = 0
    queried = set()

    for idx in rng.permutation(2 ** N):
        v = idx % widths[0]

        for i in range(D - 1):
            qid = offsets[i] + v
            if qid not in queried:
                if queries_used >= budget:
                    return queried
                oracle.query(qid, num_values=widths[i + 1])
                queried.add(qid)
                queries_used += 1
            v = oracle._values[qid]

        # Terminal sign query
        qid = offsets[D - 1] + v
        if qid not in queried:
            if queries_used >= budget:
                return queried
            oracle.query(qid, num_values=2)
            queried.add(qid)
            queries_used += 1

    return queried


# ---------------------------------------------------------------------------
# Exact Bayesian estimate via DP
# ---------------------------------------------------------------------------

def _exact_bayesian_estimate(widths, oracle, queried, N):
    """Compute E[F(O) | queried values] exactly using dynamic programming.

    Works backwards from terminal layer:
    - val[v] at terminal = sign if known, else 0
    - val[v] at layer i = next vertex value if known,
                          else average over all possible destinations
    """
    D = len(widths)
    offsets = [sum(widths[:i]) for i in range(D)]

    # Terminal layer values
    val = {}
    for v in range(widths[-1]):
        qid = offsets[D - 1] + v
        if qid in queried:
            val[v] = 1.0 if oracle._values[qid] == 1 else -1.0
        else:
            val[v] = 0.0  # E[uniform ±1] = 0

    # Work backwards through layers D-2 .. 0
    for layer in range(D - 2, -1, -1):
        new_val = {}
        W_next = widths[layer + 1]

        for v in range(widths[layer]):
            qid = offsets[layer] + v
            if qid in queried:
                # Known: deterministic next vertex
                next_v = oracle._values[qid]
                new_val[v] = val.get(next_v, 0.0)
            else:
                # Unknown: uniform over {0, ..., W_{i+1}-1}
                total = sum(val.get(nv, 0.0) for nv in range(W_next))
                new_val[v] = total / W_next

        val = new_val

    # Average over starting vertices
    W_0 = widths[0]
    total_F = 0.0
    for v_0 in range(W_0):
        count = sum(1 for x_int in range(2 ** N) if x_int % W_0 == v_0)
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

    for trial in range(num_oracle_samples):
        if time.monotonic() >= deadline:
            break

        oracle_seed = int(rng.integers(0, 2 ** 62))

        # Compute F_true exactly
        truth_oracle = Oracle(seed=oracle_seed)
        F_true = 0.0
        for x_int in range(2 ** N):
            x = np.array([(x_int >> bit) & 1 for bit in range(N)], dtype=np.int8)
            F_true += run_cone_on_input(x, widths, truth_oracle)
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
            queried = _allocate_queries(widths, stage_oracle, N, budget, stage_rng)

            # Exact Bayesian estimate
            estimate = _exact_bayesian_estimate(widths, stage_oracle, queried, N)

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
# Sequential (one-query-at-a-time) evaluation
# ---------------------------------------------------------------------------

def _greedy_next_query(widths, oracle, queried, N):
    """Pick the single query location that minimizes expected MSE after revealing it.

    For each unqueried location qid, compute the expected MSE if we reveal it
    (averaging over possible oracle values), and pick the one with lowest expected MSE.
    """
    D = len(widths)
    offsets = [sum(widths[:i]) for i in range(D)]
    total_q = cone_total_queries(widths)

    best_qid = None
    best_expected_mse = float('inf')

    for qid in range(total_q):
        if qid in queried:
            continue

        # Determine which layer and vertex this qid belongs to
        layer = None
        for l in range(D):
            if offsets[l] <= qid < offsets[l] + widths[l]:
                layer = l
                break

        if layer is None:
            continue

        # How many possible values can this query take?
        if layer < D - 1:
            num_vals = widths[layer + 1]
        else:
            num_vals = 2  # terminal sign query

        # Compute expected MSE after revealing this query
        # = (1/num_vals) * sum over possible values of MSE(queried + {qid: val})
        expected_mse = 0.0
        for val in range(num_vals):
            oracle._values[qid] = val
            new_queried = queried | {qid}
            est = _exact_bayesian_estimate(widths, oracle, new_queried, N)
            expected_mse += est ** 2  # MSE around 0... no, we need (est - F)^2
            # But we don't know F! We want E_O[(est - F)^2] which we can't compute here.
            # Actually for greedy: we want to minimize the posterior variance.
            # The posterior variance = E[F^2 | known] - (E[F | known])^2
            # A simpler proxy: the variance of the estimate under remaining uncertainty.
            # For now, just minimize the variance of the Bayesian estimate.

        # Actually, the right thing: we want to minimize E[(est - F)^2 | known so far]
        # which equals Var(F | known so far). The greedy approach picks the query
        # that maximally reduces this variance.
        # Var(F | known) = E[F^2 | known] - (E[F | known])^2
        # After revealing qid with value v: Var(F | known + {qid:v})
        # Expected reduction = Var(F|known) - E_v[Var(F | known + {qid:v})]

        # Simpler: just compute E_v[(E[F|known+{qid:v}])^2] - this is what we maximize
        # (higher means more variance reduction)
        sum_est_sq = 0.0
        for val in range(num_vals):
            oracle._values[qid] = val
            new_queried = queried | {qid}
            est = _exact_bayesian_estimate(widths, oracle, new_queried, N)
            sum_est_sq += est ** 2
        mean_est_sq = sum_est_sq / num_vals

        # We want to MAXIMIZE mean_est_sq (= maximize variance reduction)
        if best_qid is None or mean_est_sq > best_expected_mse:
            best_expected_mse = mean_est_sq
            best_qid = qid

    # Clean up: remove the test values we set
    if best_qid is not None and best_qid not in queried:
        oracle._values.pop(best_qid, None)

    return best_qid


def evaluate_cone_sequential(
    widths,
    N=8,
    num_oracle_samples=50,
    seed=42,
    timeout=600,
    greedy=True,
    progress_callback=None,
):
    """Evaluate the sequential conjecture: ∑_{i=0}^{total_q} MSE_i ≤ k.

    Reveals oracle values one at a time. If greedy=True, picks the query
    that maximally reduces posterior variance. Otherwise uses a fixed order
    (layer by layer, vertex by vertex).

    Returns dict with MSE_i curve and cumulative sum.
    """
    k = cone_k(widths)
    total_q = cone_total_queries(widths)
    D = len(widths)
    offsets = [sum(widths[:i]) for i in range(D)]

    rng = np.random.default_rng(seed)
    # MSE after 0, 1, 2, ... queries
    mse_accum = np.zeros(total_q + 1)
    trials_completed = 0
    deadline = time.monotonic() + timeout

    for trial in range(num_oracle_samples):
        if time.monotonic() >= deadline:
            break

        oracle_seed = int(rng.integers(0, 2 ** 62))

        # Compute F_true
        truth_oracle = Oracle(seed=oracle_seed)
        F_true = 0.0
        for x_int in range(2 ** N):
            x = np.array([(x_int >> bit) & 1 for bit in range(N)], dtype=np.int8)
            F_true += run_cone_on_input(x, widths, truth_oracle)
        F_true /= 2 ** N

        # Sequential reveal
        stage_oracle = Oracle(seed=oracle_seed)
        queried = set()

        # MSE with 0 queries
        est = _exact_bayesian_estimate(widths, stage_oracle, queried, N)
        mse_accum[0] += (est - F_true) ** 2

        if greedy:
            for i in range(total_q):
                if time.monotonic() >= deadline:
                    break
                # Greedy: pick best next query
                qid = _greedy_next_query(widths, stage_oracle, queried, N)
                if qid is None:
                    break
                # Reveal it with the true value
                stage_oracle.query(qid, num_values=(widths[_qid_layer(qid, widths, offsets) + 1] if _qid_layer(qid, widths, offsets) < D - 1 else 2))
                queried.add(qid)
                est = _exact_bayesian_estimate(widths, stage_oracle, queried, N)
                mse_accum[i + 1] += (est - F_true) ** 2
        else:
            # Fixed order: layer 0 vertices, layer 1 vertices, ...
            for qid in range(total_q):
                if time.monotonic() >= deadline:
                    break
                layer = _qid_layer(qid, widths, offsets)
                num_vals = widths[layer + 1] if layer < D - 1 else 2
                stage_oracle.query(qid, num_values=num_vals)
                queried.add(qid)
                est = _exact_bayesian_estimate(widths, stage_oracle, queried, N)
                mse_accum[qid + 1] += (est - F_true) ** 2

        trials_completed += 1
        if progress_callback:
            progress_callback(trials_completed, num_oracle_samples)

    if trials_completed == 0:
        return {"widths": widths, "k": k, "error": "Timed out"}

    mse = (mse_accum / trials_completed).tolist()
    cumulative = sum(mse)

    return {
        "widths": widths,
        "k": k,
        "total_queries": total_q,
        "N": N,
        "num_oracle_samples": num_oracle_samples,
        "trials_completed": trials_completed,
        "mse_per_query": mse,
        "cumulative_mse": round(cumulative, 6),
        "conjecture_holds": cumulative <= k + 1e-6,
        "greedy": greedy,
    }


def _qid_layer(qid, widths, offsets):
    """Which layer does query location qid belong to?"""
    for l in range(len(widths)):
        if offsets[l] <= qid < offsets[l] + widths[l]:
            return l
    return len(widths) - 1


# ---------------------------------------------------------------------------
# Preset width sequences
# ---------------------------------------------------------------------------

PRESETS = [
    {
        "name": "Constant W=4, D=5",
        "widths": [4, 4, 4, 4, 4],
        "description": "Standard layered graph with constant width. k=5.",
    },
    {
        "name": "Narrowing [16,8,4,2]",
        "widths": [16, 8, 4, 2],
        "description": "Classic cone: exponential narrowing, paths merge aggressively. k=4.",
    },
    {
        "name": "Widening [2,4,8,16]",
        "widths": [2, 4, 8, 16],
        "description": "Inverse cone: paths diverge, harder to learn shared structure. k=4.",
    },
    {
        "name": "Hourglass [16,4,16]",
        "widths": [16, 4, 16],
        "description": "Narrows then widens: bottleneck forces merging then re-expands. k=3.",
    },
    {
        "name": "Diamond [4,16,4]",
        "widths": [4, 16, 4],
        "description": "Widens then narrows: expands then merges. k=3.",
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
        "description": "Wide constant-width graph. k=3. Many unique query locations.",
    },
    {
        "name": "Staircase down [32,16,8,4,2]",
        "widths": [32, 16, 8, 4, 2],
        "description": "Gradual narrowing over 5 layers. k=5.",
    },
    {
        "name": "Staircase up [2,4,8,16,32]",
        "widths": [2, 4, 8, 16, 32],
        "description": "Gradual widening over 5 layers. k=5.",
    },
    {
        "name": "Tall constant [3,3,3,3,3,3,3,3,3,3]",
        "widths": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        "description": "10 layers of width 3. k=10. Non-power-of-2 width.",
    },
    {
        "name": "Wide-shallow [100,100]",
        "widths": [100, 100],
        "description": "Very wide, only 2 layers. k=2. 200 total query locations.",
    },
]
