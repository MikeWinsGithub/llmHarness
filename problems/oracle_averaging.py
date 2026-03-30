"""Oracle-Averaging Conjecture problem.

Conjecture: For any deterministic depth-k oracle algorithm f,
if E_d is the Bayes risk of the best estimator using dk queries,
then sum_{d>=1} E_d <= 1.

Strategies = estimators that try to estimate F(O) = 2^{-N} sum_x f^O(x)
Instances  = deterministic oracle algorithms f
"""

import hashlib
import numpy as np
from .base import Problem, EvalResult


# ---------------------------------------------------------------------------
# Oracle environment
# ---------------------------------------------------------------------------

class Oracle:
    """Random oracle O: query_id -> {-1, +1}, lazily sampled.

    Values are determined by (seed, query_id) so the mapping is independent
    of the order in which queries are encountered.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng or np.random.default_rng()
        self._seed: int = int(self._rng.integers(0, 2**62))
        self._seed_bytes: bytes = self._seed.to_bytes(8, "little")
        self._values: dict[int, int] = {}

    def query(self, q: int) -> int:
        if q not in self._values:
            # Fast deterministic bit from (seed, q) using MD5 — order-independent
            h = hashlib.md5(self._seed_bytes + int(q).to_bytes(8, "little", signed=True)).digest()
            self._values[q] = 1 if h[0] & 1 else -1
        return self._values[q]

    def reset(self):
        self._values.clear()

    @property
    def revealed(self) -> dict[int, int]:
        return dict(self._values)


class OracleAlgorithmRunner:
    """Runs an instance (oracle algorithm f) on inputs, tracking query usage."""

    def __init__(self, oracle: Oracle, instance_fn, N: int, k: int):
        self.oracle = oracle
        self.instance_fn = instance_fn  # f(x, oracle_query) -> float
        self.N = N
        self.k = k
        self.total_queries_used = 0

    def run_on_input(self, x: np.ndarray) -> float:
        """Run f^O(x). Returns output in [-1,1]. Tracks queries."""
        queries_this_run = []

        def tracked_query(q: int) -> int:
            queries_this_run.append(q)
            if len(queries_this_run) > self.k:
                raise RuntimeError(f"Instance exceeded depth limit k={self.k}")
            return self.oracle.query(q)

        result = self.instance_fn(x, tracked_query)
        result = float(np.clip(result, -1.0, 1.0))
        self.total_queries_used += len(queries_this_run)
        return result

    def compute_F_exact(self, oracle: "Oracle | None" = None) -> float:
        """Compute F(O) = 2^{-N} sum_x f^O(x) by brute force. Only for small N.

        Args:
            oracle: Oracle to query. If None, uses self.oracle. Pass a separate
                    oracle to avoid revealing values in the runner's own oracle.
        """
        o = oracle or self.oracle
        total = 0.0
        for i in range(2 ** self.N):
            x = np.array([(i >> bit) & 1 for bit in range(self.N)], dtype=np.int8)
            total += self.instance_fn(x, o.query)
        return total / (2 ** self.N)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    strategy_code: str,
    instance_code: str,
    N: int = 8,
    k: int = 4,
    max_d: int = 20,
    num_oracle_samples: int = 50,
    seed: int = 42,
    strategy_params: dict | None = None,
    instance_params: dict | None = None,
    timeout: int = 600,
    progress_callback=None,
) -> EvalResult:
    """Evaluate a strategy on an instance.

    strategy_code must define: estimate(runner, N, k, budget) -> float
        runner: OracleAlgorithmRunner (call runner.run_on_input(x))
        N: input dimension
        k: depth of oracle algorithm
        budget: total allowed oracle queries (= d*k)
        Returns: estimate of F(O)

    instance_code must define: oracle_algorithm(x, query) -> float
        x: np.ndarray of shape (N,) with entries in {0,1}
        query: callable, query(q) -> {-1,+1}
        Returns: value in [-1, 1]
        Must make at most k calls to query.
    """
    # Compile code
    strategy_ns = {"np": np}
    instance_ns = {"np": np}

    try:
        exec(instance_code, instance_ns)
    except Exception as e:
        return EvalResult(
            metrics={}, summary=f"Instance code failed to compile: {e}",
            conjecture_holds=None
        )

    try:
        exec(strategy_code, strategy_ns)
    except Exception as e:
        return EvalResult(
            metrics={}, summary=f"Strategy code failed to compile: {e}",
            conjecture_holds=None
        )

    # Inject tunable parameter overrides into exec namespaces
    if instance_params and "params" in instance_ns:
        instance_ns["params"].update(instance_params)
    if strategy_params and "params" in strategy_ns:
        strategy_ns["params"].update(strategy_params)

    instance_fn = instance_ns.get("oracle_algorithm")
    estimate_fn = strategy_ns.get("estimate")

    if instance_fn is None:
        return EvalResult(metrics={}, summary="Instance code must define oracle_algorithm(x, query)", conjecture_holds=None)
    if estimate_fn is None:
        return EvalResult(metrics={}, summary="Strategy code must define estimate(runner, N, k, budget)", conjecture_holds=None)

    # Auto-detect how many queries the instance actually needs per input,
    # and bump k if necessary so strategies get the right budget.
    try:
        _det_oracle = Oracle(rng=np.random.default_rng(0))
        _det_x = np.zeros(N, dtype=np.int8)
        instance_fn(_det_x, _det_oracle.query)
        detected_k = len(_det_oracle._values)
        if detected_k > k:
            k = detected_k
    except Exception:
        pass  # fall back to caller-supplied k

    # Auto-scale for expensive (high-k) instances to avoid timeouts.
    # Each trial runs max_d stages, each with budget=d*k queries. High k makes this very slow.
    if k > 20:
        # Scale down trials proportional to cost
        scaled_trials = max(5, num_oracle_samples * 20 // k)
        if scaled_trials < num_oracle_samples:
            num_oracle_samples = scaled_trials
        # Also cap max_d: diminishing returns for high-k on late stages
        scaled_d = max(5, max_d * 20 // k)
        if scaled_d < max_d:
            max_d = scaled_d

    import time as _time

    rng = np.random.default_rng(seed)
    Ed_accum = np.zeros(max_d)
    trials_completed = 0
    deadline = _time.monotonic() + timeout

    for trial in range(num_oracle_samples):
        if _time.monotonic() >= deadline:
            break

        oracle_seed = rng.integers(0, 2**62)
        # Use a separate oracle to compute F_true without revealing values
        truth_oracle = Oracle(rng=np.random.default_rng(oracle_seed))
        runner = OracleAlgorithmRunner(truth_oracle, instance_fn, N, k)
        F_true = runner.compute_F_exact()

        for d_idx in range(max_d):
            d = d_idx + 1
            budget = d * k
            # Fresh oracle with same randomness for each stage
            stage_oracle = Oracle(rng=np.random.default_rng(oracle_seed))
            stage_runner = OracleAlgorithmRunner(stage_oracle, instance_fn, N, k)

            try:
                mu_hat = estimate_fn(stage_runner, N, k, budget)
            except Exception as e:
                mu_hat = 0.0  # fallback on error

            Ed_accum[d_idx] += (mu_hat - F_true) ** 2

            if _time.monotonic() >= deadline:
                break

        trials_completed += 1
        if progress_callback:
            progress_callback(trials_completed, num_oracle_samples)

    if trials_completed == 0:
        return EvalResult(
            metrics={}, summary=f"Timed out after {timeout}s (0 trials completed, k={k})",
            conjecture_holds=None
        )

    Ed = Ed_accum / trials_completed
    cumulative_risk = float(np.sum(Ed))

    metrics = {
        "cumulative_risk": round(cumulative_risk, 6),
        "max_Ed": round(float(np.max(Ed)), 6),
        "E1": round(float(Ed[0]), 6),
    }

    holds = cumulative_risk <= 1.0 + 1e-6  # small tolerance
    partial = f" (partial: {trials_completed}/{num_oracle_samples} trials)" if trials_completed < num_oracle_samples else ""
    summary = f"∑E_d = {cumulative_risk:.4f} ({'≤ 1 ✓' if holds else '> 1 ✗'}){partial}"

    return EvalResult(
        metrics=metrics,
        summary=summary,
        details={"Ed": Ed.tolist(), "N": N, "k": k, "num_oracle_samples": num_oracle_samples,
                 "trials_completed": trials_completed},
        conjecture_holds=holds,
    )


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class OracleAveragingProblem(Problem):

    @property
    def id(self) -> str:
        return "oracle_averaging"

    @property
    def name(self) -> str:
        return "Oracle-Averaging Conjecture"

    @property
    def description(self) -> str:
        return (
            "Let O be a random ±1 oracle and f a deterministic depth-k oracle algorithm. "
            "F(O) = 2^{-N} Σ_x f^O(x). E_d = min Bayes risk with budget dk. "
            "Conjecture: Σ_{d≥1} E_d ≤ 1."
        )

    def strategy_spec(self) -> str:
        return '''\
Write a Python function with this exact signature:

def estimate(runner, N, k, budget):
    """Estimate F(O) using at most `budget` oracle queries.

    Args:
        runner: an OracleAlgorithmRunner. Use runner.run_on_input(x) to run
                the oracle algorithm f on input x (a numpy array of 0s and 1s,
                length N). Each call uses at most k oracle queries.
                runner.oracle.query(q) makes a direct oracle query (costs 1 query).
        N: int, dimension of the input space {0,1}^N
        k: int, max queries per input (depth of oracle algorithm)
        budget: int, total oracle queries allowed (= d * k for stage d)

    Returns:
        float, estimate of F(O) = 2^{-N} Σ_x f^O(x)
    """

The function is called once per stage d, with budget = d*k.
A simple baseline: sample d random inputs, run f on each, average the outputs.
Better strategies can reuse information across queries, use posterior means, etc.

numpy is available as np.
'''

    def instance_spec(self) -> str:
        return '''\
Write a Python function with this exact signature:

def oracle_algorithm(x, query):
    """A deterministic oracle algorithm.

    Args:
        x: np.ndarray of shape (N,) with entries in {0, 1}
        query: callable, query(q) -> int in {-1, +1}. This is the oracle.
               You may call query at most k times (k will be set externally).

    Returns:
        float in [-1, 1]
    """

The function should be deterministic given x and the oracle answers.
It can adaptively choose which oracle locations to query based on previous answers.
The query locations can depend on x and on previous oracle answers.

Examples of interesting instances:
- Parity of k oracle values at locations depending on x
- Layered/tree-structured computations
- Adaptive decision trees

numpy is available as np.
'''

    def evaluate(self, strategy_code: str, instance_code: str, params: dict | None = None,
                 strategy_params: dict | None = None, instance_params: dict | None = None,
                 progress_callback=None) -> EvalResult:
        p = {**self.default_params(), **(params or {})}
        return run_evaluation(strategy_code, instance_code, **p,
                              strategy_params=strategy_params, instance_params=instance_params,
                              progress_callback=progress_callback)

    def default_params(self) -> dict:
        return {"N": 8, "k": 4, "max_d": 20, "num_oracle_samples": 50, "seed": 42}
