"""Quick smoke test of the evaluation engine."""
import sys
sys.path.insert(0, '.')
from problems.oracle_averaging import OracleAveragingProblem

problem = OracleAveragingProblem()

strategy_code = '''\
def estimate(runner, N, k, budget):
    import numpy as np
    d = budget // k
    if d == 0:
        return 0.0
    rng = np.random.default_rng()
    total = 0.0
    for _ in range(d):
        x = rng.integers(0, 2, size=N).astype(np.int8)
        total += runner.run_on_input(x)
    return total / d
'''

instance_code = '''\
def oracle_algorithm(x, query):
    q = int(x[0])
    return float(query(q))
'''

params = {"N": 6, "k": 2, "max_d": 10, "num_oracle_samples": 20, "seed": 42}
result = problem.evaluate(strategy_code, instance_code, params)
print(f"Summary: {result.summary}")
print(f"Metrics: {result.metrics}")
print(f"Ed: {[round(e, 4) for e in result.details['Ed']]}")
print(f"Conjecture holds: {result.conjecture_holds}")
