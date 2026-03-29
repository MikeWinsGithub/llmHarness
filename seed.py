"""Seed the harness with baseline strategies and instances."""

import storage
from codegen import extract_tunable_params

PROBLEM_ID = "oracle_averaging"

# --- Baseline strategy: simple random sampling ---
SIMPLE_AVG = '''\
def estimate(runner, N, k, budget):
    """Sample random inputs, run f on each, average outputs."""
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

# --- Instance: single query, return oracle value ---
SINGLE_QUERY = '''\
TUNABLE_PARAMS = {
    "num_values": {"default": 256, "min": 2, "max": 65536, "type": "int",
                   "description": "Number of distinct query locations (x is hashed into [0, num_values))"},
}
params = {name: spec["default"] for name, spec in TUNABLE_PARAMS.items()}

def oracle_algorithm(x, query):
    """Encode full x as an integer, query oracle at that location mod num_values."""
    val = 0
    for i in range(len(x)):
        val = val * 2 + int(x[i])
    q = val % params["num_values"]
    return float(query(q))
'''

# --- Instance: parity of k queries ---
PARITY_K = '''\
TUNABLE_PARAMS = {
    "num_queries": {"default": 4, "min": 1, "max": 8, "type": "int",
                    "description": "Number of oracle queries (must be <= eval k)"},
    "M": {"default": 256, "min": 2, "max": 65536, "type": "int",
          "description": "Number of possible query locations"},
}
params = {name: spec["default"] for name, spec in TUNABLE_PARAMS.items()}

def oracle_algorithm(x, query):
    """Make k queries at pseudo-random locations determined by x, return product."""
    import numpy as np
    k = params["num_queries"]
    M = params["M"]
    x_int = 0
    for i in range(len(x)):
        x_int = x_int * 2 + int(x[i])
    result = 1
    for j in range(k):
        loc = int(np.random.default_rng([x_int, j, 42]).integers(0, M))
        result *= query(loc)
    return float(result)
'''

# --- Instance: adaptive chain ---
ADAPTIVE_CHAIN = '''\
TUNABLE_PARAMS = {
    "chain_length": {"default": 4, "min": 1, "max": 8, "type": "int",
                     "description": "Total oracle queries in chain (must be <= eval k)"},
    "modulus": {"default": 256, "min": 16, "max": 1024, "type": "int",
                "description": "Modular arithmetic base for locations"},
    "step_multiplier": {"default": 17, "min": 1, "max": 100, "type": "int",
                        "description": "Step multiplier for location updates"},
}
params = {name: spec["default"] for name, spec in TUNABLE_PARAMS.items()}

def oracle_algorithm(x, query):
    """Adaptive: each query location depends on the previous answer."""
    loc = 0
    for i in range(len(x)):
        loc = (loc * 2 + int(x[i])) % params["modulus"]
    val = query(loc)
    for step in range(params["chain_length"] - 1):
        loc = (loc + val + step * params["step_multiplier"]) % params["modulus"]
        val = query(loc)
    return float(val)
'''


def _exists(role, name):
    """Check if an entry with this role+name already exists."""
    for e in storage.list_entries(PROBLEM_ID, role):
        if e["name"] == name:
            return True
    return False


def _seed_entry(role, name, description, code):
    """Create and save an entry if it doesn't already exist."""
    if _exists(role, name):
        print(f"  {name} already exists, skipping")
        return None
    tp = extract_tunable_params(code)
    entry = storage.make_entry(PROBLEM_ID, role, name, description, code, tp)
    storage.save_entry(entry)
    print(f"  {name} -> {entry.id}")
    return entry


def seed():
    print("Seeding (skipping entries that already exist):")

    _seed_entry("strategy", "Simple averaging",
        "Naive Monte Carlo estimator. At stage d with budget d*k, it draws d inputs x uniformly "
        "from {0,1}^N, runs the oracle algorithm f^O(x) on each (costing k queries per run), and "
        "returns the sample mean. This gives unbiased estimates with variance ~1/d, so the per-stage "
        "Bayes risk E_d decays as O(1/d). Because the harmonic series diverges, the cumulative risk "
        "sum(E_d) diverges — this strategy is expected to fail the conjecture for any nontrivial instance.",
        SIMPLE_AVG)

    _seed_entry("instance", "Single query",
        "Encodes the entire input vector x as a single integer (treating it as a binary number) "
        "and queries the oracle at that integer modulo num_values. With N=8, x can take 256 distinct "
        "values; if num_values >= 256 then every input maps to a unique query location and F(O) is "
        "the mean of 256 independent oracle values. Reducing num_values causes collisions — multiple "
        "inputs map to the same query — which makes F(O) depend on fewer oracle bits and thus easier "
        "to estimate. This is the simplest non-trivial instance: depth 1, no adaptivity, and the "
        "difficulty is controlled entirely by how many distinct oracle locations are reachable.",
        SINGLE_QUERY)

    _seed_entry("instance", "Parity of k queries",
        "Makes num_queries oracle queries and returns their product (parity). For each query j, the "
        "query location is determined by hashing the full input x together with the query index j "
        "into the range [0, M), using a seeded RNG as a hash function. This means each query location "
        "is a pseudo-random function of x, and different queries hit essentially independent locations. "
        "The output is +1 if an even number of oracle answers are -1, and -1 otherwise. Because the "
        "output is a product of k independent oracle values, knowing any strict subset gives zero "
        "information about the product — a strategy must learn all k queried oracle values to predict "
        "the output. The parameter M controls the size of the oracle location space.",
        PARITY_K)

    _seed_entry("instance", "Adaptive chain",
        "An adaptive oracle algorithm where each query location depends on the previous oracle answer. "
        "It first hashes the input x into a starting location via loc = sum(x[i]*2^i) mod modulus, then "
        "makes chain_length sequential queries. After each query, the next location is computed as "
        "(loc + oracle_answer + step*step_multiplier) mod modulus, so the oracle's own answers determine "
        "the path through query space. This creates complex dependencies between oracle values, making "
        "it difficult for strategies that assume independence. The step_multiplier controls how far apart "
        "successive queries land in the location space.",
        ADAPTIVE_CHAIN)

    print("Done.")


if __name__ == "__main__":
    seed()
