# Cone Conjecture Experiments — Handoff

## What this is

A standalone experiment for testing a conjecture about sequential oracle estimation on "generalized cones" — layered DAGs where each layer has a width W_i, transitions are determined by oracle queries, and the terminal layer gives a ±1 sign.

## The conjecture

**Sequential conjecture:** For any generalized cone with widths [W_0, ..., W_{D-1}] and any query ordering strategy, the sum of MSE after each oracle query is at most k (the number of queries per input, which equals D for cones):

∑_{i=0}^{total_queries} E[(estimate_i - F)²] ≤ k = D

where estimate_i is the exact Bayesian estimate after i oracle values are revealed.

## The cone model

- Input x ∈ {0,1}^N maps to starting vertex v_0 = x mod W_0
- At layer i, vertex v makes ONE oracle call returning a value in {0, ..., W_{i+1}-1} (the next vertex)
- At the terminal layer (D-1), a single oracle call returns 0 or 1, mapped to ±1
- k = D (one query per layer)
- F(O) = (1/2^N) Σ_x f^O(x), the average output over all inputs

## The Bayesian estimator

Given known oracle values, we compute E[F|known] exactly via backward DP:
- Terminal vertex v: if sign known → ±1, if unknown → 0
- Layer i vertex v: if transition known → val(next_v), if unknown → average over all W_{i+1} destinations
- F_estimate = weighted average of layer-0 vertex values

Key insight: before any terminal sign is revealed, the estimate is exactly 0.

## The experiment

**`run_cones.py`** is the standalone runner. It:
1. Picks random oracle seeds
2. Computes F_true exactly (enumerate all 2^N inputs)
3. Uses "forward sample-merge" to decide query order: trace random inputs layer 0→terminal, skip already-known queries, reveal one query at a time
4. After each query, recomputes the Bayesian estimate via incremental DP (only recomputes layers above the changed query)
5. Records MSE = (estimate - F_true)² at each step
6. Sums all MSE values (including the tail where the strategy has run out of queries but MSE persists)
7. Reports ratio = ∑MSE / k

**Funnel cones [2D, 2D-1, ..., D+1]** are the primary test family.

## Current results

With fwd-merge (the best known strategy):

| D | ratio | ±stderr | samples | verdict |
|---|-------|---------|---------|---------|
| 5 | 0.902 | ±0.003 | 50000 | ✓ clearly holds |
| 10 | 0.986 | ±0.003 | 50000 | ✓ barely holds |
| 15 | 1.016 | ±0.003 | 50000 | ✗ fails by 5.7σ |
| 20 | 1.052 | ±0.004 | 20000 | ✗ fails by 13σ |
| 30 | 1.080 | ±0.005 | 20000 | ✗ |
| 50 | 1.110 | ±0.008 | 7500 | ✗ |
| 80 | 1.145 | ±0.032 | 500 | ✗ |

The ratio plateaus around 1.08-1.11 for D≥25. It does NOT diverge with D.

## Strategies tested

- **fwd-merge** (forward sample-merge): best known. Trace random inputs forward, merge on cached queries.
- **1@L1+fwd**: probe one random vertex from layer 1 to terminal, then fwd-merge. Indistinguishable from fwd-merge at D≥25, slightly worse at D<25.
- **5@L5+fwd**: probe 5 random vertices from layer 5. Always worse than fwd-merge.
- **blind-last/rev/narrow**: reveal all vertices at various layers. All much worse because they waste queries on unreachable vertices.
- **multi-pass, breadth-first, interleaved, popular-first, rare-first**: all worse than fwd-merge.

No strategy tested beats fwd-merge on funnels.

## Code structure

```
run_cones.py          # Standalone runner — just needs numpy + numba
cone_fast.py          # Full experiment harness with multiple strategies
cone_study.py         # Original pure-Python implementation (slower, more strategies)
cone_experiments.py   # Pre-numba experiment harness (superseded by cone_fast.py)
```

## How to run

```bash
pip install numpy numba
python run_cones.py --minutes 60 --workers 10
```

Options:
- `--dmin 5 --dmax 200` — range of funnel depths
- `--dstep 5` — step size
- `--d 50` — single D value
- `--minutes 60` — time budget
- `--workers 10` — number of CPU cores to use

## Performance

| D | ms/trial (4-core cloud) | estimated ms/trial (M3 Pro) |
|---|------------------------|---------------------------|
| 10 | 0.8ms | ~0.3ms |
| 20 | 3ms | ~1ms |
| 50 | 32ms | ~10ms |
| 80 | 180ms | ~60ms |
| 100 | 440ms | ~150ms |

Bottleneck is the DP loop: one DP call per query revealed. Incremental DP only recomputes layers above the changed query (~3x speedup at D=80).

## Open questions

1. **Is the constant-factor violation (~1.1) real, or an artifact of fwd-merge being suboptimal?** No strategy we tried beats fwd-merge, but the optimal (exponentially costly) strategy might.

2. **Does the ratio truly plateau, or slowly grow?** The data suggests plateau, but error bars at large D are wide.

3. **What about non-funnel cones?** Constant-width [W,W,...,W] and widening [2,4,8,...] cones behave differently. Narrowing cones are easier.

4. **Could a fundamentally different query ordering work?** All strategies tested are variations on "trace inputs forward." An adaptive strategy that chooses each query based on what's been learned (not what inputs look like) might do better, but is expensive to compute.

## What to try next

- Run with more samples at D=15 to pin down where the conjecture first fails (between D=10 and D=15)
- Try D=200+ to see if the plateau continues
- Investigate whether the violation comes from specific oracle realizations (heavy tails?)
- Try adaptive (greedy) query ordering on small D to see if the conjecture holds with optimal ordering
- Study the per-query MSE curve shape — where does the MSE spend the most "budget"?
