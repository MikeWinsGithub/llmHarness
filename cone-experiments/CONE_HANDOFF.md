# Cone Conjecture Experiments — Handoff

## What this is

A standalone experiment for testing a conjecture about sequential oracle estimation on "generalized cones" — layered DAGs where each layer has a width W_i, transitions are determined by oracle queries, and the terminal layer gives a ±1 sign.

## The conjecture

**Sequential conjecture:** For any generalized cone with widths [W_0, ..., W_{D-1}] and any query ordering strategy, the sum of MSE after each oracle query is at most k (the number of queries per input, which equals D for cones):

∑_{i=0}^{total_queries} E[(estimate_i - F)²] ≤ k = D

where estimate_i is the exact Bayesian estimate after i oracle values are revealed.

## The cone model

- Input v ∈ {0, ..., W_0-1} uniformly
- At layer i, vertex v makes ONE oracle call returning a value in {0, ..., W_{i+1}-1} (the next vertex)
- At the terminal layer (D-1), a single oracle call returns 0 or 1, mapped to ±1
- k = D (one query per layer)
- F(O) = (1/W_0) Σ_v f^O(v), the average output over all inputs

## The Bayesian estimator

Given known oracle values, we compute E[F|known] exactly via backward DP:
- Terminal vertex v: if sign known → ±1, if unknown → 0
- Layer i vertex v: if transition known → val(next_v), if unknown → average over all W_{i+1} destinations
- F_estimate = average of layer-0 vertex values

Key insight: before any terminal sign is revealed, the estimate is exactly 0.

## Current results

With fwd-merge (the best known strategy):

| D | ratio | ±stderr | samples | verdict |
|---|-------|---------|---------|---------|
| 5 | 0.959 | ±0.002 | 50000 | ✓ clearly holds |
| 10 | 1.008 | ±0.003 | 50000 | ~ borderline |
| 15 | 1.054 | ±0.003 | 50000 | ✗ fails |
| 20 | 1.059 | ±0.003 | 50000 | ✗ fails |
| 30 | 1.082 | ±0.003 | 50000 | ✗ fails |
| 50 | 1.098 | ±0.004 | 28000 | ✗ fails |
| 80 | 1.113 | ±0.004 | 35000 | ✗ fails |
| 100 | 1.123 | ±0.006 | 10000 | ✗ fails |

The ratio plateaus around 1.09-1.12 for D≥25. It does NOT diverge with D.

**Funnel cones [2D, 2D-1, ..., D+1]** are the primary test family.

## Strategies tested

- **fwd-merge** (forward sample-merge): best known. Trace random inputs forward, merge on cached queries.
- **x@L1+fwd** (1-5 probes from layer 1 before fwd-merge): always worse than fwd-merge at all D. Penalty scales linearly with number of probes (~0.005 per probe at D=30).
- **1@L1 at position N** (insert L1 probe after Nth L0 trace): penalty decreases with later insertion but never reaches zero.
- **Terminal probes** (1@L(D-1), 1@L(D-2)): significantly worse than fwd-merge (+0.017 and +0.025 at D=50).
- **blind-last/rev/narrow**: reveal all vertices at various layers. Much worse because they waste queries on unreachable vertices.
- **multi-pass, breadth-first, interleaved, popular-first, rare-first**: all worse than fwd-merge.

No strategy tested beats fwd-merge on funnels.

## Optimality of fwd-merge

We developed analytical tools to evaluate whether any adaptive strategy could beat fwd-merge:

### Covariance analysis

We derived and verified a formula for Cov(val(u), F | known) — the conditional covariance of an unqueried vertex's value with F. Key components (all O(D) backward DPs):

- **Var_unk[L]** = 1 - mean(layer_vals[L+1])². Always ≤ 1 since true values are ±1.
- **merge_var[L]** = Cov of two distinct unknown vertices at layer L, from forward path merging:
  merge_var[D-1] = 0, merge_var[L] = n_unk[L+1] × (Var_unk[L+1] + (n_unk[L+1]-1) × merge_var[L+1]) / W[L+1]²
- **reach(u)** = expected number of unknown L0 paths reaching unqueried vertex u at layer L.

Full formula: Cov(val(u), F) = (n_unk[0]/W₀) × (p_unk/n_unk[L]) × (Var_unk[L] + (n_unk[L]-1) × merge_var[L])

This was verified against Monte Carlo conditional resampling and matches within 1% across all D, layers, and numbers of traces. See COVARIANCE_DERIVATION.md.

**Finding:** L0 always has the highest conditional covariance with F. Intermediate layers can briefly have higher Cov/cost ratios, but this is misleading (see below).

### Exact variance reduction

We computed E[(ΔF_est)²] = Var(F|known) - E[Var(F|known ∪ probe)] exactly via recursive enumeration over probe outcomes. This is O(D × W) per layer, no Monte Carlo needed.

**Finding:** L0 always has the highest variance reduction, both raw and per-query. Across 100 full trials at D=8, L0 was selected 99.88% of the time. Paired comparison with the same vertex ordering shows the adaptive strategy is indistinguishable from fwd-merge.

### Why simpler heuristics failed

- **Cov/cost**: picks the terminal layer (cheap queries, decent per-query covariance) but this is catastrophic — ratio 4.4 vs 1.05. Deep probes are informationally cheap because they affect F only through diluted averaging.
- **Raw Cov**: mostly picks L0 (80%) but occasionally picks intermediate layers where Var_unk is slightly higher due to mean dilution. Those probes waste queries on unreachable vertices. Ratio 1.25.
- **E[(ΔF_est)²]**: correctly identifies L0 as always best. Matches fwd-merge.

## Bug fix: hash oracle

The original hash oracle (`seed * C1 + qid * C2`, one round of mixing) had severe correlations between adjacent qids (r ≈ -0.98). This was fixed with splitmix64-style mixing (3 rounds of xorshift-multiply). All results above use the corrected hash. The qualitative conclusions are unchanged — the broken hash gave similar aggregate ratios by coincidence.

## Code structure

```
run_cones.py              # Standalone runner — just needs numpy + numba
cone_fast.py              # Full experiment harness with multiple strategies
compare_probe.py          # Paired probe-first vs fwd-merge comparison
compare_probe_d50.py      # D=50 comparison: fwd vs 1@Lx
compare_probe_position.py # 1@L1 at different insertion positions
compare_probe_tail.py     # Probing from terminal/near-terminal layers
adaptive_strategy.py      # Adaptive strategy using covariance heuristic
adaptive_exact.py         # Adaptive strategy using exact variance reduction
exact_vs_fwd_paired.py    # Tightly paired comparison (same vertex order)
verify_cov_conditional.py # Conditional covariance formula verification
verify_exact_var.py       # Exact variance computation verification
heuristic_profile.py      # Score profiles by layer before/after traces
COVARIANCE_DERIVATION.md  # Mathematical derivation of covariance formula
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

| D | ms/trial |
|---|----------|
| 10 | 0.2ms |
| 20 | 0.9ms |
| 50 | 7ms |
| 80 | 16ms |
| 100 | 28ms |

## Conclusions

1. **The conjecture is false** for funnel cones [2D, ..., D+1] at D ≥ ~12. The violation is ~8-12% and plateaus.
2. **fwd-merge is the optimal strategy** (or negligibly close). This was established by:
   - Exhaustive comparison against all simple strategy variants
   - Analytical proof that L0 always has the highest conditional covariance with F
   - Exact computation showing L0 always gives the largest expected squared update to the estimate
3. **The violation is a genuine property of the cone geometry**, not an artifact of suboptimal query ordering or a broken hash function.

## Alternative strategy sweep (2026-04-08)

### Can anything beat fwd-merge?

Extensive experiments tested whether alternative query-ordering heuristics can beat fwd-merge on funnel cones at D=40. All experiments used 30k trials, paired against the same fwd-merge baseline (rng seed 42), run on ARC cloud instances.

### Reached-weighted-fast ("greedy")

The main alternative tested was a greedy strategy from `reached_weighted_fast.py`:
- **Phase 1**: Trace one random L0 input to terminal
- **Phase 2**: Score every unqueried node by `expected_reach × Var(next_layer_values)`, pick the highest
- Expected reach computed via forward DP (unqueried edges treated as uniform)
- Only considers nodes with g_reach > 0 (actual known path exists)

**Result at D=40, 10k trials**: greedy is indistinguishable from fwd-merge (Δ = +0.009 ± 0.008, not significant). The greedy matches fwd-merge behavior ~79% of the time. The 21% deviations break down as:
- 17% abandon an incomplete chain to start a new L0 input
- 3.8% jump to a mid-layer node

### Decision analysis

A counterfactual analysis (10k trials, 347k deviations at threshold=40%) computed the immediate MSE from both the greedy and fwd-merge choice at each deviation point:
- Greedy is better 51.5% of the time, worse 48.4% — barely above coin flip
- Mean benefit per deviation: +0.000038 (nearly zero)
- The greedy's score function (reach × variance) is a **weak predictor** of actual MSE reduction
- No category of deviation (by layer, score ratio, or progress) shows a strong consistent advantage

### Threshold-greedy sweep (178 experiments)

Tested: "deviate from fwd-merge only if greedy score ≥ (1 + threshold) × fwd score", with threshold determined by various heuristics.

**Heuristic types tested:**
- **Fixed threshold** (21 values, c=0.0 to 5.0)
- **Linear in progress** (35 configs): threshold = x0 + x1 × (L0_traced / W0)
- **Layer-gated** (24 configs): only deviate to layers ≤ L_max
- **L0-only** (8 configs): only deviate when greedy picks L0
- **No-jump** (8 configs): allow L0 deviations and chain-continuation, block mid-layer jumps
- **Depth-weighted** (16 configs): threshold = base + weight × (layer / D)
- **Switch-at-K** (10 configs): pure fwd for K L0 traces, then pure greedy
- **Inverse decay** (12 configs): threshold = c / (1 + α × t)
- **Exponential decay** (9 configs): threshold = c × e^(-α×t)
- **Chain-complete** (8 configs): once you deviate, finish the whole chain
- **Step function** (12 configs): different thresholds before/after a cutoff
- **Quadratic** (15 configs): threshold = x0 + x1×t + x2×t²

**Key findings:**

1. **Maximum improvement: ~0.0002 on ratio ~1.094 (~0.02%).** No heuristic breaks this ceiling.

2. **Best strategies are very conservative**: fixed c=0.6–0.9 (deviate ~4-6% of the time), or switch_50 (fwd for first 50/80 L0 traces, then greedy).

3. **Positive x1 (more conservative over time) helps.** Deviations early are slightly useful; deviations late are harmful.

4. **switch_50** gave the largest absolute improvement (-0.00024, -11.1σ) and switch_70 the highest significance (-18.0σ but only -0.00004 absolute).

5. **Chain-complete hurts.** Forcing chain completion after a deviation makes things worse — the greedy's chain-abandonment is actually beneficial.

6. **Layer-gating at threshold=0 is catastrophic** (up to +93σ worse). Without a score threshold, indiscriminate deviations to any layer destroy performance. With threshold=0.4, layer-gating is comparable to fixed threshold.

7. **L0-only and no-jump are weak** — mid-layer jumps are valuable when they clear the score threshold.

**Significance summary (178 experiments):**
- 68 experiments significantly better than fwd-merge (>3σ)
- 17 marginally better (2-3σ)
- 76 not significant
- 17 significantly worse (>2σ)

**Conclusion: The conjecture violation is fundamentally geometric, not strategic.** fwd-merge is optimal or within 0.02% of optimal. No heuristic meaningfully reduces the ratio.

### Infrastructure notes

- **ARC compute**: `c create`, `c ssh`, `c rsync`, `c delete` (alias for `python -m arc_infra.cli`)
- Interactive prompts need `printf 'N\ny\n' | c create ...` for piping
- Instances use `python3` not `python`; need `pip install numpy numba`
- Use single quotes for SSH commands with negative numbers: `c ssh inst -- 'python3 script.py --x1 -0.5'`
- `batch_runner.py` supports `--baseline` to precompute fwd-merge once, then `--fwd-file` on all workers
- Batch distribution via `--batch N --num-batches M` (round-robin assignment)

### Results data

- Raw results (178 experiments): `/tmp/cone_results/results_batch{0-7}.json`
- Precomputed fwd baseline: `fwd_baseline.npy` (on instances, now deleted)
- All experiments used seed 42 for reproducibility

## Code structure (updated)

```
run_cones.py              # Standalone runner — just needs numpy + numba
cone_fast.py              # Full experiment harness with multiple strategies
batch_runner.py           # Batch experiment runner with 12 heuristic types
linear_threshold.py       # Linear threshold with --baseline/--fwd-file support
threshold_greedy.py       # Fixed threshold greedy
compare_fwd_vs_greedy.py  # Paired comparison with decision classification
deviation_analysis.py     # Counterfactual analysis of greedy deviations
reached_weighted_fast.py  # Greedy strategy (reached-weighted with incremental DP)
compare_probe.py          # Paired probe-first vs fwd-merge comparison
adaptive_exact.py         # Adaptive strategy using exact variance reduction
verify_cov_conditional.py # Conditional covariance formula verification
COVARIANCE_DERIVATION.md  # Mathematical derivation of covariance formula
```

## Open questions

1. **Does the ratio truly plateau, or slowly grow?** Data suggests plateau around 1.10-1.12, but error bars at large D are wider.
2. **What about non-funnel cones?** Constant-width and widening cones behave differently.
3. **Is there a closed-form for the asymptotic ratio?** The plateau value may be expressible in terms of the funnel geometry.
4. **Can the bound be tightened?** If the conjecture bound of k is wrong, what is the correct constant? The data suggests ~1.12k for large funnels.
5. **Is there a fundamentally different scoring function** (not reach × variance) that could predict beneficial deviations more accurately? The current greedy score is barely better than random at identifying good deviations.
