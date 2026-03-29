# LLM Conjecture Harness — Handoff for Claude Code

## What this is

A web app for using LLMs to attack mathematical conjectures. The user describes strategies (algorithms) or instances (counterexamples) in natural language, an LLM generates the code, and a dashboard shows a matrix of which strategies work on which instances.

The first (and currently only) problem is the **Oracle-Averaging Conjecture** from the PDF at `oracle_averaging_notes.pdf`. The architecture is general-purpose: new problems are added by subclassing `Problem` in `problems/`.

## Current status

**The app is fully working.** Flask runs on `http://localhost:5111`, the dashboard loads, and the evaluation engine produces correct results.

### Bug fixes applied earlier (still in effect)
1. `compute_F_exact` uses a separate truth oracle (same seed) to avoid leaking values.
2. Oracle values are derived deterministically from `(seed, query_id)` — order-independent.

### Features added in previous sessions

1. **Tunable parameters on entries.** Each strategy/instance can define a `TUNABLE_PARAMS` dict in its code with metadata (default, min, max, type, description). The harness extracts these at submission time and stores them on the entry. A `params` dict in the exec namespace provides runtime values. Users hover over entries with a ⚙ gear icon to see a popover with editable number inputs; changes save via `PATCH /api/entries/<eid>/params` and invalidate stale results.

2. **Two-step LLM code generation pipeline.** When a user submits a natural-language description:
   - **Step 1 (elaborate):** Opus expands the short description into a detailed technical spec. If genuinely ambiguous, returns `UNSURE:` with a clarifying question — the UI shows the question and lets the user answer before proceeding.
   - **Step 2 (generate):** Opus generates code from the elaborated description.
   - Both steps use extended thinking (8K budget for elaboration, 10K for code gen).
   - Both steps receive **full context**: every existing entry's name, description, code, and params.

3. **AI suggestion buttons.** An "AI Suggestions" section at the bottom of the dashboard has "Suggest New Strategy" and "Suggest New Instance" buttons. These call Opus with a 20K thinking budget and all existing entries as context. Opus returns a `---NAME---`, `---DESCRIPTION---`, `---CODE---` structured response that gets parsed and saved automatically.

4. **Automatic backups.** Every write to `entries.json` or `results.json` saves a timestamped copy in `data/backups/` (keeps last 20 per file).

5. **Idempotent seeding.** `seed.py` checks for existing entries by name+role before creating; safe to re-run.

6. **`.env` file for API key.** The app loads `ANTHROPIC_API_KEY` from `.env` at startup. Colleagues on the shared Dropbox just run `python app.py` — no manual key setup.

7. **Paragraph-length descriptions.** Clicking an entry in the dashboard shows a detailed description plus a Tunable Parameters section with current values, types, ranges, and descriptions.

### Changes in this session (2026-03-29)

8. **Fast oracle hashing.** `Oracle.query` now uses `hashlib.md5` instead of constructing a full `np.random.default_rng` per unique query. 4000 unique queries complete in ~2ms (was much slower). This matters for high-k instances like Deep Layered Graph with ~3800 unique oracle locations.

9. **Auto-detect k from instance.** `run_evaluation` probes the instance on a test input before running, counts how many oracle queries it makes, and bumps k to match if the instance needs more than the default. Deep Layered Graph (k=481) now works without manually overriding eval params.

10. **Evaluation timeout.** `run_evaluation` accepts a `timeout` parameter (default 120s). Uses `SIGALRM` to cap wall-clock time, completing as many trials as possible and returning partial results. The summary notes `(partial: N/M trials)` when truncated.

11. **Hybrid exact/MC posterior in strategies.** Sample-Update and Merge-aware now use a two-tier posterior computation:
    - **Exact branching** when unknowns per input ≤ `max_exact_unknowns` (default 10, tunable). This preserves exact results for all k≤10 instances.
    - **Monte Carlo sampling** when unknowns exceed the threshold — samples `num_mc_samples` (default 64, tunable) random oracle assignments and averages. Falls back via a `_TooDeep` exception that abandons partial branching and does pure MC.
    - Both strategies now have `TUNABLE_PARAMS`: `num_mc_samples` and `max_exact_unknowns`.

## File structure

```
llmHarness/
├── app.py                    # Flask backend (routes, .env loader)
├── codegen.py                # Opus-powered elaboration, code gen, suggestions
├── storage.py                # JSON-file persistence with auto-backups
├── seed.py                   # Idempotent seeding of baseline entries
├── run.sh                    # Launch script
├── requirements.txt          # flask, anthropic, numpy
├── .env                      # ANTHROPIC_API_KEY (not in git)
├── .gitignore                # .env, backups, __pycache__
├── oracle_averaging_notes.pdf  # The math paper
├── problems/
│   ├── __init__.py
│   ├── base.py               # Entry (with tunable_params, param_values), Problem ABC
│   └── oracle_averaging.py   # Oracle, Runner, run_evaluation (with param injection)
├── static/
│   └── index.html            # Dashboard (params popover, clarification UI, suggest buttons)
├── data/
│   ├── entries.json           # All strategies and instances
│   ├── results.json           # Evaluation results
│   └── backups/               # Auto-timestamped backups
└── handoff.md                 # This file
```

## Key abstractions

### Entry (problems/base.py)
```python
@dataclass
class Entry:
    id: str
    role: str               # "strategy" or "instance"
    problem_id: str
    name: str
    description: str
    code: str
    created_at: str
    tunable_params: dict    # {name: {default, min, max, type, description}}
    param_values: dict      # current overrides {name: value}
```

### Problem (problems/base.py)
- `evaluate(strategy_code, instance_code, params, strategy_params, instance_params)` — eval params + per-entry tunable params

### Oracle-Averaging specifics (problems/oracle_averaging.py)

**Instance interface:**
```python
def oracle_algorithm(x, query):
    # x: np.ndarray of {0,1}^N, query(q) -> {-1,+1}
    # return float in [-1, 1]
```

**Strategy interface:**
```python
def estimate(runner, N, k, budget):
    # runner.run_on_input(x), runner.oracle.query(q), runner.oracle.revealed
    # runner.instance_fn for direct instance simulation
    # return float estimate of F(O)
```

**Tunable params convention:** Code defines `TUNABLE_PARAMS` dict and `params` dict at module level. The evaluation engine injects user overrides into `params` via the exec namespace before calling the functions.

**Evaluation** (`run_evaluation`): For each oracle sample, computes F(O) exactly via a truth oracle, then for each stage d=1..max_d creates a fresh stage oracle, calls estimate() with budget=dk, records (μ̂ − F)². Default params: N=8, k=4, max_d=20, num_oracle_samples=50, timeout=120. The k is auto-detected from the instance (bumped if the instance needs more than the default).

### Codegen pipeline (codegen.py)

**Model:** `claude-opus-4-6` with extended thinking for all calls.

**`elaborate_description(problem, role, description, clarification)`** → `{status: "elaborated", description}` or `{status: "unsure", question}`. Thinking budget: 8K tokens.

**`generate_code(problem, role, description)`** → Python code string. Thinking budget: 10K tokens.

**`suggest_entry(problem, role)`** → `{name, description, code}`. Thinking budget: 20K tokens.

**`extract_tunable_params(code)`** → dict. Execs code in sandbox, extracts `TUNABLE_PARAMS`.

**`_build_context(problem_id)`** → string with all existing entries (names, descriptions, code, params). Included in all LLM prompts.

### Storage (storage.py)
JSON files in `data/`. Auto-backups on every write (timestamped copies in `data/backups/`, keeps 20). Key functions:
- `save_entry`, `list_entries`, `get_entry`, `delete_entry`
- `update_param_values(entry_id, values)` — saves overrides, invalidates associated results
- `make_entry(problem_id, role, name, desc, code, tunable_params)` — creates Entry with defaults derived from tunable_params

### Flask API endpoints (app.py)

- `GET /` — serves dashboard
- `GET/POST /api/config` — API key management
- `GET /api/problems` — list problems
- `GET /api/problems/<pid>/entries` — `{strategies, instances}` with tunable_params/param_values
- `GET /api/problems/<pid>/results` — all results
- `POST /api/problems/<pid>/submit` — elaborate + generate code pipeline, handles `clarification` field for UNSURE responses
- `POST /api/problems/<pid>/submit_code` — save hand-written code (extracts TUNABLE_PARAMS)
- `POST /api/problems/<pid>/suggest` — AI suggestion via heavy reasoning
- `POST /api/problems/<pid>/evaluate` — evaluate one pair (passes entry param_values)
- `POST /api/problems/<pid>/evaluate_all` — batch evaluate
- `PATCH /api/entries/<eid>/params` — update tunable param values (type-coerces, invalidates results)
- `GET /api/entries/<eid>` — get single entry
- `DELETE /api/entries/<eid>` — delete entry + results

## Current entries in database

### Strategies (3)
| ID | Name | Description |
|----|------|-------------|
| d6b94bc8 | Simple averaging | Naive Monte Carlo, E_d ~ 1/d, cumulative diverges. Baseline that fails. |
| 08083c3c | Sample-Update | Samples d inputs, then computes exact conditional E[F\|observed] by branching on unknowns. General-purpose Bayesian estimator. |
| 07d3977a | Merge-aware | Like Sample-Update but runs instance_fn with a smart query wrapper that returns cached oracle values for free. Budget only pays for genuinely new queries. Strictly better than Sample-Update on any instance with query overlap. |

### Instances (5+)
| ID | Name | Params | Description |
|----|------|--------|-------------|
| 1c65e325 | Single query | num_values | Encodes full x as integer, queries oracle at x % num_values. Depth 1. |
| fdd50cdf | Parity of k queries | num_queries, M | k queries at hashed locations, returns product. High-degree Fourier structure. |
| 4355a17e | Adaptive chain | chain_length, modulus, step_multiplier | Sequential queries where each location depends on previous oracle answer. |
| 9ea2790c | Layered graph | width, depth | Random-map layers via log2(W) oracle bits per transition. True [W]→[W] maps. Merging paths. |
| eebbbedf | Deep Layered Graph | width, depth_multiplier | (AI-suggested variant) |
| 5dd882a1 | Cone | fanin, num_layers | (AI-suggested variant) |

### Key results
- Simple averaging fails on all instances (cumulative risk > 1).
- Sample-Update passes all standard instances but **fails on Layered graph** (1.06 > 1) — it wastes budget on redundant oracle queries.
- Merge-aware passes everything, including Layered graph (0.55). Strictly dominates Sample-Update.
- Deep Layered Graph (k=481) now evaluable. Merge-aware gets cumulative risk ≈ 0 — the extreme merging (160 layers, W=8) means nearly all paths coalesce, so F(O) depends on very few effective oracle bits. Sample-Update is slower (budget wasted on random inputs) but also passes.

## How to run

```bash
cd llmHarness
pip install -r requirements.txt
python seed.py          # first time only (idempotent)
python app.py           # starts on http://localhost:5111
```

The `.env` file has the API key — no manual setup needed.

## What to work on next

- The suggest feature can propose novel strategies/instances — try it and evaluate results
- Increase N/k for harder evaluations (may need eval param UI)
- The dashboard doesn't yet let you adjust eval params (N, k, max_d, num_oracle_samples, timeout) from the UI — this was listed as a future direction. k is now auto-detected from instances, but other params still use defaults.
- Consider adding the full multi-bit layered model from the PDF (with configurable k)
- The PDF also discusses the oblivious case proof and the general bound of k — could add instances that specifically probe these boundaries
- The MC posterior introduces noise — strategies with `max_exact_unknowns=0` (pure MC) could be compared against exact to measure approximation quality on standard instances
- Sample-Update is much slower than Merge-aware on high-k instances; could explore strategies that are smarter about which inputs to sample (e.g., targeting inputs that share queries with already-explored paths)
