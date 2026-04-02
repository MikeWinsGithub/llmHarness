# LLM Conjecture Harness — Handoff for Claude Code

## What this is

A web app for using LLMs to attack mathematical conjectures. The user describes strategies (algorithms) or instances (counterexamples) in natural language, an LLM generates the code, and a dashboard shows a matrix of which strategies work on which instances.

The first (and currently only) problem is the **Oracle-Averaging Conjecture** from the PDF at `oracle_averaging_notes.pdf`. The architecture is general-purpose: new problems are added by subclassing `Problem` in `problems/`.

## Current status

**The app is fully working and deployed on Render** (paid tier, persistent filesystem). The dashboard loads, the evaluation engine produces correct results, and the multi-model suggestion pipeline is operational.

### Bug fixes applied earlier (still in effect)
1. `compute_F_exact` uses a separate truth oracle (same seed) to avoid leaking values.
2. Oracle values are derived deterministically from `(seed, query_id)` — order-independent.

### Features from older sessions

1. **Tunable parameters on entries.** Each strategy/instance can define a `TUNABLE_PARAMS` dict in its code with metadata (default, min, max, type, description). The harness extracts these at submission time and stores them on the entry. A `params` dict in the exec namespace provides runtime values. Users hover over entries with a ⚙ gear icon to see a popover with editable number inputs; changes save via `PATCH /api/entries/<eid>/params` and invalidate stale results.

2. **Two-step LLM code generation pipeline.** When a user submits a natural-language description:
   - **Step 1 (elaborate):** Opus expands the short description into a detailed technical spec. If genuinely ambiguous, returns `UNSURE:` with a clarifying question — the UI shows the question and lets the user answer before proceeding.
   - **Step 2 (generate):** Opus generates code from the elaborated description.
   - Both steps use extended thinking (8K budget for elaboration, 10K for code gen).
   - Both steps receive **full context**: every existing entry's name, description, code, and params.

3. **Automatic backups.** Every write to `entries.json` or `results.json` saves a timestamped copy in `data/backups/` (keeps last 20 per file).

4. **Idempotent seeding.** `seed.py` checks for existing entries by name+role before creating; safe to re-run.

5. **Paragraph-length descriptions.** Clicking an entry in the dashboard shows a detailed description plus a Tunable Parameters section with current values, types, ranges, and descriptions.

6. **Fast oracle hashing.** `Oracle.query` uses `hashlib.md5` instead of constructing a full `np.random.default_rng` per unique query.

7. **Auto-detect k from instance.** `run_evaluation` probes the instance on a test input before running, counts how many oracle queries it makes, and bumps k to match if the instance needs more than the default.

8. **Hybrid exact/MC posterior in strategies.** Sample-Update and Merge-aware use exact branching when unknowns per input ≤ `max_exact_unknowns` (default 10, tunable), falling back to Monte Carlo sampling otherwise.

### Changes in this session (2026-04-02)

9. **Multi-model suggestion pipeline.** The "Suggest" buttons now query three models in parallel:
   - **Claude Opus 4.6** (20K thinking budget, streaming)
   - **GPT 5.4 Pro** (high reasoning effort)
   - **Gemini 3.1 Pro Preview** (10K thinking budget)
   Each generates a candidate (name, description, code). Then **Claude Opus 4.6** judges which is best. The thinking budgets are currently at "medium" — they can be cranked up in `codegen.py` (the generator functions at the top).

10. **Server-side job system.** Both suggestions and evaluations run as background jobs (`jobs.py`). Jobs persist to `data/jobs/*.json` and run to completion regardless of whether the browser is open. The frontend polls every 2 seconds for updates. On page load, running jobs resume and the most recent completed suggestion job is replayed.

11. **Live progress in the dashboard.**
    - Clicking any cell in the results matrix triggers a re-evaluation with a progress bar showing `N/M` trials completed.
    - The suggestion pipeline shows each model's candidate card as it arrives, with timing stats (elapsed seconds, input/output token counts), then the judge's decision.

12. **Timing stats on API calls.** Each model call records elapsed time and token usage (input/output). Displayed next to the model tag on candidate cards and saved in job JSON files.

13. **Evaluation improvements.**
    - Default timeout increased from 120s to 600s.
    - Timeout uses `time.monotonic()` deadline (thread-safe) instead of `SIGALRM`.
    - Auto-scales `num_oracle_samples` and `max_d` for high-k instances (k>20) to avoid timeouts. Summary notes the actual k and max_d used.

14. **Render deployment.** `render.yaml` configures the web service. `app.py` reads `PORT` from env, binds `0.0.0.0`, disables debug mode in production. API keys are set as environment variables in Render's dashboard.

15. **API key bar removed from dashboard.** Keys come from `.env` / environment variables only. No more in-browser key entry.

## File structure

```
llmHarness/
├── app.py                    # Flask backend (routes, .env loader, job endpoints)
├── codegen.py                # Multi-model suggestion pipeline, elaboration, code gen
├── jobs.py                   # Background job tracking with JSON persistence
├── storage.py                # JSON-file persistence with auto-backups
├── seed.py                   # Idempotent seeding of baseline entries
├── render.yaml               # Render deployment config
├── requirements.txt          # flask, anthropic, numpy, openai, google-genai
├── .env                      # API keys (not in git): ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
├── .gitignore                # .env, backups, __pycache__
├── oracle_averaging_notes.pdf  # The math paper
├── problems/
│   ├── __init__.py
│   ├── base.py               # Entry (with tunable_params, param_values), Problem ABC
│   └── oracle_averaging.py   # Oracle, Runner, run_evaluation (with param injection, auto-scaling)
├── static/
│   └── index.html            # Dashboard (progress bars, suggestion debug panel, polling)
├── data/
│   ├── entries.json           # All strategies and instances
│   ├── results.json           # Evaluation results
│   ├── jobs/                  # Background job state (one JSON per job)
│   └── backups/               # Auto-timestamped backups
└── HANDOFF.md                 # This file
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
- `evaluate(strategy_code, instance_code, params, strategy_params, instance_params, progress_callback)` — eval params + per-entry tunable params + optional progress callback

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

**Evaluation** (`run_evaluation`): For each oracle sample, computes F(O) exactly via a truth oracle, then for each stage d=1..max_d creates a fresh stage oracle, calls estimate() with budget=dk, records (μ̂ − F)². Default params: N=8, k=4, max_d=20, num_oracle_samples=50, timeout=600. The k is auto-detected from the instance. For high-k instances (k>20), trials and stages are auto-scaled down.

### Codegen pipeline (codegen.py)

**Models:** Claude Opus 4.6 (`claude-opus-4-6`), GPT 5.4 Pro (`gpt-5.4-pro`), Gemini 3.1 Pro Preview (`gemini-3.1-pro-preview`).

**`elaborate_description(problem, role, description, clarification)`** → `{status: "elaborated", description}` or `{status: "unsure", question}`. Claude only, 8K thinking.

**`generate_code(problem, role, description)`** → Python code string. Claude only, 10K thinking.

**`suggest_entry(problem, role)`** → `{name, description, code, debug}`. Three models in parallel + Claude judge. Returns debug info with all candidates.

**`suggest_entry_streaming(problem, role, event_queue)`** → same, but pushes events to a queue as each model finishes. Used by the job system.

**`extract_tunable_params(code)`** → dict. Execs code in sandbox, extracts `TUNABLE_PARAMS`.

**`_build_context(problem_id)`** → string with all existing entries. Included in all LLM prompts.

**`DEBUG_MODE`** — set to `True` in codegen.py for instant fake responses (pipeline testing).

### Job system (jobs.py)

Background jobs run in threads, saving state to `data/jobs/<id>.json`. Each job has: id, type, status (running/done/error), events list, result.

Key functions: `create_job`, `append_event`, `finish_job`, `get_job`, `list_running`, `list_recent`.

`EventSink` class adapts the job system for `suggest_entry_streaming`'s queue interface.

On startup, `load_all()` marks any stale "running" jobs as errors.

### Flask API endpoints (app.py)

- `GET /` — serves dashboard
- `GET/POST /api/config` — API key management (legacy, keys now come from env)
- `GET /api/problems` — list problems
- `GET /api/problems/<pid>/entries` — `{strategies, instances}` with tunable_params/param_values
- `GET /api/problems/<pid>/results` — all results
- `POST /api/problems/<pid>/submit` — elaborate + generate code pipeline
- `POST /api/problems/<pid>/submit_code` — save hand-written code
- `POST /api/problems/<pid>/suggest` — AI suggestion (synchronous, returns debug)
- `POST /api/problems/<pid>/suggest_stream` — AI suggestion via SSE (legacy)
- `POST /api/problems/<pid>/evaluate` — evaluate one pair (synchronous)
- `POST /api/problems/<pid>/evaluate_stream` — evaluate via SSE (legacy)
- `POST /api/problems/<pid>/evaluate_all` — batch evaluate
- **`POST /api/jobs`** — start a background job (`type`: "suggest" or "evaluate")
- **`GET /api/jobs/<id>`** — poll job status and events (`?after=N` for incremental)
- **`GET /api/jobs`** — list recent jobs (or `?status=running` for running only)
- `PATCH /api/entries/<eid>/params` — update tunable param values
- `GET /api/entries/<eid>` — get single entry
- `DELETE /api/entries/<eid>` — delete entry + results

## Current entries in database

### Strategies (3)
| ID | Name | Description |
|----|------|-------------|
| d6b94bc8 | Simple averaging | Naive Monte Carlo, E_d ~ 1/d, cumulative diverges. Baseline that fails. |
| 08083c3c | Sample-Update | Samples d inputs, then computes exact conditional E[F\|observed] by branching on unknowns. General-purpose Bayesian estimator. |
| 07d3977a | Merge-aware | Like Sample-Update but runs instance_fn with a smart query wrapper that returns cached oracle values for free. Budget only pays for genuinely new queries. Strictly better than Sample-Update on any instance with query overlap. |

### Instances (6)
| ID | Name | Params | Description |
|----|------|--------|-------------|
| 1c65e325 | Single query | num_values | Depth 1. |
| fdd50cdf | Parity of k queries | num_queries, M | Product of k queries at hashed locations. |
| 4355a17e | Adaptive chain | chain_length, modulus, step_multiplier | Sequential adaptive queries. |
| 9ea2790c | Layered graph | width, depth | Random-map layers. |
| eebbbedf | Deep Layered Graph | width, depth_multiplier | 160 layers, W=8, k=481. Extreme merging. |
| 5dd882a1 | Cone | fanin, num_layers | Narrowing funnel graph. Currently fanin=2, num_layers=6 (k=16). |

### Key results (fresh, 2026-04-02)
| | Adaptive chain | Single query | Parity k | Layered graph | Deep Layered | Cone |
|---|---|---|---|---|---|---|
| **Simple avg** | 3.58 ✗ | 3.81 ✗ | 3.35 ✗ | 2.68 ✗ | 0.00 ✓ | 0.95 ✓ |
| **Sample-Update** | 0.13 ✓ | 0.07 ✓ | 0.07 ✓ | **1.04 ✗** | 0.00 ✓ | 0.69 ✓ |
| **Merge-aware** | 0.13 ✓ | 0.06 ✓ | 0.07 ✓ | 0.62 ✓ | 0.00 ✓ | 0.49 ✓ |

Notes:
- Deep Layered Graph results are auto-scaled (max_d=5, k=481) — the 0.00 is genuine due to extreme path merging.
- Sample-Update fails on Layered graph (1.04 > 1) — it wastes budget on redundant queries.
- Merge-aware passes everything currently. The Cone instance is too easy because its output is degree 1 (single terminal oracle query). Modifying it to output a product of all oracle values along the path would make it degree-k and much harder.

## Deployment

**Render** (paid tier, persistent filesystem):
- Auto-deploys from branch `claude/read-handoff-doc-27yPw` on push
- API keys set as environment variables in Render dashboard: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
- `render.yaml` configures the service
- `seed.py` runs at build time

**Local:**
```bash
cd llmHarness
pip install -r requirements.txt
python seed.py          # first time only
python app.py           # starts on http://localhost:5111
```

The `.env` file needs `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`.

**Note on network restrictions:** Claude Code on the web (Anthropic's cloud environment) blocks outbound connections to OpenAI. The GPT and Gemini APIs only work from Render or a local machine.

## What to work on next

- **Fix the Cone instance** — output should be product of all oracle values along the path (degree-k), not just the terminal sign query (degree 1). This would make it genuinely hard for Merge-aware at large depth/fanin.
- **Crank up thinking budgets** once the pipeline is verified working. The generator functions in `codegen.py` control this. Claude can go up to 100K (requires streaming), GPT supports `"xhigh"` effort, Gemini up to 24K thinking budget.
- **Add eval param UI** — the dashboard doesn't let you adjust N, k, max_d, num_oracle_samples, timeout from the UI.
- **Explore novel instances** that could push cumulative risk above 1 — the suggest pipeline can help.
- **Add strategies** that are smarter about which inputs to sample (e.g., targeting inputs that share queries with already-explored paths).
- **Database for persistence** — currently JSON files on disk. If the app grows, consider SQLite or Postgres.
- **Merge to main** — all work is on `claude/read-handoff-doc-27yPw`. Consider merging to main and updating Render to deploy from main.
