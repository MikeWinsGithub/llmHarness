"""LLM code generation via Anthropic, OpenAI, and Google GenAI APIs."""

import os
import concurrent.futures
from problems.base import Problem

# In-memory API key override (set via /api/config)
_api_key_override: str | None = None


def set_api_key(key: str):
    global _api_key_override
    _api_key_override = key


def get_api_key() -> str | None:
    return _api_key_override or os.environ.get("ANTHROPIC_API_KEY")


def get_openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def get_gemini_api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY")


def extract_tunable_params(code: str) -> dict:
    """Extract TUNABLE_PARAMS dict from generated code."""
    try:
        import numpy as np
        ns = {"np": np}
        exec(code, ns)
        tp = ns.get("TUNABLE_PARAMS")
        if isinstance(tp, dict):
            return tp
        return {}
    except Exception:
        return {}


CLAUDE_MODEL = "claude-opus-4-6"
GPT_MODEL = "gpt-5.4-pro"
GEMINI_MODEL = "gemini-3.1-pro"


def _build_context(problem_id: str) -> str:
    """Build a context string with all existing entries for the problem."""
    import storage
    entries = storage.list_entries(problem_id)
    strategies = [e for e in entries if e["role"] == "strategy"]
    instances = [e for e in entries if e["role"] == "instance"]

    parts = []
    for label, items in [("Existing strategies", strategies), ("Existing instances", instances)]:
        if not items:
            continue
        parts.append(f"# {label}")
        for e in items:
            parts.append(f"\n## {e['name']}")
            parts.append(e["description"])
            if e.get("tunable_params"):
                ps = ", ".join(f'{k} (default={v["default"]}, {v["type"]}, {v.get("description","")})'
                               for k, v in e["tunable_params"].items())
                parts.append(f"Tunable parameters: {ps}")
            parts.append(f"```python\n{e['code'].strip()}\n```")

    return "\n".join(parts) if parts else ""


def elaborate_description(problem: Problem, role: str, description: str,
                          clarification: str | None = None) -> dict:
    """Expand a short description into a detailed spec, or ask for clarification.

    Returns {"status": "elaborated", "description": "..."} or
            {"status": "unsure", "question": "..."}.
    """
    import anthropic

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No Anthropic API key configured.")

    spec = problem.strategy_spec() if role == "strategy" else problem.instance_spec()

    system = (
        "You are a mathematical researcher helping to precisely specify algorithms for "
        "the Oracle-Averaging Conjecture.\n\n"
        "Given a brief description of a strategy (estimator) or instance (oracle algorithm), "
        "expand it into a detailed, unambiguous technical specification that a programmer "
        "could implement correctly.\n\n"
        "Your expanded description should include:\n"
        "- What the algorithm does, step by step\n"
        "- What its parameters are and what they control\n"
        "- Any mathematical intuition for why it works or why it's interesting\n\n"
        "If the description is genuinely ambiguous (multiple reasonable interpretations that "
        "would lead to very different implementations), respond with exactly:\n"
        "UNSURE: followed by a specific, focused question.\n\n"
        "Do NOT ask for clarification on minor details you can fill in with reasonable "
        "defaults — only ask when the core algorithm is unclear.\n\n"
        "Output ONLY the expanded description or the UNSURE: question. No other text."
    )

    context = _build_context(problem.id)

    user_msg = f"""\
# Problem
{problem.name}: {problem.description}

# Code interface
{spec}

{context}

# User's description of the NEW {role} to add
{description}
"""
    if clarification:
        user_msg += f"\n# Additional clarification from the user\n{clarification}\n"

    user_msg += (
        "\nElaborate this into a precise technical specification, "
        "or respond UNSURE: <question> if the core idea is genuinely ambiguous."
    )

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 8000},
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = next(b.text for b in resp.content if b.type == "text").strip()
    if text.upper().startswith("UNSURE:"):
        return {"status": "unsure", "question": text[7:].strip()}
    return {"status": "elaborated", "description": text}


# ---------------------------------------------------------------------------
# Multi-model suggestion pipeline
# ---------------------------------------------------------------------------

def _parse_candidate(text: str) -> dict:
    """Parse ---NAME---, ---DESCRIPTION---, ---CODE--- sections from model output."""
    parts = {}
    for section in ("NAME", "DESCRIPTION", "CODE"):
        marker = f"---{section}---"
        if marker not in text:
            raise ValueError(f"Response missing {marker} section")
        start = text.index(marker) + len(marker)
        next_markers = [f"---{s}---" for s in ("NAME", "DESCRIPTION", "CODE") if s != section]
        end = len(text)
        for nm in next_markers:
            if nm in text[start:]:
                pos = text.index(nm, start)
                if pos < end:
                    end = pos
        parts[section] = text[start:end].strip()

    code = parts["CODE"]
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return {
        "name": parts["NAME"],
        "description": parts["DESCRIPTION"],
        "code": code,
    }


def _generate_claude(system: str, user_msg: str) -> dict:
    """Generate a candidate using Claude Opus 4.6 with extended thinking."""
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key())
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=128000,
        thinking={"type": "adaptive", "effort": "max"},
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        stream=False,
    )
    text = next(b.text for b in resp.content if b.type == "text")
    result = _parse_candidate(text)
    result["model"] = "Claude Opus 4.6"
    return result


def _generate_gpt(system: str, user_msg: str) -> dict:
    """Generate a candidate using GPT 5.4 Pro Max with high reasoning effort."""
    from openai import OpenAI
    client = OpenAI(api_key=get_openai_api_key())
    resp = client.responses.create(
        model=GPT_MODEL,
        instructions=system,
        input=user_msg,
        max_output_tokens=128000,
        reasoning={"effort": "xhigh", "summary": "auto"},
    )
    text = resp.output_text
    result = _parse_candidate(text)
    result["model"] = "GPT 5.4 Pro"
    return result


def _generate_gemini(system: str, user_msg: str) -> dict:
    """Generate a candidate using Gemini 3.1 with thinking enabled."""
    from google import genai
    client = genai.Client(api_key=get_gemini_api_key())
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_msg,
        config=genai.types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=genai.types.ThinkingConfig(thinking_level="MAX"),
            max_output_tokens=65536,
        ),
    )
    text = resp.text
    result = _parse_candidate(text)
    result["model"] = "Gemini 3.1"
    return result


def _judge_candidates(candidates: list[dict], problem: Problem, role: str) -> dict:
    """Use Claude Opus 4.6 to judge which candidate is best."""
    import anthropic

    labels = "ABCDEFGHIJ"
    system = (
        "You are an expert judge evaluating algorithm proposals for a mathematical research problem. "
        "You will be shown multiple candidate algorithms. Pick the single best one based on:\n"
        "1. Mathematical soundness and novelty\n"
        "2. Code correctness and cleanliness\n"
        "3. Clear, well-motivated description\n"
        "4. Likelihood of performing well on the problem\n\n"
        "Respond with ONLY the letter of the best candidate (e.g. A, B, or C). Nothing else."
    )

    parts = [f"# Problem\n{problem.name}: {problem.description}\n"]
    for i, c in enumerate(candidates):
        parts.append(
            f"## Candidate {labels[i]} ({c['model']})\n"
            f"**Name:** {c['name']}\n"
            f"**Description:** {c['description']}\n"
            f"```python\n{c['code']}\n```\n"
        )
    parts.append(f"Which candidate is best? Respond with only the letter ({', '.join(labels[i] for i in range(len(candidates)))}).")

    client = anthropic.Anthropic(api_key=get_api_key())
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive", "effort": "max"},
        system=system,
        messages=[{"role": "user", "content": "\n".join(parts)}],
    )
    choice = next(b.text for b in resp.content if b.type == "text").strip().upper()

    # Parse the letter
    for i, label in enumerate(labels[:len(candidates)]):
        if choice.startswith(label):
            return candidates[i]

    # Fallback: return first candidate
    return candidates[0]


def suggest_entry(problem: Problem, role: str) -> dict:
    """Use multiple models with heavy reasoning to suggest a novel strategy or instance.

    Phase 1: Claude, GPT, and Gemini each generate a candidate in parallel.
    Phase 2: GPT 5.4 Pro Max judges which candidate is best.

    Returns {"name": "...", "description": "...", "code": "..."}.
    """
    spec = problem.strategy_spec() if role == "strategy" else problem.instance_spec()
    context = _build_context(problem.id)

    if role == "strategy":
        task = (
            "Suggest a novel estimation STRATEGY that could perform better than the existing ones. "
            "Think deeply about the mathematical structure of the problem. Consider approaches from "
            "Bayesian estimation, information theory, adaptive sampling, importance weighting, or "
            "any other technique that could reduce the cumulative Bayes risk. The strategy should "
            "be meaningfully different from existing strategies — not a minor tweak."
        )
    else:
        task = (
            "Suggest a novel oracle algorithm INSTANCE that could stress-test the conjecture. "
            "Think deeply about what makes estimation hard. Consider instances with complex "
            "adaptive query patterns, high-degree Fourier structure, information-hiding schemes, "
            "or structural features that could push the cumulative risk close to or above 1. "
            "The instance should be meaningfully different from existing instances."
        )

    system = (
        "You are a world-class mathematician and algorithm designer working on the "
        "Oracle-Averaging Conjecture. Your task is to propose a novel, creative, and "
        "mathematically well-motivated algorithm.\n\n"
        "You MUST respond in EXACTLY this format — three sections separated by the exact "
        "delimiters shown:\n\n"
        "---NAME---\n"
        "A short name for the algorithm\n"
        "---DESCRIPTION---\n"
        "A detailed paragraph explaining the algorithm, its mathematical motivation, "
        "why it should work well, and what its parameters control.\n"
        "---CODE---\n"
        "The complete Python code implementing the algorithm.\n\n"
        "The code must follow the specification below. numpy is available as np. "
        "If the implementation has tunable parameters, define TUNABLE_PARAMS and params "
        "at the top of the code."
    )

    user_msg = f"""\
# Problem
{problem.name}: {problem.description}

# Code specification
{spec}

{context}

# Results summary
Review the existing strategies and instances above. Look at what works and what doesn't.

# Your task
{task}

Respond with ---NAME---, ---DESCRIPTION---, and ---CODE--- sections."""

    # Phase 1: parallel candidate generation
    generators = []
    if get_api_key():
        generators.append(("Claude", _generate_claude))
    if get_openai_api_key():
        generators.append(("GPT", _generate_gpt))
    if get_gemini_api_key():
        generators.append(("Gemini", _generate_gemini))

    if not generators:
        raise RuntimeError("No API keys configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY in .env.")

    candidates = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fn, system, user_msg): label
            for label, fn in generators
        }
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                candidates.append(future.result())
            except Exception as e:
                errors.append(f"{label}: {e}")
                print(f"[suggest] {label} generation failed: {e}")

    if not candidates:
        raise RuntimeError(f"All model generations failed: {'; '.join(errors)}")

    # Phase 2: judge picks the best (skip if only 1 candidate)
    if len(candidates) == 1:
        winner = candidates[0]
    elif get_api_key():
        try:
            winner = _judge_candidates(candidates, problem, role)
        except Exception as e:
            print(f"[suggest] Judge failed, using first candidate: {e}")
            winner = candidates[0]
    else:
        # No Anthropic key for judging — just return first candidate
        winner = candidates[0]

    return {"name": winner["name"], "description": winner["description"], "code": winner["code"]}


def generate_code(problem: Problem, role: str, description: str, model: str | None = None) -> str:
    """Generate strategy or instance code from a natural-language description."""
    import anthropic

    model = model or CLAUDE_MODEL
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "No Anthropic API key configured. Set it in the dashboard settings "
            "or export ANTHROPIC_API_KEY in your shell."
        )

    if role == "strategy":
        spec = problem.strategy_spec()
    elif role == "instance":
        spec = problem.instance_spec()
    else:
        raise ValueError(f"Unknown role: {role}")

    system = (
        "You are a mathematical programmer. You write clean, correct Python code. "
        "Output ONLY the Python code — no markdown fences, no explanation, no comments "
        "beyond what's in the code itself. numpy is available as np."
    )

    context = _build_context(problem.id)

    user_msg = f"""\
# Problem
{problem.name}: {problem.description}

# Code specification
{spec}

{context}

# What to implement
{description}

# Tunable parameters
If the implementation has natural parameters (constants that could meaningfully be varied),
define them at the top of your code like this:

TUNABLE_PARAMS = {{
    "param_name": {{"default": value, "min": min_val, "max": max_val, "type": "int", "description": "..."}},
}}
params = {{name: spec["default"] for name, spec in TUNABLE_PARAMS.items()}}

Then use params["param_name"] in the code instead of hardcoding the constant.
Types must be "int" or "float". If there are no meaningful tunable parameters, omit TUNABLE_PARAMS.

Write the code now. Output only the Python code, nothing else."""

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    code = next(b.text for b in resp.content if b.type == "text")

    # Strip markdown fences if the model included them anyway
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code
