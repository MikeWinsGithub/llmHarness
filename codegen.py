"""LLM code generation via Anthropic API."""

import os
from problems.base import Problem

# In-memory API key override (set via /api/config)
_api_key_override: str | None = None


def set_api_key(key: str):
    global _api_key_override
    _api_key_override = key


def get_api_key() -> str | None:
    return _api_key_override or os.environ.get("ANTHROPIC_API_KEY")


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


MODEL = "claude-opus-4-6"


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
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 8000},
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = next(b.text for b in resp.content if b.type == "text").strip()
    if text.upper().startswith("UNSURE:"):
        return {"status": "unsure", "question": text[7:].strip()}
    return {"status": "elaborated", "description": text}


def suggest_entry(problem: Problem, role: str) -> dict:
    """Use heavy reasoning to suggest a novel strategy or instance.

    Returns {"name": "...", "description": "...", "code": "..."}.
    """
    import anthropic

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No Anthropic API key configured.")

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

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=32000,
        thinking={"type": "enabled", "budget_tokens": 20000},
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = next(b.text for b in resp.content if b.type == "text")

    # Parse the three sections
    parts = {}
    for section in ("NAME", "DESCRIPTION", "CODE"):
        marker = f"---{section}---"
        if marker not in text:
            raise ValueError(f"Response missing {marker} section")
        start = text.index(marker) + len(marker)
        # Find next marker or end
        next_markers = [f"---{s}---" for s in ("NAME", "DESCRIPTION", "CODE") if s != section]
        end = len(text)
        for nm in next_markers:
            if nm in text[start:]:
                pos = text.index(nm, start)
                if pos < end:
                    end = pos
        parts[section] = text[start:end].strip()

    code = parts["CODE"]
    # Strip markdown fences if present
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


def generate_code(problem: Problem, role: str, description: str, model: str | None = None) -> str:
    """Generate strategy or instance code from a natural-language description."""
    import anthropic

    model = model or MODEL
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
