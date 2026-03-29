"""Flask backend for the LLM conjecture harness."""

import os
import traceback
from flask import Flask, jsonify, request, send_from_directory

# Load .env file if present (so colleagues don't need to set env vars)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())
from problems.oracle_averaging import OracleAveragingProblem
import storage
import codegen

app = Flask(__name__, static_folder="static")

# Registry of problems
PROBLEMS = {}


def register(problem):
    PROBLEMS[problem.id] = problem


register(OracleAveragingProblem())

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@app.route("/api/config", methods=["GET"])
def get_config():
    key = codegen.get_api_key()
    return jsonify({"has_api_key": bool(key), "key_preview": f"...{key[-6:]}" if key else None})


@app.route("/api/config", methods=["POST"])
def set_config():
    data = request.json
    key = data.get("api_key", "").strip()
    if key:
        codegen.set_api_key(key)
    return jsonify({"has_api_key": bool(codegen.get_api_key())})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.route("/api/problems")
def list_problems():
    return jsonify([
        {"id": p.id, "name": p.name, "description": p.description}
        for p in PROBLEMS.values()
    ])


@app.route("/api/problems/<pid>/entries")
def list_entries(pid):
    strategies = storage.list_entries(pid, "strategy")
    instances = storage.list_entries(pid, "instance")
    return jsonify({"strategies": strategies, "instances": instances})


@app.route("/api/problems/<pid>/results")
def list_results(pid):
    return jsonify(storage.get_results(pid))


@app.route("/api/problems/<pid>/submit", methods=["POST"])
def submit_entry(pid):
    """Submit a new strategy or instance via natural language."""
    problem = PROBLEMS.get(pid)
    if not problem:
        return jsonify({"error": f"Unknown problem: {pid}"}), 404

    data = request.json
    role = data.get("role")  # "strategy" or "instance"
    name = data.get("name", "Unnamed")
    description = data.get("description", "")

    clarification = data.get("clarification")

    if role not in ("strategy", "instance"):
        return jsonify({"error": "role must be 'strategy' or 'instance'"}), 400

    # Step 1: Elaborate the description
    try:
        result = codegen.elaborate_description(problem, role, description, clarification)
    except Exception as e:
        return jsonify({"error": f"Elaboration failed: {e}"}), 500

    if result["status"] == "unsure" and not clarification:
        return jsonify({
            "needs_clarification": True,
            "question": result["question"],
        })

    # If still unsure after clarification, proceed with best effort
    elaborated = result.get("description") or f"{description}\n\n{clarification or ''}"

    # Step 2: Generate code from elaborated description
    try:
        code = codegen.generate_code(problem, role, elaborated)
    except Exception as e:
        return jsonify({"error": f"Code generation failed: {e}"}), 500

    tunable_params = codegen.extract_tunable_params(code)
    entry = storage.make_entry(pid, role, name, elaborated, code, tunable_params)
    storage.save_entry(entry)

    return jsonify({
        "id": entry.id,
        "name": entry.name,
        "code": entry.code,
        "role": entry.role,
        "tunable_params": entry.tunable_params,
        "param_values": entry.param_values,
    })


@app.route("/api/problems/<pid>/submit_code", methods=["POST"])
def submit_code(pid):
    """Submit a strategy or instance with hand-written code (bypass LLM)."""
    problem = PROBLEMS.get(pid)
    if not problem:
        return jsonify({"error": f"Unknown problem: {pid}"}), 404

    data = request.json
    role = data.get("role")
    name = data.get("name", "Unnamed")
    description = data.get("description", "")
    code = data.get("code", "")

    if role not in ("strategy", "instance"):
        return jsonify({"error": "role must be 'strategy' or 'instance'"}), 400

    tunable_params = codegen.extract_tunable_params(code)
    entry = storage.make_entry(pid, role, name, description, code, tunable_params)
    storage.save_entry(entry)

    return jsonify({
        "id": entry.id, "name": entry.name, "code": entry.code, "role": entry.role,
        "tunable_params": entry.tunable_params, "param_values": entry.param_values,
    })


@app.route("/api/problems/<pid>/evaluate", methods=["POST"])
def evaluate(pid):
    """Evaluate a strategy on an instance."""
    problem = PROBLEMS.get(pid)
    if not problem:
        return jsonify({"error": f"Unknown problem: {pid}"}), 404

    data = request.json
    strategy_id = data.get("strategy_id")
    instance_id = data.get("instance_id")
    params = data.get("params")

    strategy = storage.get_entry(strategy_id)
    instance = storage.get_entry(instance_id)

    if not strategy:
        return jsonify({"error": f"Strategy {strategy_id} not found"}), 404
    if not instance:
        return jsonify({"error": f"Instance {instance_id} not found"}), 404

    try:
        result = problem.evaluate(strategy["code"], instance["code"], params,
                                  strategy_params=strategy.get("param_values"),
                                  instance_params=instance.get("param_values"))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Evaluation failed: {e}"}), 500

    storage.save_result(pid, strategy_id, instance_id, result)

    return jsonify({
        "strategy_id": strategy_id,
        "instance_id": instance_id,
        "metrics": result.metrics,
        "summary": result.summary,
        "details": result.details,
        "conjecture_holds": result.conjecture_holds,
    })


@app.route("/api/problems/<pid>/evaluate_all", methods=["POST"])
def evaluate_all(pid):
    """Evaluate all strategy × instance pairs."""
    problem = PROBLEMS.get(pid)
    if not problem:
        return jsonify({"error": f"Unknown problem: {pid}"}), 404

    params = (request.json or {}).get("params")
    strategies = storage.list_entries(pid, "strategy")
    instances = storage.list_entries(pid, "instance")
    results = []

    for s in strategies:
        for inst in instances:
            # Skip if already evaluated
            existing = storage.get_result(s["id"], inst["id"])
            if existing and not (request.json or {}).get("force"):
                results.append(existing)
                continue
            try:
                result = problem.evaluate(s["code"], inst["code"], params,
                                          strategy_params=s.get("param_values"),
                                          instance_params=inst.get("param_values"))
                storage.save_result(pid, s["id"], inst["id"], result)
                results.append({
                    "strategy_id": s["id"],
                    "instance_id": inst["id"],
                    "metrics": result.metrics,
                    "summary": result.summary,
                    "conjecture_holds": result.conjecture_holds,
                })
            except Exception as e:
                results.append({
                    "strategy_id": s["id"],
                    "instance_id": inst["id"],
                    "error": str(e),
                })

    return jsonify(results)


@app.route("/api/problems/<pid>/suggest", methods=["POST"])
def suggest(pid):
    """Ask Opus to suggest a novel strategy or instance."""
    problem = PROBLEMS.get(pid)
    if not problem:
        return jsonify({"error": f"Unknown problem: {pid}"}), 404

    role = (request.json or {}).get("role")
    if role not in ("strategy", "instance"):
        return jsonify({"error": "role must be 'strategy' or 'instance'"}), 400

    try:
        result = codegen.suggest_entry(problem, role)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Suggestion failed: {e}"}), 500

    code = result["code"]
    tunable_params = codegen.extract_tunable_params(code)
    entry = storage.make_entry(pid, role, result["name"], result["description"], code, tunable_params)
    storage.save_entry(entry)

    return jsonify({
        "id": entry.id,
        "name": entry.name,
        "description": entry.description,
        "code": entry.code,
        "role": entry.role,
        "tunable_params": entry.tunable_params,
        "param_values": entry.param_values,
    })


@app.route("/api/entries/<eid>/params", methods=["PATCH"])
def update_params(eid):
    """Update tunable parameter values for an entry."""
    entry = storage.get_entry(eid)
    if not entry:
        return jsonify({"error": "Not found"}), 404
    data = request.json
    # Type-coerce based on tunable_params spec
    tp = entry.get("tunable_params", {})
    coerced = {}
    for k, v in data.items():
        if k in tp:
            if tp[k].get("type") == "int":
                coerced[k] = int(v)
            else:
                coerced[k] = float(v)
        else:
            coerced[k] = v
    storage.update_param_values(eid, coerced)
    return jsonify({"ok": True})


@app.route("/api/entries/<eid>", methods=["DELETE"])
def delete_entry(eid):
    storage.delete_entry(eid)
    return jsonify({"ok": True})


@app.route("/api/entries/<eid>")
def get_entry(eid):
    entry = storage.get_entry(eid)
    if not entry:
        return jsonify({"error": "Not found"}), 404
    return jsonify(entry)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5111))
    debug = os.environ.get("RENDER") is None  # debug off in production
    app.run(debug=debug, host="0.0.0.0", port=port)
