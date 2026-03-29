"""JSON-file-backed storage for entries and results."""

import json
import os
import shutil
import time
import uuid
from problems.base import Entry, EvalResult

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "data")
BACKUP_DIR = os.path.join(STORAGE_DIR, "backups")


def _entries_path() -> str:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    return os.path.join(STORAGE_DIR, "entries.json")


def _results_path() -> str:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    return os.path.join(STORAGE_DIR, "results.json")


def _load_json(path: str) -> list | dict:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _save_json(path: str, data):
    # Auto-backup before overwriting
    if os.path.exists(path):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        basename = os.path.basename(path)
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(BACKUP_DIR, f"{basename}.{ts}")
        # Only keep one backup per second; skip if identical
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        # Prune old backups: keep last 20 per file
        existing = sorted(
            [f for f in os.listdir(BACKUP_DIR) if f.startswith(basename + ".")],
        )
        for old in existing[:-20]:
            os.remove(os.path.join(BACKUP_DIR, old))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --- Entries ---

def save_entry(entry: Entry):
    path = _entries_path()
    entries = _load_json(path)
    entries.append({
        "id": entry.id,
        "role": entry.role,
        "problem_id": entry.problem_id,
        "name": entry.name,
        "description": entry.description,
        "code": entry.code,
        "created_at": entry.created_at,
        "tunable_params": entry.tunable_params,
        "param_values": entry.param_values,
    })
    _save_json(path, entries)


def list_entries(problem_id: str, role: str | None = None) -> list[dict]:
    entries = _load_json(_entries_path())
    out = [e for e in entries if e["problem_id"] == problem_id]
    if role:
        out = [e for e in out if e["role"] == role]
    for e in out:
        e.setdefault("tunable_params", {})
        e.setdefault("param_values", {})
    return out


def get_entry(entry_id: str) -> dict | None:
    for e in _load_json(_entries_path()):
        if e["id"] == entry_id:
            e.setdefault("tunable_params", {})
            e.setdefault("param_values", {})
            return e
    return None


def delete_entry(entry_id: str):
    path = _entries_path()
    entries = _load_json(path)
    entries = [e for e in entries if e["id"] != entry_id]
    _save_json(path, entries)
    # Also delete associated results
    rpath = _results_path()
    results = _load_json(rpath)
    results = [r for r in results if r["strategy_id"] != entry_id and r["instance_id"] != entry_id]
    _save_json(rpath, results)


# --- Results ---

def save_result(problem_id: str, strategy_id: str, instance_id: str, result: EvalResult):
    path = _results_path()
    results = _load_json(path)
    # Replace existing result for this pair
    results = [r for r in results if not (r["strategy_id"] == strategy_id and r["instance_id"] == instance_id)]
    results.append({
        "problem_id": problem_id,
        "strategy_id": strategy_id,
        "instance_id": instance_id,
        "metrics": result.metrics,
        "summary": result.summary,
        "details": result.details,
        "conjecture_holds": result.conjecture_holds,
        "timestamp": time.time(),
    })
    _save_json(path, results)


def get_results(problem_id: str) -> list[dict]:
    results = _load_json(_results_path())
    return [r for r in results if r["problem_id"] == problem_id]


def get_result(strategy_id: str, instance_id: str) -> dict | None:
    for r in _load_json(_results_path()):
        if r["strategy_id"] == strategy_id and r["instance_id"] == instance_id:
            return r
    return None


def update_param_values(entry_id: str, param_values: dict):
    """Update param_values for an entry and invalidate its results."""
    path = _entries_path()
    entries = _load_json(path)
    for e in entries:
        if e["id"] == entry_id:
            e.setdefault("param_values", {}).update(param_values)
            break
    _save_json(path, entries)
    # Invalidate results involving this entry
    rpath = _results_path()
    results = _load_json(rpath)
    results = [r for r in results if r["strategy_id"] != entry_id and r["instance_id"] != entry_id]
    _save_json(rpath, results)


def make_entry(problem_id: str, role: str, name: str, description: str, code: str,
               tunable_params: dict | None = None) -> Entry:
    return Entry(
        id=str(uuid.uuid4())[:8],
        role=role,
        problem_id=problem_id,
        name=name,
        description=description,
        code=code,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        tunable_params=tunable_params or {},
        param_values={name: spec["default"] for name, spec in (tunable_params or {}).items()},
    )
