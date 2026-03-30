"""Simple server-side job tracking with JSON persistence.

Jobs run in background threads and save all events to disk,
so progress is never lost if the client disconnects.
"""

import json
import os
import threading
import time
import uuid

JOBS_DIR = os.path.join(os.path.dirname(__file__), "data", "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory registry: job_id -> job dict
_jobs = {}
_lock = threading.Lock()


def _flush(job):
    """Write job state to disk."""
    path = os.path.join(JOBS_DIR, f"{job['id']}.json")
    with open(path, "w") as f:
        json.dump(job, f, indent=2)


def create_job(job_type, params):
    """Create a new job and return its id."""
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "type": job_type,
        "status": "running",
        "created_at": time.time(),
        "params": params,
        "events": [],
        "result": None,
    }
    with _lock:
        _jobs[job_id] = job
        _flush(job)
    return job_id


def append_event(job_id, event):
    """Append an event to the job's event log and flush to disk."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["events"].append(event)
        _flush(job)


def finish_job(job_id, result=None, status="done"):
    """Mark a job as finished."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = status
        job["result"] = result
        _flush(job)


def get_job(job_id):
    """Get a job by id."""
    with _lock:
        job = _jobs.get(job_id)
        if job:
            return dict(job)  # shallow copy
    # Try loading from disk
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            job = json.load(f)
        with _lock:
            _jobs[job_id] = job
        return dict(job)
    return None


def list_running():
    """List all running jobs."""
    with _lock:
        return [dict(j) for j in _jobs.values() if j["status"] == "running"]


def list_recent(job_type=None, limit=5):
    """List recent jobs, newest first."""
    with _lock:
        jobs_list = list(_jobs.values())
    if job_type:
        jobs_list = [j for j in jobs_list if j["type"] == job_type]
    jobs_list.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return [dict(j) for j in jobs_list[:limit]]


def load_all():
    """Load all jobs from disk on startup. Mark stale running jobs as errors."""
    with _lock:
        for fname in os.listdir(JOBS_DIR):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(JOBS_DIR, fname)
            try:
                with open(path) as f:
                    job = json.load(f)
                if job["status"] == "running":
                    job["status"] = "error"
                    job["events"].append({"type": "error", "error": "Server restarted during execution"})
                    _flush(job)
                _jobs[job["id"]] = job
            except Exception:
                pass


class EventSink:
    """Adapter that lets suggest_entry_streaming push events to a job."""
    def __init__(self, job_id):
        self.job_id = job_id

    def put(self, event):
        append_event(self.job_id, event)


# Load existing jobs on import
load_all()
