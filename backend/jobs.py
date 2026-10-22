"""Background jobs with progress, for anything slow enough to need a progress bar.

`submit(kind, fn)` runs `fn(progress)` in a worker thread and returns at once; the browser polls
`get(id)`. Work functions report with `progress(stage, done, total, message)`; total 0 means "no
measurable progress inside this stage" (e.g. one model forward pass) and shows as indeterminate.
GPU and analysis work still serialize on the locks in store.py, so jobs can't overload the machine.
"""
import logging
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

log = logging.getLogger("playground")
KEEP_S = 3600  # finished jobs are forgotten after an hour


@dataclass
class Job:
    id: str
    kind: str
    state: str = "running"  # running | done | error
    stage: str = "queued"
    done: float = 0
    total: float = 0
    message: str = ""
    result: Any = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancelled: bool = False


class Cancelled(Exception):
    """Raised from progress() once the browser has given up on the job."""


Progress = Callable[..., None]
_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def submit(kind: str, fn: Callable[[Progress], Any]) -> Job:
    _prune()
    job = Job(id=uuid.uuid4().hex[:12], kind=kind)
    with _lock:
        _jobs[job.id] = job

    def progress(stage: str, done: float = 0, total: float = 0, message: str = "") -> None:
        if job.cancelled:
            raise Cancelled
        job.stage, job.done, job.total, job.message = stage, done, total, message

    def run() -> None:
        try:
            job.result = fn(progress)
            job.state, job.stage, job.done = "done", "done", job.total
        except Cancelled:
            job.state, job.error = "error", "cancelled"
        except Exception as e:  # report to the browser instead of dying silently in a thread
            log.error("job %s (%s) failed:\n%s", job.id, kind, traceback.format_exc())
            job.state, job.error = "error", f"{type(e).__name__}: {e}"
        finally:
            job.finished = time.time()

    threading.Thread(target=run, daemon=True, name=f"job-{kind}-{job.id}").start()
    return job


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def cancel(job_id: str) -> None:
    """Stops the job at its next progress report (work between reports still finishes)."""
    if job := get(job_id):
        job.cancelled = True


def as_dict(job: Job) -> dict:
    d = asdict(job)
    d["elapsed"] = round((job.finished or time.time()) - job.started, 1)
    return d


def _prune() -> None:
    cutoff = time.time() - KEEP_S
    with _lock:
        for k in [k for k, j in _jobs.items() if j.finished and j.finished < cutoff]:
            del _jobs[k]
