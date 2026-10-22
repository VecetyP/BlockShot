import threading
import time

from backend import jobs


def _wait(job, timeout=5):
    t0 = time.time()
    while job.state == "running" and time.time() - t0 < timeout:
        time.sleep(0.01)
    return job


def test_progress_is_visible_while_running_and_result_arrives():
    step = threading.Event()
    go = threading.Event()

    def work(progress):
        progress("halfway", 1, 2, "msg")
        step.set()
        go.wait(5)
        return {"answer": 42}

    job = jobs.submit("test", work)
    assert step.wait(5)
    seen = jobs.as_dict(jobs.get(job.id))
    assert (seen["state"], seen["stage"], seen["done"], seen["total"], seen["message"]) == ("running", "halfway", 1, 2, "msg")
    go.set()
    _wait(job)
    assert job.state == "done" and job.result == {"answer": 42} and job.done == job.total == 2


def test_errors_are_reported_not_lost():
    def work(progress):
        raise ValueError("boom")

    job = _wait(jobs.submit("test", work))
    assert job.state == "error" and "boom" in job.error and job.finished is not None
