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


def test_a_cancelled_build_leaves_the_queue_without_building():
    from backend import store

    built = []
    store._build_lock.acquire()  # another build is running
    try:
        job = jobs.submit("test", lambda progress: built.append(store.minecraft_build("no-such-run", "auto", 300, progress)))
        t0 = time.time()
        while job.stage != "waiting for another build" and time.time() - t0 < 5:
            time.sleep(0.01)
        jobs.cancel(job.id)
        _wait(job)
    finally:
        store._build_lock.release()
    assert job.state == "error" and job.error == "cancelled" and not built
