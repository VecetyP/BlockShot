import json

import pytest

from backend import store


def test_delete_run_removes_it_and_refuses_anything_outside_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "abc").mkdir(parents=True)
    (runs / "abc" / "meta.json").write_text(json.dumps({"id": "abc"}))
    (tmp_path / "meta.json").write_text("{}")  # a meta.json outside runs/ must stay untouchable
    monkeypatch.setattr(store, "RUNS", runs)
    store._builds[("abc", "auto", 300.0)] = object()
    store.delete_run("abc")
    assert not (runs / "abc").exists() and ("abc", "auto", 300.0) not in store._builds
    for bad in ("abc", "..", "../runs"):
        with pytest.raises(FileNotFoundError):
            store.delete_run(bad)
    assert (tmp_path / "meta.json").exists()
