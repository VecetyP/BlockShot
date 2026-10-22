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


def test_a_failed_recompute_puts_the_old_run_back(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "abc_moge_1").mkdir(parents=True)
    (runs / "abc_moge_1" / "meta.json").write_text(json.dumps({"id": "abc_moge_1", "image_sha256": "abc", "created": "2026-01-01T00:00:00"}))
    (runs / "abc_moge_1" / "input.png").write_bytes(b"png")
    monkeypatch.setattr(store, "RUNS", runs)

    def boom(*a, **k):
        assert store.list_runs() == []  # the run being redone is hidden meanwhile
        raise RuntimeError("model crashed")

    monkeypatch.setattr(store, "predict_cached", boom)
    with pytest.raises(RuntimeError):
        store.recompute("abc_moge_1")
    assert (runs / "abc_moge_1" / "input.png").read_bytes() == b"png"
    assert [m["id"] for m in store.list_runs()] == ["abc_moge_1"]


def test_only_runs_made_by_an_older_version_are_outdated():
    from backend.models import registry

    model = "depth_anything"
    current = store.run_id("ab" * 32, model, registry.cls(model).version)
    older = store.run_id("ab" * 32, model, "an older version")
    assert not store.is_outdated({"id": current, "image_sha256": "ab" * 32})
    assert store.is_outdated({"id": older, "image_sha256": "ab" * 32})
