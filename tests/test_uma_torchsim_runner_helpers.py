from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hydrorelith.pipelines import uma_torchsim_screen_run as run_mod


def test_resume_plan_truncates_to_common_last_step(tmp_path: Path, monkeypatch) -> None:
    files = [tmp_path / "a.h5", tmp_path / "b.h5"]
    for path in files:
        path.touch()

    calls: dict[str, int | bool] = {"truncated_to": -1, "closed": False}

    class FakeReporter:
        def __init__(self, paths, trajectory_kwargs):
            _ = (paths, trajectory_kwargs)

        def truncate_to_step(self, step: int) -> None:
            calls["truncated_to"] = step

        def close(self) -> None:
            calls["closed"] = True

    class FakeTS:
        TrajectoryReporter = FakeReporter

        @staticmethod
        def concatenate_states(states):
            return tuple(states)

    monkeypatch.setattr(run_mod, "_h5_last_step", lambda _ts, p: 10 if p.name == "a.h5" else 8)
    monkeypatch.setattr(run_mod, "_load_resume_state", lambda _ts, p, device, dtype: f"state-{p.stem}-{device}-{dtype}")

    plan = run_mod._determine_resume_plan(FakeTS(), files, n_steps_target=20, device="cpu", dtype="float32")

    assert plan["resume_mode"] is True
    assert plan["reporter_mode"] == "a"
    assert plan["resume_from_step"] == 8
    assert plan["steps_remaining"] == 12
    assert plan["completed"] is False
    assert plan["state"] == ("state-a-cpu-float32", "state-b-cpu-float32")
    assert calls["truncated_to"] == 8
    assert calls["closed"] is True


def test_resume_plan_requires_all_or_none_existing_files(tmp_path: Path) -> None:
    files = [tmp_path / "a.h5", tmp_path / "b.h5"]
    files[0].touch()

    with pytest.raises(RuntimeError, match="all cohort trajectories"):
        run_mod._determine_resume_plan(SimpleNamespace(), files, n_steps_target=10, device="cpu", dtype="float32")


def test_resume_plan_with_existing_files_and_no_steps_starts_fresh(tmp_path: Path, monkeypatch) -> None:
    files = [tmp_path / "a.h5", tmp_path / "b.h5"]
    for path in files:
        path.touch()

    monkeypatch.setattr(run_mod, "_h5_last_step", lambda _ts, _p: None)

    plan = run_mod._determine_resume_plan(SimpleNamespace(), files, n_steps_target=10, device="cpu", dtype="float32")

    assert plan["resume_mode"] is False
    assert plan["reporter_mode"] == "w"
    assert plan["resume_from_step"] == 0
    assert plan["steps_remaining"] == 10
    assert plan["completed"] is False
    assert plan["state"] is None

