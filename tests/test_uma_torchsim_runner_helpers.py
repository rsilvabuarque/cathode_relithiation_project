from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

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


def test_benchmark_batch_scaling_writes_sorted_csv(tmp_path: Path, monkeypatch) -> None:
    class FakeIntegrator:
        nvt_langevin = "nvt_langevin"

    class FakeTS:
        Integrator = FakeIntegrator

        @staticmethod
        def initialize_state(atoms, device=None, dtype=None):
            _ = (device, dtype)
            n = len(atoms)
            return SimpleNamespace(
                atomic_numbers=np.array(atoms.get_atomic_numbers(), dtype=np.int64),
                positions=np.zeros((n, 3), dtype=np.float64),
                rng=0,
            )

        @staticmethod
        def concatenate_states(states):
            total = sum(len(s.atomic_numbers) for s in states)
            return SimpleNamespace(
                positions=np.zeros((total, 3), dtype=np.float64),
                atomic_numbers=np.concatenate([s.atomic_numbers for s in states]),
                rng=0,
            )

        @staticmethod
        def integrate(system, model, integrator, n_steps, temperature, timestep, trajectory_reporter, autobatcher, pbar):
            _ = (model, integrator, n_steps, temperature, timestep, trajectory_reporter, autobatcher, pbar)
            return SimpleNamespace(positions=system.positions, atomic_numbers=system.atomic_numbers)

    monkeypatch.setattr(run_mod, "_build_autobatcher", lambda *args, **kwargs: None)
    perf_values = iter([0.0, 1.0, 1.0, 2.0, 2.0, 4.0])
    monkeypatch.setattr(run_mod.time, "perf_counter", lambda: next(perf_values))

    atoms_list = [Atoms("H2"), Atoms("H2O"), Atoms("LiOH")]
    config = SimpleNamespace(
        benchmark_max_systems=3,
        benchmark_step_size=1,
        benchmark_warmup_steps=2,
        benchmark_temperature_k=300.0,
        timestep_ps=0.001,
        benchmark_steps=5,
    )

    run_mod._benchmark_batch_scaling(
        FakeTS(),
        atoms_list,
        model=SimpleNamespace(),
        config=config,
        device="cpu",
        dtype="float32",
        seed=123,
        out_dir=tmp_path,
        memory_scalers=[1.0, 2.0, 3.0],
        resolved_max_memory_scaler=12000.0,
        scaler_source="user_provided",
    )

    csv_path = tmp_path / "batch_benchmark" / "batch_scaling.csv"
    assert csv_path.exists()

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [int(r["n_systems"]) for r in rows] == [1, 2, 3]
    assert [float(r["memory_scaler_total"]) for r in rows] == [1.0, 3.0, 6.0]
    assert all(r["max_memory_scaler_source"] == "user_provided" for r in rows)
    assert all(float(r["estimated_max_memory_scaler"]) == 12000.0 for r in rows)