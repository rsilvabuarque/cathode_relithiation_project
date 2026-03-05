from __future__ import annotations

from typing import Any

from ase import Atoms


def load_fairchem_model(model_name: str, task_name: str, device: str, compute_stress: bool) -> Any:
    try:
        from torch_sim.models.fairchem import FairChemModel
    except Exception as exc:  # pragma: no cover - import guard for optional runtime
        raise RuntimeError(
            "torch-sim-atomistic is required for TorchSim MD execution. "
            "Install dependencies from pyproject.toml."
        ) from exc

    return FairChemModel(
        model=model_name,
        task_name=task_name,
        compute_stress=compute_stress,
        device=device,
    )


def prepare_atoms_for_task(atoms: Atoms, task_name: str, charge: int | None, spin: int | None) -> Atoms:
    if task_name == "omol":
        atoms.info["charge"] = 0 if charge is None else int(charge)
        atoms.info["spin"] = 1 if spin is None else int(spin)
    return atoms
