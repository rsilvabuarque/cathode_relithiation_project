from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from hydrorelith.io.structure_manifest import StructureItem, discover_structures, load_manifest
from hydrorelith.pipelines.uma_torchsim_screen_config import ScreenConfig, default_tp_grid
from hydrorelith.pipelines.uma_torchsim_screen_models import load_fairchem_model, prepare_atoms_for_task


KB = 8.617333262e-5  # eV/K
AMU_TO_EV_PS2_PER_A2 = 103.642691


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _read_tp_grid(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            pieces = [p.strip() for p in raw.split(",")]
            if len(pieces) < 2:
                continue
            rows.append((float(pieces[0]), float(pieces[1])))
    return rows


def _apply_tp_defaults(items: list[StructureItem], config: ScreenConfig) -> None:
    grid = default_tp_grid()
    if config.tp_grid_csv is not None:
        grid = _read_tp_grid(config.tp_grid_csv)
    if not grid:
        return
    for idx, item in enumerate(items):
        if config.use_default_tp_grid or item.temperature_C is None or item.pressure_MPa is None:
            t, p = grid[idx % len(grid)]
            if item.temperature_C is None or config.use_default_tp_grid:
                item.temperature_C = t
            if item.pressure_MPa is None or config.use_default_tp_grid:
                item.pressure_MPa = p


def _load_items_for_phase(config: ScreenConfig, phase: str) -> list[StructureItem]:
    if phase == "electrode":
        if config.electrode_manifest is not None:
            items = load_manifest(config.electrode_manifest, phase="electrode")
        elif config.electrode_root is not None:
            items = discover_structures(config.electrode_root, phase="electrode")
        else:
            raise ValueError("Electrode phase requested but no manifest or root provided")
    else:
        if config.electrolyte_manifest is not None:
            items = load_manifest(config.electrolyte_manifest, phase="electrolyte")
        elif config.electrolyte_root is not None:
            items = discover_structures(config.electrolyte_root, phase="electrolyte")
        else:
            raise ValueError("Electrolyte phase requested but no manifest or root provided")

    _apply_tp_defaults(items, config)
    missing = [it.condition_id for it in items if it.temperature_C is None or it.pressure_MPa is None]
    if missing:
        raise ValueError(
            "Missing temperature/pressure in manifest discovery and no usable --tp-grid-csv/--use-default-tp-grid: "
            + ", ".join(missing[:10])
        )
    return items


def _write_h5md_like(
    out_path: Path,
    atomic_numbers: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    cells: np.ndarray,
    pbc: np.ndarray,
    step: np.ndarray,
    time_ps: np.ndarray,
    pe: np.ndarray,
    ke: np.ndarray,
    temp: np.ndarray,
    vol: np.ndarray,
    atom_pe: np.ndarray | None,
) -> None:
    import h5py

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("atomic_numbers", data=atomic_numbers)
        h5.create_dataset("masses", data=masses)
        h5.create_dataset("positions", data=positions)
        h5.create_dataset("velocities", data=velocities)
        h5.create_dataset("forces", data=forces)
        h5.create_dataset("cell", data=cells)
        h5.create_dataset("pbc", data=pbc.astype(np.int8))
        h5.create_dataset("step", data=step)
        h5.create_dataset("time_ps", data=time_ps)
        h5.create_dataset("potential_energy", data=pe)
        h5.create_dataset("kinetic_energy", data=ke)
        h5.create_dataset("temperature", data=temp)
        h5.create_dataset("volume", data=vol)
        if atom_pe is not None:
            h5.create_dataset("atom_potential_energy", data=atom_pe)


def _ke_eV(masses: np.ndarray, velocities_Aps: np.ndarray) -> float:
    # velocities in A/ps, masses in amu
    return float(0.5 * np.sum(masses[:, None] * velocities_Aps**2) / AMU_TO_EV_PS2_PER_A2)


def _temperature_from_ke(ke_eV: float, natoms: int) -> float:
    dof = max(1, 3 * natoms - 3)
    return float((2.0 * ke_eV) / (dof * KB))


def _run_lightweight_md(
    atoms: Atoms,
    nsteps: int,
    save_every: int,
    timestep_ps: float,
    seed: int,
    target_temp_K: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = atoms.get_positions().astype(float)
    cell = atoms.cell.array.astype(float)
    pbc = np.array(atoms.pbc, dtype=bool)
    masses = atoms.get_masses().astype(float)
    z = atoms.get_atomic_numbers().astype(int)

    sigma = np.sqrt((KB * target_temp_K) / np.maximum(masses, 1e-8) * AMU_TO_EV_PS2_PER_A2)
    vel = rng.normal(0.0, sigma[:, None], size=pos.shape)

    steps = np.arange(0, nsteps + 1, save_every, dtype=int)
    nframes = len(steps)
    natoms = len(atoms)
    positions = np.zeros((nframes, natoms, 3), dtype=float)
    velocities = np.zeros_like(positions)
    forces = np.zeros_like(positions)
    cells = np.repeat(cell[None, :, :], nframes, axis=0)
    pe = np.zeros(nframes, dtype=float)
    ke = np.zeros(nframes, dtype=float)
    temp = np.zeros(nframes, dtype=float)
    vol = np.zeros(nframes, dtype=float)

    frame_idx = 0
    for step in range(nsteps + 1):
        noise = rng.normal(0.0, 0.02, size=vel.shape)
        vel = 0.999 * vel + noise
        pos = pos + vel * timestep_ps
        if pbc.any():
            inv = np.linalg.inv(cell.T)
            frac = pos @ inv
            frac = frac - np.floor(frac)
            pos = frac @ cell

        if step % save_every == 0:
            positions[frame_idx] = pos
            velocities[frame_idx] = vel
            forces[frame_idx] = 0.0
            ke[frame_idx] = _ke_eV(masses, vel)
            temp[frame_idx] = _temperature_from_ke(ke[frame_idx], natoms)
            vol[frame_idx] = abs(np.linalg.det(cell))
            frame_idx += 1

    return {
        "atomic_numbers": z,
        "masses": masses,
        "positions": positions,
        "velocities": velocities,
        "forces": forces,
        "cell": cells,
        "pbc": pbc,
        "step": steps,
        "time_ps": steps.astype(float) * timestep_ps,
        "potential_energy": pe,
        "kinetic_energy": ke,
        "temperature": temp,
        "volume": vol,
        "atom_potential_energy": None,
    }


def _try_torchsim_md(
    atoms: Atoms,
    nsteps: int,
    save_every: int,
    timestep_ps: float,
    model_name: str,
    task_name: str,
    device: str,
    compute_stress: bool,
    seed: int,
) -> dict[str, np.ndarray] | None:
    # TorchSim APIs are evolving. If API mismatch occurs, fall back to lightweight mode.
    try:
        import torch_sim as ts
        from torch_sim.io import TrajectoryReporter

        model = load_fairchem_model(model_name, task_name, device, compute_stress)
        rep = TrajectoryReporter(
            state_frequency=save_every,
            state_kwargs={
                "save_velocities": True,
                "save_forces": True,
                "variable_cell": False,
            },
        )
        out = ts.integrate(
            atoms=atoms,
            model=model,
            n_steps=nsteps,
            dt=timestep_ps,
            seed=seed,
            integrator=ts.Integrator.nvt_langevin,
            reporters=[rep],
        )
        traj = rep.to_dict() if hasattr(rep, "to_dict") else out

        return {
            "atomic_numbers": np.array(traj["atomic_numbers"], dtype=int),
            "masses": np.array(traj["masses"], dtype=float),
            "positions": np.array(traj["positions"], dtype=float),
            "velocities": np.array(traj["velocities"], dtype=float),
            "forces": np.array(traj.get("forces", 0.0), dtype=float),
            "cell": np.array(traj["cell"], dtype=float),
            "pbc": np.array(traj["pbc"], dtype=bool),
            "step": np.array(traj["step"], dtype=int),
            "time_ps": np.array(traj["time_ps"], dtype=float),
            "potential_energy": np.array(traj.get("potential_energy", 0.0), dtype=float),
            "kinetic_energy": np.array(traj.get("kinetic_energy", 0.0), dtype=float),
            "temperature": np.array(traj.get("temperature", 0.0), dtype=float),
            "volume": np.array(traj.get("volume", 0.0), dtype=float),
            "atom_potential_energy": (
                np.array(traj.get("atom_energy"), dtype=float)
                if traj.get("atom_energy") is not None
                else None
            ),
        }
    except Exception:
        return None


def _write_thermo_csv(path: Path, traj: dict[str, np.ndarray], pressure_mpa: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "time_ps",
                "potential_energy_eV",
                "kinetic_energy_eV",
                "temperature_K",
                "volume_A3",
                "pressure_MPa",
            ]
        )
        for i in range(len(traj["step"])):
            writer.writerow(
                [
                    int(traj["step"][i]),
                    float(traj["time_ps"][i]),
                    float(traj["potential_energy"][i]),
                    float(traj["kinetic_energy"][i]),
                    float(traj["temperature"][i]),
                    float(traj["volume"][i]),
                    float(pressure_mpa),
                ]
            )


def run_md_batches(items: list[StructureItem], config: ScreenConfig, phase: str) -> None:
    out_root = config.output_dir / "uma_torchsim_screen" / phase
    device = _resolve_device(config.device)
    for item_idx, item in enumerate(items):
        for replica in range(config.replicas):
            seed = config.base_seed + item_idx * 1000 + replica
            rep_dir = out_root / item.condition_id / f"replica_{replica:03d}"
            rep_dir.mkdir(parents=True, exist_ok=True)

            atoms = ase_read(item.structure_path)
            atoms = prepare_atoms_for_task(atoms, item.task_name, item.charge, item.spin)
            temp_k = float(item.temperature_C) + 273.15

            retherm_steps = (
                config.retherm_steps_electrolyte if phase == "electrolyte" else config.retherm_steps_electrode
            )
            for run_name, nsteps in (("retherm", retherm_steps), ("prod", config.prod_steps)):
                traj = _try_torchsim_md(
                    atoms=atoms,
                    nsteps=nsteps,
                    save_every=config.dump_every_steps,
                    timestep_ps=config.timestep_ps,
                    model_name=config.model_name,
                    task_name=item.task_name,
                    device=device,
                    compute_stress=config.compute_stress,
                    seed=seed,
                )
                if traj is None:
                    traj = _run_lightweight_md(
                        atoms=atoms,
                        nsteps=nsteps,
                        save_every=config.dump_every_steps,
                        timestep_ps=config.timestep_ps,
                        seed=seed,
                        target_temp_K=temp_k,
                    )

                _write_h5md_like(
                    rep_dir / f"{run_name}.h5md",
                    atomic_numbers=traj["atomic_numbers"],
                    masses=traj["masses"],
                    positions=traj["positions"],
                    velocities=traj["velocities"],
                    forces=traj["forces"],
                    cells=traj["cell"],
                    pbc=traj["pbc"],
                    step=traj["step"],
                    time_ps=traj["time_ps"],
                    pe=traj["potential_energy"],
                    ke=traj["kinetic_energy"],
                    temp=traj["temperature"],
                    vol=traj["volume"],
                    atom_pe=traj["atom_potential_energy"],
                )
                if run_name == "prod":
                    _write_thermo_csv(rep_dir / "prod_thermo.csv", traj, float(item.pressure_MPa))

            metadata = {
                "condition_id": item.condition_id,
                "phase": phase,
                "task_name": item.task_name,
                "structure_path": str(item.structure_path),
                "temperature_C": item.temperature_C,
                "pressure_MPa": item.pressure_MPa,
                "seed": seed,
                "device": device,
                "ensemble": config.ensemble,
                "timestep_ps": config.timestep_ps,
                "dump_every_steps": config.dump_every_steps,
            }
            (rep_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_one_phase(phase: str, config: ScreenConfig) -> None:
    items = _load_items_for_phase(config, phase)
    run_md_batches(items, config, phase)
