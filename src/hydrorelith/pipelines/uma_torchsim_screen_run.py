from __future__ import annotations

import csv
import json
import math
import secrets
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from tqdm.auto import tqdm

from hydrorelith.io.structure_manifest import StructureItem, discover_structures, load_manifest
from hydrorelith.pipelines.uma_torchsim_screen_config import ScreenConfig, default_tp_grid
from hydrorelith.pipelines.uma_torchsim_screen_models import load_fairchem_model, prepare_atoms_for_task


KB = 8.617333262e-5  # eV/K
AMU_TO_EV_PS2_PER_A2 = 103.642691
AMU_TO_G = 1.66053906660e-24
A3_TO_CM3 = 1.0e-24
PA_PER_EV_A3 = 1.602176634e11
EV_A3_PER_ATM = 101325.0 / PA_PER_EV_A3
THERMO_COLUMNS = [
    "step",
    "time_fs",
    "temperature_K",
    "pressure_eV_per_A3",
    "pressure_atm",
    "volume_A3",
    "density_g_cm3",
    "potential_energy_eV",
    "kinetic_energy_eV",
    "total_energy_eV",
    "delta_total_energy_eV",
    "wall_delta_s",
]


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class _EmaRateTracker:
    def __init__(self, alpha: float = 0.30) -> None:
        self.alpha = float(alpha)
        self.steps_per_sec: float | None = None

    def update(self, nsteps: int, elapsed_s: float) -> None:
        if nsteps <= 0 or elapsed_s <= 0:
            return
        sample = float(nsteps) / float(elapsed_s)
        if self.steps_per_sec is None:
            self.steps_per_sec = sample
        else:
            self.steps_per_sec = self.alpha * sample + (1.0 - self.alpha) * self.steps_per_sec

    def eta(self, remaining_steps: int) -> float | None:
        if self.steps_per_sec is None or self.steps_per_sec <= 0:
            return None
        return float(remaining_steps) / self.steps_per_sec


def _debug_log(config: ScreenConfig, msg: str) -> None:
    if config.debug:
        tqdm.write(f"[debug] {msg}")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _density_g_cm3(total_mass_amu: float, volume_A3: np.ndarray | float) -> np.ndarray:
    vol = np.asarray(volume_A3, dtype=float)
    safe_vol = np.where(vol > 1.0e-12, vol, np.nan)
    return (float(total_mass_amu) * AMU_TO_G) / (safe_vol * A3_TO_CM3)


def _select_equilibrated_start_idx(density_g_cm3: np.ndarray) -> int:
    dens = np.asarray(density_g_cm3, dtype=float)
    n = int(dens.size)
    if n <= 6:
        return max(0, n // 2)

    finite = np.isfinite(dens)
    if not np.all(finite):
        idx = np.where(finite)[0]
        if idx.size == 0:
            return max(0, n // 2)
        dens = dens[idx]
        n = int(dens.size)
        if n <= 6:
            return max(0, n // 2)

    tail_start = max(1, int(0.75 * n))
    tail = dens[tail_start:]
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))
    tol = max(0.5 * tail_std, 0.01 * max(tail_mean, 1.0e-6), 0.003)

    start_min = max(1, int(0.20 * n))
    start_max = max(start_min + 1, int(0.85 * n))
    for s in range(start_min, start_max):
        seg_mean = float(np.mean(dens[s:]))
        if abs(seg_mean - tail_mean) <= tol:
            return s
    return max(1, int(0.50 * n))


def _set_atoms_volume_isotropic(atoms: Atoms, target_volume_A3: float) -> None:
    target_v = float(target_volume_A3)
    if target_v <= 1.0e-12:
        return
    cell = atoms.cell.array.astype(float)
    cur_v = abs(float(np.linalg.det(cell)))
    if cur_v <= 1.0e-12:
        return
    scale = (target_v / cur_v) ** (1.0 / 3.0)
    atoms.set_cell(cell * scale, scale_atoms=True)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _update_run_state(path: Path, **updates: Any) -> dict[str, Any]:
    state = _read_json_if_exists(path) or {}
    state.update(updates)
    state["updated_at_unix_s"] = int(time.time())
    _write_json_atomic(path, state)
    return state


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


def _read_atoms_any(path: Path | str) -> Atoms:
    p = Path(path)
    name_upper = p.name.upper()
    ext = p.suffix.lower()
    if name_upper.startswith("POSCAR") or ext == ".vasp":
        return ase_read(p, format="vasp")
    if ext in {".data", ".lammps", ".lmp"} or name_upper.startswith("DATA."):
        return ase_read(p, format="lammps-data")
    return ase_read(p)


def _write_h5md_flat(
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


def _write_thermo_csv(path: Path, traj: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_ps", "potential_energy_eV", "kinetic_energy_eV", "temperature_K", "volume_A3", "density_g_cm3", "pressure_MPa"])
        for i in range(len(traj["step"])):
            writer.writerow(
                [
                    int(traj["step"][i]),
                    float(traj["time_ps"][i]),
                    float(traj["potential_energy"][i]),
                    float(traj["kinetic_energy"][i]),
                    float(traj["temperature"][i]),
                    float(traj["volume"][i]),
                    float(traj["density_g_cm3"][i]),
                    float(traj["pressure_MPa"][i]),
                ]
            )


def _get_dtype(config: ScreenConfig):
    import torch

    return torch.float64 if config.precision == "float64" else torch.float32


def _build_autobatcher(ts, model, config: ScreenConfig, max_memory_scaler: float | None):
    model_device = getattr(model, "device", None)
    if str(model_device) == "cpu":
        return False
    if max_memory_scaler is None:
        return True
    return ts.BinningAutoBatcher(model=model, max_memory_scaler=float(max_memory_scaler))


def _pick_integrator(ts, ensemble: str):
    if ensemble == "npt":
        return ts.Integrator.npt_nose_hoover
    return ts.Integrator.nvt_langevin


def _build_prop_calculators(ts):
    prev_total = {"value": None}
    prev_wall = {"value": None}

    def _volume(state, model=None):
        import torch

        return torch.abs(torch.linalg.det(state.cell))

    def _kinetic_energy(state, model=None):
        return ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)

    def _potential_energy(state, model=None):
        import torch

        if torch.is_tensor(state.energy):
            if state.energy.ndim == 0:
                return state.energy.reshape(1).repeat(int(state.cell.shape[0]))
            return state.energy.reshape(-1)
        return torch.full((int(state.cell.shape[0]),), float(state.energy), device=state.positions.device, dtype=state.positions.dtype)

    def _temperature(state, model=None):
        return ts.calc_temperature(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)

    def _density(state, model=None):
        import torch

        nsys = int(state.cell.shape[0])
        m = torch.zeros(nsys, device=state.masses.device, dtype=state.masses.dtype)
        m.index_add_(0, state.system_idx.long(), state.masses)
        v = _volume(state)
        return (m * AMU_TO_G) / (v * A3_TO_CM3)

    def _pressure_ev_a3(state, model=None):
        if model is None:
            raise RuntimeError("Pressure reporter requires model")
        out = model(state)
        if "stress" not in out:
            raise RuntimeError("Model output has no stress tensor; set compute_stress=True")
        return ts.get_pressure(out["stress"], _kinetic_energy(state), _volume(state))

    def _pressure_atm(state, model=None):
        return _pressure_ev_a3(state, model) * (PA_PER_EV_A3 / 101325.0)

    def _atom_potential_energy(state, model=None):
        if model is None:
            raise RuntimeError("Per-atom energy reporter requires model")

        out = model(state)
        for key in (
            "energies",
            "atom_potential_energy",
            "pe_atom",
            "potential_energy_per_atom",
            "potential_energy_atom",
            "energy_per_atom_eV",
        ):
            if key not in out:
                continue
            values = out[key]
            if values is None:
                continue
            if values.ndim == 0:
                continue
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1:
                continue
            return values

        raise RuntimeError(
            "Model output has no per-atom potential-energy array; ensure model "
            "is configured with compute_per_atom_energies=True"
        )

    def _total_energy(state, model=None):
        return _potential_energy(state) + _kinetic_energy(state)

    def _delta_total(state, model=None):
        import torch

        cur = _total_energy(state).detach().clone()
        if prev_total["value"] is None:
            out = torch.full_like(cur, float("nan"))
        else:
            out = cur - prev_total["value"]
        prev_total["value"] = cur
        return out

    def _wall_delta(state, model=None):
        import torch

        now = time.perf_counter()
        nsys = int(state.cell.shape[0])
        if prev_wall["value"] is None:
            out = torch.full((nsys,), float("nan"), device=state.positions.device, dtype=state.positions.dtype)
        else:
            out = torch.full((nsys,), now - prev_wall["value"], device=state.positions.device, dtype=state.positions.dtype)
        prev_wall["value"] = now
        return out

    return {
        1: {
            "temperature_K": lambda state, model=None: _temperature(state),
            "pressure_eV_per_A3": lambda state, model=None: _pressure_ev_a3(state, model),
            "pressure_atm": lambda state, model=None: _pressure_atm(state, model),
            "volume_A3": lambda state, model=None: _volume(state),
            "density_g_cm3": lambda state, model=None: _density(state),
            "potential_energy_eV": lambda state, model=None: _potential_energy(state),
            "atom_potential_energy": lambda state, model=None: _atom_potential_energy(state, model),
            "kinetic_energy_eV": lambda state, model=None: _kinetic_energy(state),
            "total_energy_eV": lambda state, model=None: _total_energy(state),
            "delta_total_energy_eV": _delta_total,
            "wall_delta_s": _wall_delta,
        }
    }


def _export_thermo_from_torchsim(ts, input_h5: Path, out_csv: Path, timestep_fs: float) -> None:
    rows_by_step: dict[int, dict[str, float]] = {}
    with ts.TorchSimTrajectory(str(input_h5), mode="r") as traj:
        for key in THERMO_COLUMNS[2:]:
            try:
                steps = np.asarray(traj.get_steps(key), dtype=int)
                values = np.asarray(traj.get_array(key))
            except Exception:
                continue

            if values.ndim == 0:
                values = values.reshape(1, 1)
            elif values.ndim == 1:
                values = values.reshape(-1, 1)
            else:
                values = values.reshape(values.shape[0], -1)
            if values.shape[1] != 1:
                continue

            for step, value in zip(steps, values[:, 0], strict=True):
                row = rows_by_step.setdefault(int(step), {"step": int(step), "time_fs": float(step) * timestep_fs})
                row[key] = float(value)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=THERMO_COLUMNS)
        writer.writeheader()
        for step in sorted(rows_by_step):
            row = {k: rows_by_step[step].get(k, math.nan) for k in THERMO_COLUMNS}
            writer.writerow(row)


def _load_flat_traj_for_analysis(ts, torchsim_h5: Path, flat_h5: Path, pressure_mpa: float, timestep_ps: float) -> dict[str, np.ndarray]:
    def _frame_scalars(arr: np.ndarray, nframes: int) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.ndim == 0:
            out = np.full((nframes,), float(out), dtype=float)
        elif out.ndim == 1:
            if out.shape[0] == 1 and nframes > 1:
                out = np.full((nframes,), float(out[0]), dtype=float)
        else:
            out = out.reshape(out.shape[0], -1)
            if out.shape[1] == 1:
                out = out[:, 0]
            elif out.shape[0] == 1 and nframes > 1:
                out = np.full((nframes,), float(out[0, 0]), dtype=float)
            else:
                out = out[:, 0]
        return out

    def _frame_atom_array(arr: np.ndarray, nframes: int, natoms: int) -> np.ndarray | None:
        out = np.asarray(arr, dtype=float)
        if out.ndim == 0:
            return None
        if out.ndim == 1:
            if nframes == 1 and out.shape[0] == natoms:
                return out.reshape(1, natoms)
            return None
        out = out.reshape(out.shape[0], -1)
        if out.shape == (nframes, natoms):
            return out
        if out.shape[0] == 1 and out.shape[1] == natoms and nframes > 1:
            return np.repeat(out, nframes, axis=0)
        return None

    with ts.TorchSimTrajectory(str(torchsim_h5), mode="r") as traj:
        nframes = len(traj)
        if nframes <= 0:
            raise RuntimeError(f"Empty trajectory generated at {torchsim_h5}")
        steps = np.asarray(traj.get_steps("positions"), dtype=int)
        positions = np.asarray(traj.get_array("positions"), dtype=float)
        natoms = int(positions.shape[1])
        velocities = np.asarray(traj.get_array("velocities"), dtype=float) if "velocities" in traj.array_registry else np.zeros_like(positions)
        forces = np.asarray(traj.get_array("forces"), dtype=float) if "forces" in traj.array_registry else np.zeros_like(positions)
        cells = np.asarray(traj.get_array("cell"), dtype=float)
        if cells.ndim == 2:
            cells = np.repeat(cells[None, :, :], nframes, axis=0)
        if cells.shape[0] == 1 and nframes > 1:
            cells = np.repeat(cells, nframes, axis=0)

        atom_potential_energy = None
        if "atom_potential_energy" in traj.array_registry:
            atom_potential_energy = _frame_atom_array(
                np.asarray(traj.get_array("atom_potential_energy"), dtype=float),
                nframes,
                natoms,
            )

        if "potential_energy_eV" in traj.array_registry:
            pe_raw = np.asarray(traj.get_array("potential_energy_eV"), dtype=float)
            pe_atom_from_total = _frame_atom_array(pe_raw, nframes, natoms)
            if pe_atom_from_total is not None:
                atom_potential_energy = pe_atom_from_total
                potential_energy = np.sum(pe_atom_from_total, axis=1)
            else:
                potential_energy = _frame_scalars(pe_raw, nframes)
        else:
            potential_energy = np.zeros((nframes,), dtype=float)

        if atom_potential_energy is None:
            for key in ("energies", "pe_atom", "potential_energy_per_atom", "potential_energy_atom", "energy_per_atom_eV"):
                if key not in traj.array_registry:
                    continue
                candidate = _frame_atom_array(np.asarray(traj.get_array(key), dtype=float), nframes, natoms)
                if candidate is not None:
                    atom_potential_energy = candidate
                    break

        kinetic_energy = _frame_scalars(np.asarray(traj.get_array("kinetic_energy_eV"), dtype=float), nframes) if "kinetic_energy_eV" in traj.array_registry else np.zeros((nframes,), dtype=float)
        temperature = _frame_scalars(np.asarray(traj.get_array("temperature_K"), dtype=float), nframes) if "temperature_K" in traj.array_registry else np.zeros((nframes,), dtype=float)
        volume = _frame_scalars(np.asarray(traj.get_array("volume_A3"), dtype=float), nframes) if "volume_A3" in traj.array_registry else np.abs(np.linalg.det(cells))

        atoms0 = traj.get_atoms(0)
        atomic_numbers = np.asarray(atoms0.get_atomic_numbers(), dtype=int)
        masses = np.asarray(atoms0.get_masses(), dtype=float)
        pbc = np.asarray(atoms0.pbc, dtype=bool)

    density = _density_g_cm3(float(np.sum(masses)), volume)
    time_ps = steps.astype(float) * float(timestep_ps)
    pressure_mpa_series = np.full((len(steps),), float(pressure_mpa), dtype=float)
    out = {
        "atomic_numbers": atomic_numbers,
        "masses": masses,
        "positions": positions,
        "velocities": velocities,
        "forces": forces,
        "cell": cells,
        "pbc": pbc,
        "step": steps,
        "time_ps": time_ps,
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
        "temperature": temperature,
        "volume": volume,
        "atom_potential_energy": atom_potential_energy,
        "density_g_cm3": density,
        "pressure_MPa": pressure_mpa_series,
    }
    _write_h5md_flat(
        flat_h5,
        atomic_numbers=out["atomic_numbers"],
        masses=out["masses"],
        positions=out["positions"],
        velocities=out["velocities"],
        forces=out["forces"],
        cells=out["cell"],
        pbc=out["pbc"],
        step=out["step"],
        time_ps=out["time_ps"],
        pe=out["potential_energy"],
        ke=out["kinetic_energy"],
        temp=out["temperature"],
        vol=out["volume"],
        atom_pe=out["atom_potential_energy"],
    )
    return out


def _h5_last_step(ts, path: Path) -> int | None:
    if not path.exists():
        return None
    with ts.TorchSimTrajectory(str(path), mode="r") as traj:
        return traj.last_step


def _load_resume_state(ts, path: Path, device, dtype):
    import torch

    with ts.TorchSimTrajectory(str(path), mode="r") as traj:
        frame = len(traj) - 1
        state = traj.get_state(frame=frame, device=device, dtype=dtype)
        if "velocities" in traj.array_registry:
            velocities = torch.tensor(traj.get_array("velocities", start=frame, stop=frame + 1)[0], device=device, dtype=dtype)
            momenta = velocities * state.masses.unsqueeze(-1)
            forces = torch.tensor(traj.get_array("forces", start=frame, stop=frame + 1)[0], device=device, dtype=dtype) if "forces" in traj.array_registry else torch.zeros_like(state.positions)
            energy = torch.zeros((state.n_systems,), device=device, dtype=dtype)
            state_cls = getattr(ts, "MDState", None)
            if state_cls is None:
                state_cls = getattr(ts, "SimState", None)
            if state_cls is not None and hasattr(state_cls, "from_state"):
                return state_cls.from_state(state, momenta=momenta, energy=energy, forces=forces)
            return state
        return state


def _truncate_to_step(ts, files: list[Path], step: int) -> None:
    rep = ts.TrajectoryReporter([str(p) for p in files], trajectory_kwargs={"mode": "a"})
    rep.truncate_to_step(step)
    rep.close()


def _determine_resume_plan(ts, files: list[Path], n_steps_target: int, device, dtype):
    existing = [p.exists() for p in files]
    plan: dict[str, Any] = {
        "resume_mode": False,
        "reporter_mode": "w",
        "resume_from_step": 0,
        "steps_remaining": n_steps_target,
        "completed": False,
        "state": None,
    }
    if not any(existing):
        return plan
    if not all(existing):
        raise RuntimeError("Resume requires either all cohort trajectories to exist or none")

    steps = [_h5_last_step(ts, p) for p in files]
    nonnull = [int(s) for s in steps if s is not None]
    if not nonnull:
        # Existing files but no recorded steps: treat as a fresh run.
        # Keep reporter in write mode so stale/partial files are replaced.
        return plan

    common_last = min(nonnull)
    if len(set(nonnull)) != 1:
        _truncate_to_step(ts, files, common_last)

    states = [_load_resume_state(ts, p, device=device, dtype=dtype) for p in files]
    plan["resume_mode"] = True
    plan["reporter_mode"] = "a"
    plan["resume_from_step"] = common_last
    plan["state"] = ts.concatenate_states(states)
    plan["steps_remaining"] = max(0, int(n_steps_target) - common_last)
    plan["completed"] = common_last >= int(n_steps_target)
    return plan


def _resolve_memory_scalers(ts, states, model, config: ScreenConfig):
    if str(getattr(model, "device", "")) == "cpu":
        return [float(len(s.atomic_numbers)) for s in states], None, "cpu_disabled"

    metric_name = getattr(model, "memory_scales_with", "n_atoms")
    try:
        metrics = list(ts.calculate_memory_scalers(states, memory_scales_with=metric_name))
    except Exception:
        metrics = [float(len(s.atomic_numbers)) for s in states]

    resolved = config.max_memory_scaler
    source = "user_provided" if resolved is not None else "estimated"
    if resolved is None:
        try:
            resolved = float(ts.estimate_max_memory_scaler(states, model, metrics))
        except Exception:
            source = "unavailable"
            resolved = None
    return metrics, resolved, source


def _benchmark_counts(config: ScreenConfig, n_total: int) -> list[int]:
    max_systems = config.benchmark_max_systems or n_total
    max_systems = min(max_systems, n_total)
    counts = list(range(1, max_systems + 1, max(1, config.benchmark_step_size)))
    if counts and counts[-1] != max_systems:
        counts.append(max_systems)
    return counts


def _save_benchmark_rows(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["n_systems"]))
    out_csv = out_dir / "batch_scaling.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _benchmark_batch_scaling(ts, atoms_list: list[Atoms], model, config: ScreenConfig, *, device, dtype, seed: int, out_dir: Path, memory_scalers: list[float], resolved_max_memory_scaler: float | None, scaler_source: str) -> None:
    counts = _benchmark_counts(config, len(atoms_list))
    if not counts:
        return
    bench_dir = out_dir / "batch_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for nsys in counts:
        subset = atoms_list[:nsys]
        states = [ts.initialize_state(at, device=device, dtype=dtype) for at in subset]
        for idx, st in enumerate(states):
            st.rng = int(seed + idx)
        state = ts.concatenate_states(states)
        state.rng = int(seed)

        warm = ts.integrate(
            system=state,
            model=model,
            integrator=ts.Integrator.nvt_langevin,
            n_steps=config.benchmark_warmup_steps,
            temperature=float(config.benchmark_temperature_k),
            timestep=float(config.timestep_ps),
            trajectory_reporter=None,
            autobatcher=_build_autobatcher(ts, model, config, resolved_max_memory_scaler),
            pbar=False,
        )
        t0 = time.perf_counter()
        final = ts.integrate(
            system=warm,
            model=model,
            integrator=ts.Integrator.nvt_langevin,
            n_steps=config.benchmark_steps,
            temperature=float(config.benchmark_temperature_k),
            timestep=float(config.timestep_ps),
            trajectory_reporter=None,
            autobatcher=_build_autobatcher(ts, model, config, resolved_max_memory_scaler),
            pbar=False,
        )
        elapsed = time.perf_counter() - t0
        natoms = int(final.positions.shape[0])
        atom_steps_per_s = (natoms * config.benchmark_steps) / max(elapsed, 1e-12)
        scaler_total = float(sum(memory_scalers[:nsys])) if memory_scalers else math.nan
        row = {
            "n_systems": nsys,
            "total_atoms": natoms,
            "steps": int(config.benchmark_steps),
            "elapsed_s": float(elapsed),
            "atom_steps_per_s": float(atom_steps_per_s),
            "memory_scaler_total": scaler_total,
            "estimated_max_memory_scaler": resolved_max_memory_scaler if resolved_max_memory_scaler is not None else math.nan,
            "max_memory_scaler_source": scaler_source,
        }
        rows.append(row)
        _save_benchmark_rows(rows, bench_dir)


def _is_prod_segment_name(run_name: str) -> bool:
    return run_name.startswith("prod_2pt_")


def _stage_group_key(item: StructureItem, run_name: str) -> tuple[float, float | None, str]:
    temp_k = float(item.temperature_C) + 273.15
    if run_name == "minimize":
        # Minimization is temperature-independent; grouping by task maximizes batching.
        return (1.0, None, item.task_name)
    if run_name in {"heat", "retherm"} or _is_prod_segment_name(run_name) or run_name == "prod":
        pressure = float(item.pressure_MPa) if run_name == "retherm" else None
        return (temp_k, pressure, item.task_name)
    pressure = float(item.pressure_MPa) if run_name == "retherm" else None
    return (temp_k, pressure, item.task_name)


def _split_steps_for_segments(total_steps: int, n_segments: int) -> list[int]:
    if total_steps <= 0 or n_segments <= 0:
        return []
    base = total_steps // n_segments
    rem = total_steps % n_segments
    out = [base + (1 if idx < rem else 0) for idx in range(n_segments)]
    return [v for v in out if v > 0]


def _trajectory_paths(rep_dir: Path, run_name: str) -> tuple[Path, Path, Path]:
    ts_h5 = rep_dir / f"{run_name}.trajectory.h5"
    flat_h5 = rep_dir / f"{run_name}.h5md"
    thermo_csv = rep_dir / f"{run_name}_thermo.csv"
    return ts_h5, flat_h5, thermo_csv


def _integrate_with_step_eta(
    ts,
    integrate_kwargs: dict[str, Any],
    total_steps: int,
    *,
    bar_desc: str,
    group_size: int,
    global_bar,
    global_tracker: _EmaRateTracker,
) -> float:
    if total_steps <= 0:
        return 0.0

    chunk_steps = max(1, min(1000, int(total_steps)))
    local_tracker = _EmaRateTracker(alpha=0.30)
    state = integrate_kwargs["system"]
    base_kwargs = {k: v for k, v in integrate_kwargs.items() if k not in {"system", "n_steps"}}

    done = 0
    elapsed_total = 0.0
    step_bar = tqdm(total=int(total_steps), desc=bar_desc, unit="step", leave=False)
    try:
        while done < int(total_steps):
            n_chunk = min(chunk_steps, int(total_steps) - done)
            call_kwargs = dict(base_kwargs)
            call_kwargs["system"] = state
            call_kwargs["n_steps"] = int(n_chunk)

            t0 = time.perf_counter()
            state = ts.integrate(**call_kwargs)
            elapsed = time.perf_counter() - t0
            elapsed_total += elapsed
            done += int(n_chunk)

            local_tracker.update(int(n_chunk), elapsed)
            global_tracker.update(int(n_chunk) * int(group_size), elapsed)
            step_bar.update(int(n_chunk))
            global_bar.update(int(n_chunk) * int(group_size))

            eta_local = local_tracker.eta(int(total_steps) - done)
            local_rate = local_tracker.steps_per_sec
            remaining_global = max(0, int(global_bar.total) - int(global_bar.n))
            eta_global = global_tracker.eta(remaining_global)

            postfix_bits = []
            if local_rate is not None:
                postfix_bits.append(f"rate={local_rate:.1f} step/s")
            if eta_local is not None:
                postfix_bits.append(f"eta={_format_duration(eta_local)}")
            if eta_global is not None:
                postfix_bits.append(f"eta_total={_format_duration(eta_global)}")
            if postfix_bits:
                step_bar.set_postfix_str(" | ".join(postfix_bits))
    finally:
        step_bar.close()
    return elapsed_total


def _integrate_with_temperature_ramp_eta(
    ts,
    integrate_kwargs: dict[str, Any],
    total_steps: int,
    *,
    temperature_start_k: float,
    temperature_end_k: float,
    bar_desc: str,
    group_size: int,
    global_bar,
    global_tracker: _EmaRateTracker,
) -> float:
    if total_steps <= 0:
        return 0.0

    chunk_steps = max(1, min(1000, int(total_steps)))
    local_tracker = _EmaRateTracker(alpha=0.30)
    state = integrate_kwargs["system"]
    base_kwargs = {k: v for k, v in integrate_kwargs.items() if k not in {"system", "n_steps", "temperature"}}

    done = 0
    elapsed_total = 0.0
    step_bar = tqdm(total=int(total_steps), desc=bar_desc, unit="step", leave=False)
    try:
        while done < int(total_steps):
            n_chunk = min(chunk_steps, int(total_steps) - done)
            frac = float(done + n_chunk) / float(total_steps)
            target_temp = float(temperature_start_k) + (float(temperature_end_k) - float(temperature_start_k)) * frac

            call_kwargs = dict(base_kwargs)
            call_kwargs["system"] = state
            call_kwargs["n_steps"] = int(n_chunk)
            call_kwargs["temperature"] = float(target_temp)

            t0 = time.perf_counter()
            state = ts.integrate(**call_kwargs)
            elapsed = time.perf_counter() - t0
            elapsed_total += elapsed
            done += int(n_chunk)

            local_tracker.update(int(n_chunk), elapsed)
            global_tracker.update(int(n_chunk) * int(group_size), elapsed)
            step_bar.update(int(n_chunk))
            global_bar.update(int(n_chunk) * int(group_size))

            eta_local = local_tracker.eta(int(total_steps) - done)
            eta_global = global_tracker.eta(max(0, int(global_bar.total) - int(global_bar.n)))
            postfix_bits = [f"T={target_temp:.1f}K"]
            if local_tracker.steps_per_sec is not None:
                postfix_bits.append(f"rate={local_tracker.steps_per_sec:.1f} step/s")
            if eta_local is not None:
                postfix_bits.append(f"eta={_format_duration(eta_local)}")
            if eta_global is not None:
                postfix_bits.append(f"eta_total={_format_duration(eta_global)}")
            step_bar.set_postfix_str(" | ".join(postfix_bits))
    finally:
        step_bar.close()
    return elapsed_total


def _optimize_with_step_eta(
    ts,
    optimize_kwargs: dict[str, Any],
    total_steps: int,
    *,
    bar_desc: str,
    group_size: int,
    global_bar,
    global_tracker: _EmaRateTracker,
) -> float:
    if total_steps <= 0:
        return 0.0

    tqdm.write(f"[md] {bar_desc}: running FIRE minimization up to {int(total_steps)} steps")
    t0 = time.perf_counter()
    _ = ts.optimize(**optimize_kwargs)
    elapsed = time.perf_counter() - t0
    global_bar.update(int(total_steps) * int(group_size))
    global_tracker.update(int(total_steps) * int(group_size), max(elapsed, 1.0e-12))
    return elapsed


def _concat_h5md_segments(segment_paths: list[Path], out_path: Path) -> None:
    if not segment_paths:
        return
    import h5py

    frame_keys = {
        "positions",
        "velocities",
        "forces",
        "cell",
        "step",
        "time_ps",
        "potential_energy",
        "kinetic_energy",
        "temperature",
        "volume",
        "atom_potential_energy",
    }

    static_data: dict[str, np.ndarray] = {}
    frame_data: dict[str, list[np.ndarray]] = {}
    for seg_path in segment_paths:
        if not seg_path.exists():
            continue
        with h5py.File(seg_path, "r") as h5:
            for key in h5.keys():
                arr = np.asarray(h5[key])
                if key in frame_keys and arr.ndim >= 1:
                    frame_data.setdefault(key, []).append(arr)
                elif key not in static_data:
                    static_data[key] = arr

    if not frame_data:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        for key, arr in static_data.items():
            h5.create_dataset(key, data=arr)
        for key, pieces in frame_data.items():
            if not pieces:
                continue
            merged = np.concatenate(pieces, axis=0)
            h5.create_dataset(key, data=merged)


def _concat_thermo_segments(segment_paths: list[Path], out_path: Path) -> None:
    if not segment_paths:
        return
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for seg_path in segment_paths:
        if not seg_path.exists():
            continue
        with seg_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    if fieldnames is None:
        return

    def _step_key(row: dict[str, str]) -> int:
        try:
            return int(float(row.get("step", "0")))
        except ValueError:
            return 0

    rows.sort(key=_step_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finalize_production_segments(rep_dir: Path, prod_stage_names: list[str]) -> None:
    if not prod_stage_names:
        return
    h5_segments = [rep_dir / f"{name}.h5md" for name in prod_stage_names]
    thermo_segments = [rep_dir / f"{name}_thermo.csv" for name in prod_stage_names]
    _concat_h5md_segments(h5_segments, rep_dir / "prod.h5md")
    _concat_thermo_segments(thermo_segments, rep_dir / "prod_thermo.csv")


def _state_to_atoms_list(state, atoms_list: list[Atoms]) -> list[Atoms]:
    def _to_numpy(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            return np.asarray(x.detach().cpu().numpy())
        return np.asarray(x)

    positions = _to_numpy(state.positions).astype(float, copy=False)
    system_idx = _to_numpy(state.system_idx).astype(int, copy=False)
    cell = _to_numpy(state.cell).astype(float, copy=False)

    out: list[Atoms] = []
    cursor = 0
    for sys_i, atoms in enumerate(atoms_list):
        nat = int(len(atoms))
        sel = np.where(system_idx == int(sys_i))[0]
        if int(sel.size) == nat:
            pos_sys = positions[sel]
        else:
            # Fall back to contiguous slicing if system indices were reordered.
            pos_sys = positions[cursor : cursor + nat]
        cursor += nat

        atoms_next = atoms.copy()
        atoms_next.set_positions(np.asarray(pos_sys, dtype=float))
        if cell.ndim == 3 and sys_i < cell.shape[0]:
            atoms_next.set_cell(np.asarray(cell[sys_i], dtype=float), scale_atoms=False)
        out.append(atoms_next)
    return out


def run_md_batches(items: list[StructureItem], config: ScreenConfig, phase: str) -> None:
    import torch
    import torch_sim as ts

    out_root = config.output_dir / "uma_torchsim_screen" / phase
    out_root.mkdir(parents=True, exist_ok=True)

    device_name = _resolve_device(config.device)
    device = torch.device(device_name)
    dtype = _get_dtype(config)

    model = load_fairchem_model(config.model_name, "omol" if phase == "electrolyte" else "omat", device_name, True)
    initial_atoms = []
    for item in items:
        atoms = _read_atoms_any(item.structure_path)
        atoms = prepare_atoms_for_task(atoms, item.task_name, item.charge, item.spin)
        initial_atoms.append(atoms)

    initial_states = [ts.initialize_state(at, device=device, dtype=dtype) for at in initial_atoms]
    memory_scalers, resolved_max_memory_scaler, scaler_source = _resolve_memory_scalers(ts, initial_states, model, config)
    retherm_steps = int(config.retherm_steps_electrolyte if phase == "electrolyte" else config.retherm_steps_electrode)
    production_segment_steps = _split_steps_for_segments(int(config.prod_steps), int(config.production_stages))
    production_stage_names = [f"prod_2pt_{idx + 1:02d}" for idx in range(len(production_segment_steps))]

    stage_plan: list[tuple[str, int]] = []
    if int(config.minimize_steps) > 0:
        stage_plan.append(("minimize", int(config.minimize_steps)))
    if int(config.heat_steps) > 0:
        stage_plan.append(("heat", int(config.heat_steps)))
    if retherm_steps > 0:
        stage_plan.append(("retherm", retherm_steps))
    for name, steps in zip(production_stage_names, production_segment_steps, strict=True):
        if int(steps) > 0:
            stage_plan.append((name, int(steps)))

    if not stage_plan:
        raise ValueError("No MD stages configured. Increase --minimize-steps/--heat-steps/--retherm-steps/--prod-steps.")

    run_config_path = out_root / "run_config.json"
    _write_json_atomic(
        run_config_path,
        {
            "phase": phase,
            "model_name": config.model_name,
            "device": device_name,
            "precision": config.precision,
            "resolved_max_memory_scaler": resolved_max_memory_scaler,
            "max_memory_scaler_source": scaler_source,
            "replicas": config.replicas,
            "minimize_steps": int(config.minimize_steps),
            "heat_steps": int(config.heat_steps),
            "heat_start_temperature_k": float(config.heat_start_temperature_k),
            "retherm_steps": retherm_steps,
            "prod_steps": config.prod_steps,
            "production_stages": int(config.production_stages),
            "production_stage_names": production_stage_names,
            "stage_plan": [{"name": name, "steps": int(steps)} for name, steps in stage_plan],
            "dump_every_steps": config.dump_every_steps,
            "timestep_ps": config.timestep_ps,
            "benchmark_skipped": config.skip_batch_benchmark,
        },
    )

    if not config.skip_batch_benchmark and device.type != "cpu":
        _benchmark_batch_scaling(
            ts,
            initial_atoms,
            model,
            config,
            device=device,
            dtype=dtype,
            seed=config.base_seed,
            out_dir=out_root,
            memory_scalers=memory_scalers,
            resolved_max_memory_scaler=resolved_max_memory_scaler,
            scaler_source=scaler_source,
        )

    total_steps_per_system = int(sum(int(steps) for _, steps in stage_plan))
    total_system_steps = int(config.replicas) * int(len(items)) * int(total_steps_per_system)
    global_tracker = _EmaRateTracker(alpha=0.30)
    global_bar = tqdm(total=total_system_steps, desc=f"{phase} MD system-steps", unit="sys-step")

    for replica in range(config.replicas):
        start_atoms_by_condition: dict[str, Atoms] = {}
        for item_idx, item in enumerate(items):
            atoms = _read_atoms_any(item.structure_path)
            atoms = prepare_atoms_for_task(atoms, item.task_name, item.charge, item.spin)
            start_atoms_by_condition[item.condition_id] = atoms

        for run_name, nsteps in stage_plan:
            grouped: dict[tuple[float, float | None, str], list[tuple[int, StructureItem]]] = {}
            for item_idx, item in enumerate(items):
                key = _stage_group_key(item, run_name)
                grouped.setdefault(key, []).append((item_idx, item))

            for group_key, cohort in grouped.items():
                temp_k, pressure_mpa, task_name = group_key
                phase_task_model = load_fairchem_model(
                    config.model_name,
                    task_name,
                    device_name,
                    run_name != "minimize",
                )

                atoms_list: list[Atoms] = []
                ts_files: list[Path] = []
                flat_files: list[Path] = []
                thermo_files: list[Path] = []
                rep_dirs: list[Path] = []

                for _item_idx, item in cohort:
                    rep_dir = out_root / item.condition_id / f"replica_{replica:03d}"
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    ts_h5, flat_h5, thermo_csv = _trajectory_paths(rep_dir, run_name)
                    atoms_list.append(start_atoms_by_condition[item.condition_id].copy())
                    ts_files.append(ts_h5)
                    flat_files.append(flat_h5)
                    thermo_files.append(thermo_csv)
                    rep_dirs.append(rep_dir)

                resume = _determine_resume_plan(ts, ts_files, nsteps, device, dtype)
                already_done_steps = max(0, int(nsteps) - int(resume["steps_remaining"]))
                if already_done_steps > 0:
                    global_bar.update(already_done_steps * len(cohort))
                remaining_global = max(0, int(global_bar.total) - int(global_bar.n))
                eta_total = global_tracker.eta(remaining_global)
                global_bar.set_postfix_str(
                    "ETA total: collecting performance..."
                    if eta_total is None
                    else f"ETA total: {_format_duration(eta_total)}"
                )

                if not resume["completed"]:
                    if resume["resume_mode"] and resume["state"] is not None:
                        batch_state = resume["state"]
                    else:
                        states = []
                        for idx, atoms in enumerate(atoms_list):
                            st = ts.initialize_state(atoms, device=device, dtype=dtype)
                            st.rng = int(config.base_seed + replica * 100_000 + idx)
                            states.append(st)
                        batch_state = ts.concatenate_states(states)
                        batch_state.rng = int(config.base_seed + replica)

                    stage_dump_every = max(1, min(int(config.dump_every_steps), int(resume["steps_remaining"])))
                    reporter_state_kwargs = {
                        "save_velocities": run_name != "minimize",
                        "save_forces": run_name != "minimize",
                        "variable_cell": run_name == "retherm",
                        "variable_masses": False,
                        "variable_atomic_numbers": False,
                    }
                    reporter_props = {} if run_name == "minimize" else {stage_dump_every: _build_prop_calculators(ts)[1]}
                    rep = ts.TrajectoryReporter(
                        filenames=[str(p) for p in ts_files],
                        state_frequency=stage_dump_every,
                        prop_calculators=reporter_props,
                        state_kwargs=reporter_state_kwargs,
                        trajectory_kwargs={"mode": resume["reporter_mode"]},
                    )

                    try:
                        if run_name == "minimize":
                            # Use a short low-temperature NVT relaxation as the minimization stage.
                            relax_kwargs: dict[str, Any] = {
                                "system": batch_state,
                                "model": phase_task_model,
                                "integrator": _pick_integrator(ts, "nvt"),
                                "n_steps": int(resume["steps_remaining"]),
                                "temperature": float(config.heat_start_temperature_k),
                                "timestep": float(config.timestep_ps),
                                "trajectory_reporter": rep,
                                "autobatcher": _build_autobatcher(ts, phase_task_model, config, resolved_max_memory_scaler),
                                "pbar": False,
                            }
                            elapsed = _integrate_with_step_eta(
                                ts,
                                relax_kwargs,
                                int(resume["steps_remaining"]),
                                bar_desc=(
                                    f"{phase} rep{replica:03d} minimize "
                                    f"T={config.heat_start_temperature_k:.1f}K batch={len(cohort)}"
                                ),
                                group_size=len(cohort),
                                global_bar=global_bar,
                                global_tracker=global_tracker,
                            )

                        elif run_name == "heat":
                            heat_kwargs: dict[str, Any] = {
                                "system": batch_state,
                                "model": phase_task_model,
                                "integrator": _pick_integrator(ts, "nvt"),
                                "n_steps": int(resume["steps_remaining"]),
                                "temperature": float(config.heat_start_temperature_k),
                                "timestep": float(config.timestep_ps),
                                "trajectory_reporter": rep,
                                "autobatcher": _build_autobatcher(ts, phase_task_model, config, resolved_max_memory_scaler),
                                "pbar": False,
                            }
                            elapsed = _integrate_with_temperature_ramp_eta(
                                ts,
                                heat_kwargs,
                                int(resume["steps_remaining"]),
                                temperature_start_k=float(config.heat_start_temperature_k),
                                temperature_end_k=float(temp_k),
                                bar_desc=(
                                    f"{phase} rep{replica:03d} heat "
                                    f"{config.heat_start_temperature_k:.1f}->{temp_k:.1f}K batch={len(cohort)}"
                                ),
                                group_size=len(cohort),
                                global_bar=global_bar,
                                global_tracker=global_tracker,
                            )

                        else:
                            kwargs: dict[str, Any] = {
                                "system": batch_state,
                                "model": phase_task_model,
                                "integrator": _pick_integrator(ts, "npt" if run_name == "retherm" else "nvt"),
                                "n_steps": int(resume["steps_remaining"]),
                                "temperature": float(temp_k),
                                "timestep": float(config.timestep_ps),
                                "trajectory_reporter": rep,
                                "autobatcher": _build_autobatcher(ts, phase_task_model, config, resolved_max_memory_scaler),
                                "pbar": False,
                            }
                            if run_name == "retherm" and pressure_mpa is not None:
                                kwargs["external_pressure"] = (float(pressure_mpa) / 101325.0) * EV_A3_PER_ATM

                            elapsed = _integrate_with_step_eta(
                                ts,
                                kwargs,
                                int(resume["steps_remaining"]),
                                bar_desc=(
                                    f"{phase} rep{replica:03d} {run_name} "
                                    f"T={temp_k:.1f}K batch={len(cohort)}"
                                ),
                                group_size=len(cohort),
                                global_bar=global_bar,
                                global_tracker=global_tracker,
                            )
                    finally:
                        rep.close()
                else:
                    elapsed = 0.0

                state_fallback_atoms: list[Atoms] | None = None
                have_stage_traj = all(p.exists() for p in ts_files)
                if run_name == "minimize" and not have_stage_traj and batch_state is not None:
                    state_fallback_atoms = _state_to_atoms_list(batch_state, atoms_list)

                for idx, (_item_idx, item) in enumerate(cohort):
                    ts_h5 = ts_files[idx]
                    flat_h5 = flat_files[idx]
                    thermo_csv = thermo_files[idx]
                    if have_stage_traj:
                        pressure_for_export = float(item.pressure_MPa)
                        traj = _load_flat_traj_for_analysis(ts, ts_h5, flat_h5, pressure_for_export, config.timestep_ps)
                        _write_thermo_csv(thermo_csv, traj)
                        _export_thermo_from_torchsim(ts, ts_h5, rep_dirs[idx] / f"{run_name}_thermo_detailed.csv", config.timestep_ps * 1000.0)

                        atoms_next = atoms_list[idx].copy()
                        atoms_next.set_positions(np.asarray(traj["positions"][-1], dtype=float))
                        atoms_next.set_cell(np.asarray(traj["cell"][-1], dtype=float), scale_atoms=False)

                        if run_name == "retherm":
                            eq_start_idx = _select_equilibrated_start_idx(traj["density_g_cm3"])
                            eq_slice = slice(eq_start_idx, None)
                            vol_eq = np.asarray(traj["volume"][eq_slice], dtype=float)
                            den_eq = np.asarray(traj["density_g_cm3"][eq_slice], dtype=float)
                            retherm_summary = {
                                "equilibration_start_index": int(eq_start_idx),
                                "equilibration_start_step": int(traj["step"][eq_start_idx]),
                                "equilibration_start_time_ps": float(traj["time_ps"][eq_start_idx]),
                                "equilibrated_volume_mean_A3": float(np.mean(vol_eq)),
                                "equilibrated_volume_std_A3": float(np.std(vol_eq)),
                                "equilibrated_density_mean_g_cm3": float(np.mean(den_eq)),
                                "equilibrated_density_std_g_cm3": float(np.std(den_eq)),
                            }
                            (rep_dirs[idx] / "retherm_equilibration.json").write_text(json.dumps(retherm_summary, indent=2), encoding="utf-8")
                            _set_atoms_volume_isotropic(atoms_next, float(retherm_summary["equilibrated_volume_mean_A3"]))

                        start_atoms_by_condition[item.condition_id] = atoms_next
                    elif state_fallback_atoms is not None and idx < len(state_fallback_atoms):
                        start_atoms_by_condition[item.condition_id] = state_fallback_atoms[idx]

                    prod_stage_index = production_stage_names.index(run_name) + 1 if run_name in production_stage_names else None
                    metadata = {
                        "condition_id": item.condition_id,
                        "phase": phase,
                        "task_name": item.task_name,
                        "structure_path": str(item.structure_path),
                        "temperature_C": item.temperature_C,
                        "pressure_MPa": item.pressure_MPa,
                        "liOH_M": item.liOH_M,
                        "kOH_M": item.kOH_M,
                        "lithiation_fraction": item.lithiation_fraction,
                        "vacancy_config_id": item.vacancy_config_id,
                        "charge": item.charge,
                        "spin": item.spin,
                        "n_li": item.n_li,
                        "seed": int(config.base_seed + _item_idx * 1000 + replica),
                        "device": device_name,
                        "ensemble_retherm": "npt",
                        "ensemble_production": "nvt",
                        "ensemble_requested": config.ensemble,
                        "minimize_steps": int(config.minimize_steps),
                        "heat_steps": int(config.heat_steps),
                        "heat_start_temperature_k": float(config.heat_start_temperature_k),
                        "stage_steps": int(nsteps),
                        "timestep_ps": config.timestep_ps,
                        "dump_every_steps": config.dump_every_steps,
                        "run_name": run_name,
                        "production_stage_index": prod_stage_index,
                        "production_stage_count": int(len(production_stage_names)),
                        "batch_group_size": len(cohort),
                        "batch_group_temperature_K": float(temp_k),
                        "batch_group_pressure_MPa": pressure_mpa,
                        "resume_used": bool(resume["resume_mode"]),
                        "run_elapsed_s": float(elapsed),
                        "torchsim_trajectory": str(ts_h5) if have_stage_traj else None,
                    }
                    (rep_dirs[idx] / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                _update_run_state(
                    out_root / "run_state.json",
                    phase=phase,
                    replica=replica,
                    run_name=run_name,
                    batch_group_size=len(cohort),
                    steps_completed_runs=int(global_bar.n),
                    system_steps_completed=int(global_bar.n),
                )

        for item in items:
            rep_dir = out_root / item.condition_id / f"replica_{replica:03d}"
            _finalize_production_segments(rep_dir, production_stage_names)

    global_bar.set_postfix_str("completed")
    global_bar.close()


def run_one_phase(phase: str, config: ScreenConfig) -> None:
    items = _load_items_for_phase(config, phase)
    run_md_batches(items, config, phase)
