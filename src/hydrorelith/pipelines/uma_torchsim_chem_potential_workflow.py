from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from hydrorelith.pipelines.uma_torchsim_2pt_experiment import _export_2pt_for_phase
from hydrorelith.pipelines.uma_torchsim_screen_config import ScreenConfig, default_config
from hydrorelith.pipelines.uma_torchsim_screen_run import run_one_phase


DEFAULT_TP_COMBOS = "220:2.02,200:1.32,160:0.46,120:0.08"
SUPPORTED_EXTS = {".cif", ".vasp", ".xyz", ".extxyz", ".pdb", ".bgf", ".data", ".lammps", ".lmp"}
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _format_num_token(value: float, ndigits: int = 2) -> str:
    return f"{value:.{ndigits}f}".replace(".", "p")


def _parse_tp_combos(raw: str) -> list[tuple[float, float]]:
    entries = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not entries:
        raise ValueError("Temperature/pressure combos cannot be empty")
    out: list[tuple[float, float]] = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid T/P entry '{entry}'. Use format 220:2.02")
        t_raw, p_raw = [part.strip() for part in entry.split(":", maxsplit=1)]
        out.append((float(t_raw), float(p_raw)))
    return out


def _steps_from_fs(duration_fs: int, timestep_ps: float) -> int:
    timestep_fs = float(timestep_ps) * 1000.0
    if timestep_fs <= 0:
        raise ValueError("timestep_ps must be > 0")
    steps = int(round(float(duration_fs) / timestep_fs))
    if steps <= 0:
        raise ValueError("Computed step count must be > 0")
    return steps


@dataclass(slots=True)
class PreparedStructure:
    source_path: Path
    prepared_path: Path
    condition_label: str
    lithiation_fraction: float | None
    lioh_m: float | None
    koh_m: float | None


@dataclass(slots=True)
class Py2PTJob:
    condition_id: str
    replica_dir: Path
    replica_index: int
    ini_path: Path
    prefix: str
    thermo_path: Path
    run_row: dict[str, str]


@dataclass(slots=True)
class Py2PTResult:
    condition_id: str
    replica_dir: Path
    status: str
    returncode: int | None
    elapsed_s: float
    thermo_path: Path
    aq_li_kjmol: float | None
    message: str


def _sanitize_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return clean or "structure"


def _extract_lithiation(path: Path) -> float | None:
    token = path.as_posix()
    match = re.search(r"Li(?P<pct>\d+(?:\.\d+)?)pct", token, flags=re.IGNORECASE)
    if match:
        return float(match.group("pct")) / 100.0
    match = re.search(r"lith[_-]?(?P<pct>\d+(?:\.\d+)?)pct", token, flags=re.IGNORECASE)
    if match:
        return float(match.group("pct")) / 100.0
    # Publication electrode inputs include POSCAR_000451-style labels.
    # Interpret the numeric suffix as milli-fractional lithiation (0.451 here).
    match = re.search(r"POSCAR[_-]0*(?P<milli>\d{3,4})(?:\D|$)", path.name, flags=re.IGNORECASE)
    if match:
        value = float(match.group("milli")) / 1000.0
        if 0.0 <= value <= 1.0:
            return value
    return None


def _extract_li_k_conc(path: Path) -> tuple[float | None, float | None]:
    token = path.as_posix()
    li = None
    k = None
    m_li = re.search(r"LiOH[_-]?(\d+(?:[p\.]\d+)?)", token, flags=re.IGNORECASE)
    m_k = re.search(r"KOH[_-]?(\d+(?:[p\.]\d+)?)", token, flags=re.IGNORECASE)
    if m_li:
        li = float(m_li.group(1).replace("p", "."))
    if m_k:
        k = float(m_k.group(1).replace("p", "."))
    return li, k


def _is_lammps_data_path(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in {".data", ".lammps", ".lmp"}:
        return True
    name_lower = path.name.lower()
    if name_lower == "data":
        return True
    # Publication electrolyte files use names like data.LiOH_2p0_KOH_2p0_seed01.
    # Treat these as LAMMPS data unless they carry another known supported extension.
    if name_lower.startswith("data.") and ext not in SUPPORTED_EXTS:
        return True
    return False


def _discover_structure_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.upper().startswith("POSCAR"):
            out.append(path)
            continue
        if _is_lammps_data_path(path):
            out.append(path)
            continue
        if path.suffix.lower() in SUPPORTED_EXTS:
            out.append(path)
    return out


def _read_bgf_with_mda(path: Path) -> Atoms:
    try:
        import MDAnalysis as mda
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "BGF input requires MDAnalysis. Install it in your environment to use .bgf files."
        ) from exc

    u = mda.Universe(str(path))
    ag = u.atoms

    symbols: list[str] = []
    for atom in ag:
        raw = getattr(atom, "element", None) or getattr(atom, "name", "")
        token = re.sub(r"[^A-Za-z]", "", str(raw))
        if not token:
            token = "H"
        if len(token) == 1:
            sym = token.upper()
        else:
            sym = token[0].upper() + token[1].lower()
        symbols.append(sym)

    positions = np.asarray(ag.positions, dtype=float)
    atoms = Atoms(symbols=symbols, positions=positions)

    dims = getattr(u.trajectory.ts, "dimensions", None)
    if dims is not None and len(dims) >= 6:
        a, b, c, alpha, beta, gamma = [float(v) for v in dims[:6]]
        atoms.set_cell([a, b, c, alpha, beta, gamma], scale_atoms=False)
        atoms.set_pbc([True, True, True])

    return atoms


def _read_structure_any(path: Path) -> Atoms:
    name_upper = path.name.upper()
    ext = path.suffix.lower()

    if name_upper.startswith("POSCAR") or ext == ".vasp":
        return ase_read(path, format="vasp")
    if _is_lammps_data_path(path):
        return ase_read(path, format="lammps-data")
    if ext == ".bgf":
        try:
            return ase_read(path)
        except Exception:
            return _read_bgf_with_mda(path)
    return ase_read(path)


def _prepare_structures(system_type: str, input_dir: Path, prepared_root: Path) -> list[PreparedStructure]:
    prepared_root.mkdir(parents=True, exist_ok=True)
    files = _discover_structure_files(input_dir)
    if not files:
        raise ValueError(f"No supported structures found under {input_dir}")

    out: list[PreparedStructure] = []
    for idx, src in enumerate(files):
        atoms = _read_structure_any(src)
        label = _sanitize_label(src.stem)
        prepared_path = prepared_root / f"{idx:05d}_{label}.extxyz"
        ase_write(prepared_path, atoms, format="extxyz")

        lith = _extract_lithiation(src) if system_type == "electrode" else None
        li_m, k_m = _extract_li_k_conc(src) if system_type == "electrolyte" else (None, None)
        out.append(
            PreparedStructure(
                source_path=src,
                prepared_path=prepared_path,
                condition_label=label,
                lithiation_fraction=lith,
                lioh_m=li_m,
                koh_m=k_m,
            )
        )
    return out


def _default_lithiation_for_index(idx: int, n: int) -> float:
    if n <= 1:
        return 1.0
    return max(0.0, min(1.0, idx / float(n - 1)))


def _build_manifest_rows(system_type: str, structures: list[PreparedStructure], tp_combos: list[tuple[float, float]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    n = len(structures)
    for s_idx, item in enumerate(structures):
        for temp_c, pressure_mpa in tp_combos:
            temp_k = temp_c + 273.15
            if system_type == "electrode":
                lith = item.lithiation_fraction
                if lith is None:
                    lith = _default_lithiation_for_index(s_idx, n)
                condition_id = (
                    f"ed_t{int(round(temp_k)):03d}k_"
                    f"p{_format_num_token(pressure_mpa, ndigits=2)}mpa_"
                    f"lith{_format_num_token(lith * 100.0, ndigits=2)}pct_"
                    f"{item.condition_label}"
                )
                row = {
                    "condition_id": condition_id,
                    "structure_path": str(item.prepared_path),
                    "temperature_C": f"{temp_c:.2f}",
                    "pressure_MPa": f"{pressure_mpa:.6g}",
                    "lithiation_fraction": f"{lith:.8f}",
                    "vacancy_config_id": item.condition_label,
                    "phase": "electrode",
                    "task_name": "omat",
                    "n_li": "",
                    "notes": f"source={item.source_path}",
                }
            else:
                li_m = item.lioh_m if item.lioh_m is not None else 0.0
                k_m = item.koh_m if item.koh_m is not None else 0.0
                condition_id = (
                    f"el_t{int(round(temp_k)):03d}k_"
                    f"p{_format_num_token(pressure_mpa, ndigits=2)}mpa_"
                    f"li{_format_num_token(li_m, ndigits=2)}_"
                    f"k{_format_num_token(k_m, ndigits=2)}_"
                    f"{item.condition_label}"
                )
                row = {
                    "condition_id": condition_id,
                    "structure_path": str(item.prepared_path),
                    "temperature_C": f"{temp_c:.2f}",
                    "pressure_MPa": f"{pressure_mpa:.6g}",
                    "liOH_M": f"{li_m:.6g}",
                    "kOH_M": f"{k_m:.6g}",
                    "phase": "electrolyte",
                    "task_name": "omol",
                    "charge": "0",
                    "spin": "1",
                    "notes": f"source={item.source_path}",
                }
            rows.append(row)
    return rows


def _write_manifest(system_type: str, rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if system_type == "electrode":
        fieldnames = [
            "condition_id",
            "structure_path",
            "temperature_C",
            "pressure_MPa",
            "lithiation_fraction",
            "vacancy_config_id",
            "phase",
            "task_name",
            "n_li",
            "notes",
        ]
    else:
        fieldnames = [
            "condition_id",
            "structure_path",
            "temperature_C",
            "pressure_MPa",
            "liOH_M",
            "kOH_M",
            "phase",
            "task_name",
            "charge",
            "spin",
            "notes",
        ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_config(args: argparse.Namespace, system_type: str, manifest_path: Path) -> ScreenConfig:
    cfg = default_config()
    cfg.output_dir = args.output_dir
    cfg.phase = system_type
    cfg.stages = ("md", "export-2pt")
    if system_type == "electrode":
        cfg.electrode_manifest = manifest_path
        cfg.electrolyte_manifest = None
    else:
        cfg.electrolyte_manifest = manifest_path
        cfg.electrode_manifest = None
    cfg.electrode_root = None
    cfg.electrolyte_root = None
    cfg.model_name = args.model_name
    cfg.device = args.device
    cfg.ensemble = "nvt"
    cfg.timestep_ps = args.timestep_ps
    cfg.dump_every_steps = args.dump_every_steps
    cfg.minimize_steps = int(args.minimize_steps)
    cfg.heat_steps = int(args.heat_steps)
    cfg.heat_start_temperature_k = float(args.heat_start_temperature_k)
    cfg.retherm_steps_electrode = int(args.npt_equil_steps)
    cfg.retherm_steps_electrolyte = cfg.retherm_steps_electrode
    cfg.prod_steps = int(args.nvt_prod_steps)
    cfg.production_stages = int(args.production_stages)
    cfg.replicas = args.replicas
    cfg.base_seed = args.base_seed
    cfg.compute_stress = True
    cfg.max_memory_scaler = args.max_memory_scaler
    cfg.precision = args.precision
    cfg.debug = args.debug
    return cfg


def _atom_ranges(atom_ids_1based: list[int]) -> str:
    values = sorted(set(atom_ids_1based))
    if not values:
        return ""
    ranges: list[tuple[int, int]] = []
    start = values[0]
    prev = values[0]
    for val in values[1:]:
        if val == prev + 1:
            prev = val
            continue
        ranges.append((start, prev))
        start = val
        prev = val
    ranges.append((start, prev))

    out: list[str] = []
    for left, right in ranges:
        if left == right:
            out.append(str(left))
        else:
            out.append(f"{left} - {right}")
    return " ".join(out)


def _write_py2pt_group_file(h5md_path: Path, out_grps: Path) -> bool:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("py2pt TorchSim flow requires h5py") from exc

    with h5py.File(h5md_path, "r") as h5:
        if "atomic_numbers" not in h5:
            raise ValueError(f"{h5md_path}: missing atomic_numbers")
        z = np.asarray(h5["atomic_numbers"], dtype=int).reshape(-1)

    li_ids = [int(i + 1) for i in np.where(z == 3)[0].tolist()]
    if not li_ids:
        return False
    other_ids = [int(i + 1) for i in np.where(z != 3)[0].tolist()]

    lines = [
        "# Auto-generated TorchSim py2pt groups",
        "[group1]",
        f"atoms = {_atom_ranges(li_ids)}",
        "constraints = 0",
        "rotsym = 1",
        "linear = 0",
        "",
    ]
    if other_ids:
        lines.extend(
            [
                "[group2]",
                f"atoms = {_atom_ranges(other_ids)}",
                "constraints = 0",
                "rotsym = 1",
                "linear = 0",
                "",
            ]
        )
    out_grps.parent.mkdir(parents=True, exist_ok=True)
    out_grps.write_text("\n".join(lines), encoding="utf-8")
    return True


def _write_py2pt_ini(
    out_ini: Path,
    *,
    trajectory_name: str,
    group_name: str,
    timestep_ps: float,
    prefix: str,
    mode: int = 1,
    topology_path: str | None = None,
    topology_format: str | None = None,
) -> None:
    files_block = [
        "[files]",
        f"trajectory = {trajectory_name}",
        "trajectory_format = TORCHSIM",
        "thermo_file = prod_thermo.csv",
        f"group_file = {group_name}",
    ]
    if topology_path:
        files_block.insert(1, f"topology = {topology_path}")
        if topology_format:
            files_block.insert(2, f"topology_format = {topology_format}")

    text = (
        "\n".join(files_block)
        + "\n\n"
        + "[frames]\n"
        + "start = 1\n"
        + "stop = 0\n"
        + "step = 1\n\n"
        + "[analysis]\n"
        + "corlen = 0.5\n"
        + f"mode = {int(mode)}\n"
        + "vel_scale = 1.0\n"
        + "lammps_units = metal\n"
        + "check_grp_eng = false\n\n"
        + "[thermodynamics]\n"
        + f"timestep = {timestep_ps:.12g}\n\n"
        + "[output]\n"
        + f"prefix = {prefix}\n"
        + "show_2pt_split = false\n"
        + "normalize = true\n"
        + "out_units = kj/mol\n"
    )
    out_ini.parent.mkdir(parents=True, exist_ok=True)
    out_ini.write_text(text, encoding="utf-8")


def _infer_topology_for_py2pt(structure_path: str | None) -> tuple[str | None, str | None]:
    if not structure_path:
        return (None, None)

    path = Path(structure_path).expanduser()
    if not path.exists():
        return (None, None)

    suffix = path.suffix.lower()
    fmt = {
        ".data": "LAMMPS",
        ".lmpdata": "LAMMPS",
        ".bgf": "BGF",
        ".pdb": "PDB",
        ".gro": "GRO",
        ".psf": "PSF",
        ".prmtop": "AMBER",
        ".parm7": "AMBER",
    }.get(suffix)
    if fmt is None:
        return (None, None)
    return (str(path), fmt)


def _extract_aq_group_total(thermo_path: Path, group_idx: int) -> float:
    legacy_target_pos = 4 * (group_idx - 1) + 3
    with thermo_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.lstrip().startswith("A_q"):
                continue
            values = [float(tok) for tok in FLOAT_RE.findall(line)]
            if legacy_target_pos < len(values):
                return float(values[legacy_target_pos])
            direct_idx = group_idx - 1
            if direct_idx < len(values):
                return float(values[direct_idx])
            raise ValueError(
                f"{thermo_path}: A_q row has {len(values)} numbers, cannot index group {group_idx}"
            )
    raise ValueError(f"{thermo_path}: A_q row not found")


def _run_single_py2pt(job: Py2PTJob, py2pt_command: str) -> Py2PTResult:
    import time

    t0 = time.perf_counter()
    cmd = [*shlex.split(py2pt_command), job.ini_path.name]
    logs_dir = job.replica_dir / "2pt_output" / "launcher_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / f"{job.ini_path.stem}.stdout.log"
    stderr_log = logs_dir / f"{job.ini_path.stem}.stderr.log"

    proc = subprocess.run(cmd, cwd=job.replica_dir, capture_output=True, text=True, check=False)
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")

    aq_val: float | None = None
    status = "failed"
    msg = ""
    if proc.returncode == 0 and job.thermo_path.exists():
        try:
            aq_val = _extract_aq_group_total(job.thermo_path, group_idx=1)
            status = "success"
        except Exception as exc:  # noqa: BLE001
            status = "parse_failed"
            msg = str(exc)
    else:
        msg = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        msg = msg[-1] if msg else "py2pt failed"

    elapsed = time.perf_counter() - t0
    return Py2PTResult(
        condition_id=job.condition_id,
        replica_dir=job.replica_dir,
        status=status,
        returncode=int(proc.returncode),
        elapsed_s=float(elapsed),
        thermo_path=job.thermo_path,
        aq_li_kjmol=aq_val,
        message=msg,
    )


def _prepare_py2pt_jobs(
    phase_root: Path,
    manifest_rows: list[dict[str, str]],
    timestep_ps: float,
    system_type: str,
) -> tuple[list[Py2PTJob], list[dict[str, str]]]:
    row_by_condition = {row["condition_id"]: row for row in manifest_rows}
    jobs: list[Py2PTJob] = []
    skipped: list[dict[str, str]] = []

    for condition_id, row in sorted(row_by_condition.items()):
        cond_dir = phase_root / condition_id
        if not cond_dir.exists():
            skipped.append({"condition_id": condition_id, "status": "missing_condition_dir"})
            continue
        rep_dirs = sorted(path for path in cond_dir.iterdir() if path.is_dir() and path.name.startswith("replica_"))
        for rep_dir in rep_dirs:
            h5md = rep_dir / "prod.h5md"
            thermo_csv = rep_dir / "prod_thermo.csv"
            if not (h5md.exists() and thermo_csv.exists()):
                skipped.append(
                    {
                        "condition_id": condition_id,
                        "replica": rep_dir.name,
                        "status": "missing_prod_files",
                    }
                )
                continue

            rep_num = int(rep_dir.name.split("_")[-1]) + 1
            stem = f"lammps_2pt_py2pt.{rep_num}"
            grps = rep_dir / f"{stem}.grps"
            has_li = _write_py2pt_group_file(h5md, grps)
            if not has_li:
                skipped.append(
                    {
                        "condition_id": condition_id,
                        "replica": rep_dir.name,
                        "status": "no_li_atoms",
                    }
                )
                continue

            ini = rep_dir / f"{stem}.ini"
            prefix = f"2pt_output/{stem}"
            topology_path, topology_format = _infer_topology_for_py2pt(row.get("structure_path"))
            use_molecular_mode = system_type == "electrolyte" and topology_path is not None
            _write_py2pt_ini(
                ini,
                trajectory_name=h5md.name,
                group_name=grps.name,
                timestep_ps=timestep_ps,
                prefix=prefix,
                mode=4 if use_molecular_mode else 1,
                topology_path=topology_path,
                topology_format=topology_format,
            )
            jobs.append(
                Py2PTJob(
                    condition_id=condition_id,
                    replica_dir=rep_dir,
                    replica_index=rep_num,
                    ini_path=ini,
                    prefix=prefix,
                    thermo_path=rep_dir / f"{prefix}.thermo",
                    run_row=row,
                )
            )

    return jobs, skipped


def _run_py2pt_phase(args: argparse.Namespace, manifest_rows: list[dict[str, str]]) -> list[Py2PTResult]:
    phase_root = args.output_dir / "uma_torchsim_screen" / args.system_type
    jobs, skipped = _prepare_py2pt_jobs(phase_root, manifest_rows, args.timestep_ps, args.system_type)

    results: list[Py2PTResult] = []
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.py2pt_workers)) as pool:
            futures = [pool.submit(_run_single_py2pt, job, args.py2pt_command) for job in jobs]
            for fut in as_completed(futures):
                results.append(fut.result())

    detail_csv = args.output_dir / "uma_torchsim_screen" / "py2pt_batch_detail.csv"
    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    with detail_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition_id",
                "replica_dir",
                "status",
                "returncode",
                "elapsed_s",
                "thermo_path",
                "aq_li_kjmol",
                "message",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "condition_id": row.condition_id,
                    "replica_dir": str(row.replica_dir),
                    "status": row.status,
                    "returncode": "" if row.returncode is None else row.returncode,
                    "elapsed_s": f"{row.elapsed_s:.6f}",
                    "thermo_path": str(row.thermo_path),
                    "aq_li_kjmol": "" if row.aq_li_kjmol is None else f"{row.aq_li_kjmol:.10g}",
                    "message": row.message,
                }
            )
        for row in skipped:
            writer.writerow(
                {
                    "condition_id": row.get("condition_id", ""),
                    "replica_dir": row.get("replica", ""),
                    "status": row.get("status", ""),
                    "returncode": "",
                    "elapsed_s": "0",
                    "thermo_path": "",
                    "aq_li_kjmol": "",
                    "message": "",
                }
            )

    return sorted(results, key=lambda r: (r.condition_id, r.replica_dir.name))


def _safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    if not np.isfinite(val):
        return None
    return val


def _load_thermo_csv(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {
        "time_ps": [],
        "temperature_K": [],
        "pressure_MPa": [],
        "density_g_cm3": [],
        "total_energy_eV": [],
    }
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_ps = _safe_float(row.get("time_ps"))
            if time_ps is None:
                continue
            pe = _safe_float(row.get("potential_energy_eV")) or 0.0
            ke = _safe_float(row.get("kinetic_energy_eV")) or 0.0
            out["time_ps"].append(time_ps)
            out["temperature_K"].append(_safe_float(row.get("temperature_K")) or math.nan)
            out["pressure_MPa"].append(_safe_float(row.get("pressure_MPa")) or math.nan)
            out["density_g_cm3"].append(_safe_float(row.get("density_g_cm3")) or math.nan)
            out["total_energy_eV"].append(pe + ke)

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _tail_stats(arr: np.ndarray, frac: float = 0.10) -> tuple[float, float, int]:
    if arr.size == 0:
        return (math.nan, math.nan, 0)
    n = max(5, int(round(frac * arr.size)))
    n = min(n, arr.size)
    tail = arr[-n:]
    mean = float(np.nanmean(tail))
    std = float(np.nanstd(tail, ddof=0))
    return mean, std, int(n)


def _tp_label(row: dict[str, str]) -> str:
    t = float(row.get("temperature_C", "0"))
    p = float(row.get("pressure_MPa", "0"))
    return f"T={t:.0f}C, P={p:.2f} MPa"


def _x_value(system_type: str, row: dict[str, str]) -> float:
    if system_type == "electrode":
        return float(row.get("lithiation_fraction", "0")) * 100.0
    return float(row.get("liOH_M", "0"))


def _x_label(system_type: str) -> str:
    return "Lithiation (%)" if system_type == "electrode" else "LiOH concentration (M)"


def _final_stats_name(system_type: str) -> str:
    return "master_final_stats_vs_lithiation.png" if system_type == "electrode" else "master_final_stats_vs_concentration.png"


def _li_mu_name(system_type: str) -> str:
    return "li_chemical_potential_vs_lithiation.png" if system_type == "electrode" else "li_chemical_potential_vs_concentration.png"


def _plot_master_thermo(
    system_type: str,
    phase_root: Path,
    manifest_rows: list[dict[str, str]],
    out_png: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, np.ndarray]]]:
    condition_series: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, str]] = []

    for row in manifest_rows:
        condition_id = row["condition_id"]
        cond_dir = phase_root / condition_id
        rep_dirs = sorted(path for path in cond_dir.iterdir() if path.is_dir() and path.name.startswith("replica_")) if cond_dir.exists() else []
        if not rep_dirs:
            continue

        rep_metrics: dict[str, list[np.ndarray]] = {
            "time_ps": [],
            "temperature_K": [],
            "pressure_MPa": [],
            "density_g_cm3": [],
            "total_energy_eV": [],
        }
        final_metric_values: dict[str, list[float]] = {
            "temperature_K": [],
            "pressure_MPa": [],
            "density_g_cm3": [],
            "total_energy_eV": [],
        }

        for rep_dir in rep_dirs:
            retherm = rep_dir / "retherm_thermo.csv"
            prod = rep_dir / "prod_thermo.csv"
            if not (retherm.exists() and prod.exists()):
                continue

            r = _load_thermo_csv(retherm)
            p = _load_thermo_csv(prod)

            if r["time_ps"].size == 0 or p["time_ps"].size == 0:
                continue

            prod_shift = r["time_ps"][-1] + (p["time_ps"][1] - p["time_ps"][0] if p["time_ps"].size > 1 else 0.0)
            merged_time = np.concatenate([r["time_ps"], p["time_ps"] + prod_shift])
            merged_temp = np.concatenate([r["temperature_K"], p["temperature_K"]])
            merged_press = np.concatenate([r["pressure_MPa"], p["pressure_MPa"]])
            merged_density = np.concatenate([r["density_g_cm3"], p["density_g_cm3"]])
            merged_energy = np.concatenate([r["total_energy_eV"], p["total_energy_eV"]])

            rep_metrics["time_ps"].append(merged_time)
            rep_metrics["temperature_K"].append(merged_temp)
            rep_metrics["pressure_MPa"].append(merged_press)
            rep_metrics["density_g_cm3"].append(merged_density)
            rep_metrics["total_energy_eV"].append(merged_energy)

            for metric in final_metric_values:
                mean, _std, _n = _tail_stats(p[metric], frac=0.10)
                if np.isfinite(mean):
                    final_metric_values[metric].append(mean)

        if not rep_metrics["time_ps"]:
            continue

        base_len = min(arr.size for arr in rep_metrics["time_ps"])
        trunc = {
            key: np.vstack([arr[:base_len] for arr in arrs])
            for key, arrs in rep_metrics.items()
            if key != "time_ps"
        }
        mean_time = rep_metrics["time_ps"][0][:base_len]
        condition_series[condition_id] = {
            "time_ps": mean_time,
            "temperature_K": np.nanmean(trunc["temperature_K"], axis=0),
            "pressure_MPa": np.nanmean(trunc["pressure_MPa"], axis=0),
            "density_g_cm3": np.nanmean(trunc["density_g_cm3"], axis=0),
            "total_energy_eV": np.nanmean(trunc["total_energy_eV"], axis=0),
        }

        summary_rows.append(
            {
                "condition_id": condition_id,
                "temperature_C": row.get("temperature_C", ""),
                "pressure_MPa": row.get("pressure_MPa", ""),
                "x_value": f"{_x_value(system_type, row):.10g}",
                "temp_prod_tail_mean": f"{np.nanmean(final_metric_values['temperature_K']):.10g}" if final_metric_values["temperature_K"] else "",
                "press_prod_tail_mean": f"{np.nanmean(final_metric_values['pressure_MPa']):.10g}" if final_metric_values["pressure_MPa"] else "",
                "density_prod_tail_mean": f"{np.nanmean(final_metric_values['density_g_cm3']):.10g}" if final_metric_values["density_g_cm3"] else "",
                "etotal_prod_tail_mean": f"{np.nanmean(final_metric_values['total_energy_eV']):.10g}" if final_metric_values["total_energy_eV"] else "",
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panel = [
        ("temperature_K", "Temperature (K)"),
        ("pressure_MPa", "Pressure (MPa)"),
        ("density_g_cm3", "Density (g/cm^3)"),
        ("total_energy_eV", "Total energy (eV)"),
    ]
    flat_axes = axes.flatten()

    for ax, (metric, ylabel) in zip(flat_axes, panel):
        for condition_id, series in sorted(condition_series.items()):
            ax.plot(series["time_ps"], series[metric], linewidth=1.1, alpha=0.85, label=condition_id)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    flat_axes[-1].set_xlabel("Time (ps)")
    flat_axes[-2].set_xlabel("Time (ps)")
    if condition_series:
        flat_axes[0].legend(fontsize=6, ncol=2, framealpha=0.85)
    fig.suptitle(f"{system_type.capitalize()} thermo evolution (retherm + production)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return summary_rows, condition_series


def _plot_final_stats(
    system_type: str,
    manifest_rows: list[dict[str, str]],
    thermo_summary_rows: list[dict[str, str]],
    out_png: Path,
) -> None:
    row_by_condition = {row["condition_id"]: row for row in manifest_rows}

    grouped: dict[str, list[tuple[float, dict[str, str]]]] = {}
    for row in thermo_summary_rows:
        cond = row["condition_id"]
        manifest_row = row_by_condition.get(cond)
        if manifest_row is None:
            continue
        tp = _tp_label(manifest_row)
        grouped.setdefault(tp, []).append((_x_value(system_type, manifest_row), row))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    metric_map = [
        ("temp_prod_tail_mean", "Temperature (K)"),
        ("press_prod_tail_mean", "Pressure (MPa)"),
        ("density_prod_tail_mean", "Density (g/cm^3)"),
        ("etotal_prod_tail_mean", "Total energy (eV)"),
    ]

    for ax, (key, ylabel) in zip(axes.flatten(), metric_map):
        for tp, pairs in sorted(grouped.items()):
            pairs = sorted(pairs, key=lambda item: item[0])
            xs = np.asarray([item[0] for item in pairs], dtype=float)
            ys = np.asarray([float(item[1][key]) for item in pairs], dtype=float)
            ax.plot(xs, ys, marker="o", linewidth=1.2, label=tp)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[-1, 0].set_xlabel(_x_label(system_type))
    axes[-1, 1].set_xlabel(_x_label(system_type))
    if grouped:
        axes[0, 0].legend(fontsize=7, ncol=2, framealpha=0.85)
    fig.suptitle(f"{system_type.capitalize()} production-tail stats")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_li_chemical_potential(
    system_type: str,
    manifest_rows: list[dict[str, str]],
    py2pt_results: list[Py2PTResult],
    out_png: Path,
    out_csv: Path,
) -> None:
    row_by_condition = {row["condition_id"]: row for row in manifest_rows}
    grouped_values: dict[str, list[float]] = {}
    for res in py2pt_results:
        if res.status != "success" or res.aq_li_kjmol is None:
            continue
        grouped_values.setdefault(res.condition_id, []).append(float(res.aq_li_kjmol))

    summary_rows: list[dict[str, str]] = []
    series_by_tp: dict[str, list[tuple[float, float, float]]] = {}
    for condition_id, vals in sorted(grouped_values.items()):
        manifest_row = row_by_condition.get(condition_id)
        if manifest_row is None:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        x_val = _x_value(system_type, manifest_row)
        tp = _tp_label(manifest_row)
        series_by_tp.setdefault(tp, []).append((x_val, mean, std))

        summary_rows.append(
            {
                "condition_id": condition_id,
                "temperature_C": manifest_row.get("temperature_C", ""),
                "pressure_MPa": manifest_row.get("pressure_MPa", ""),
                "x_value": f"{x_val:.10g}",
                "n_values": str(arr.size),
                "mu_li_kjmol_mean": f"{mean:.10g}",
                "mu_li_kjmol_std": f"{std:.10g}",
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition_id",
                "temperature_C",
                "pressure_MPa",
                "x_value",
                "n_values",
                "mu_li_kjmol_mean",
                "mu_li_kjmol_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tp, points in sorted(series_by_tp.items()):
        points = sorted(points, key=lambda item: item[0])
        xs = np.asarray([p[0] for p in points], dtype=float)
        ys = np.asarray([p[1] for p in points], dtype=float)
        yerr = np.asarray([p[2] for p in points], dtype=float)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.2, capsize=3, label=tp)

    ax.set_xlabel(_x_label(system_type))
    ax.set_ylabel("Li chemical potential from 2PT (kJ/mol)")
    ax.grid(alpha=0.25)
    if series_by_tp:
        ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_thermo_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition_id",
                "temperature_C",
                "pressure_MPa",
                "x_value",
                "temp_prod_tail_mean",
                "press_prod_tail_mean",
                "density_prod_tail_mean",
                "etotal_prod_tail_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _generate_slurm_scripts(args: argparse.Namespace, manifest_path: Path, manifest_rows: list[dict[str, str]]) -> Path:
    slurm_root = args.output_dir / "uma_torchsim_screen" / "slurm"
    manifests_dir = slurm_root / "manifests"
    scripts_dir = slurm_root / "scripts"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    phase_manifest_arg = "--electrode-manifest" if args.system_type == "electrode" else "--electrolyte-manifest"

    for row in manifest_rows:
        one_manifest = manifests_dir / f"{row['condition_id']}.csv"
        _write_manifest(args.system_type, [row], one_manifest)

        script = scripts_dir / f"run_md_{row['condition_id']}.sbatch"
        content = (
            "#!/bin/bash\n"
            f"#SBATCH -J uma_{args.system_type[:2]}_{row['condition_id'][:24]}\n"
            f"#SBATCH -A {args.slurm_account}\n"
            f"#SBATCH -q {args.slurm_queue}\n"
            f"#SBATCH -t {args.slurm_time}\n"
            "#SBATCH -C gpu\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --gpus-per-node=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --cpus-per-task=32\n\n"
            "set -euo pipefail\n\n"
            "# source /path/to/venv/bin/activate\n\n"
            "hrw-uma-torchsim-screen "
            f"--phase {args.system_type} "
            "--stage md,export-2pt "
            f"{phase_manifest_arg} {one_manifest} "
            f"--output-dir {args.output_dir} "
            f"--model-name {args.model_name} "
            f"--device {args.device} "
            f"--precision {args.precision} "
            f"--replicas {args.replicas} "
            f"--timestep-ps {args.timestep_ps} "
            f"--dump-every-steps {args.dump_every_steps} "
            f"--minimize-steps {args.minimize_steps} "
            f"--heat-steps {args.heat_steps} "
            f"--heat-start-temperature-k {args.heat_start_temperature_k} "
            f"--retherm-steps-electrode {args.npt_equil_steps} "
            f"--retherm-steps-electrolyte {args.npt_equil_steps} "
            f"--prod-steps {args.nvt_prod_steps} "
            f"--production-stages {args.production_stages} "
            f"{f'--max-memory-scaler {args.max_memory_scaler}' if args.max_memory_scaler is not None else ''}\n"
        )
        script.write_text(content, encoding="utf-8")

    py2pt_script = scripts_dir / "run_py2pt_all.sbatch"
    py2pt_script.write_text(
        (
            "#!/bin/bash\n"
            "#SBATCH -J uma_py2pt\n"
            f"#SBATCH -A {args.slurm_account}\n"
            f"#SBATCH -q {args.slurm_queue}\n"
            f"#SBATCH -t {args.slurm_time}\n"
            "#SBATCH -C gpu\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --gpus-per-node=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --cpus-per-task=16\n\n"
            "set -euo pipefail\n\n"
            "hrw-uma-torchsim-chem-potential "
            f"--system-type {args.system_type} "
            f"--output-dir {args.output_dir} "
            f"--manifest-path {manifest_path} "
            f"--skip-md --skip-plots "
            f"--py2pt-command \"{args.py2pt_command}\" "
            f"--py2pt-workers {args.py2pt_workers}\n"
        ),
        encoding="utf-8",
    )

    plots_script = scripts_dir / "run_plots_all.sbatch"
    plots_script.write_text(
        (
            "#!/bin/bash\n"
            "#SBATCH -J uma_plots\n"
            f"#SBATCH -A {args.slurm_account}\n"
            f"#SBATCH -q {args.slurm_queue}\n"
            f"#SBATCH -t {args.slurm_time}\n"
            "#SBATCH -C gpu\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --gpus-per-node=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --cpus-per-task=16\n\n"
            "set -euo pipefail\n\n"
            "hrw-uma-torchsim-chem-potential "
            f"--system-type {args.system_type} "
            f"--output-dir {args.output_dir} "
            f"--manifest-path {manifest_path} "
            f"--skip-md --skip-py2pt\n"
        ),
        encoding="utf-8",
    )

    return slurm_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-uma-torchsim-chem-potential",
        description=(
            "Run TorchSim UMA MD + py2pt workflow to estimate Li chemical potential for "
            "electrode or electrolyte systems."
        ),
    )
    parser.add_argument("--system-type", choices=["electrode", "electrolyte"], required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=["local", "slurm"], default="local")
    parser.add_argument(
        "--tp-combos",
        type=str,
        default=DEFAULT_TP_COMBOS,
        help="Comma-separated temperature_C:pressure_MPa list (e.g. 220:2.02,200:1.32)",
    )

    parser.add_argument("--model-name", type=str, default="uma-s-1p2")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")

    parser.add_argument("--minimize-steps", type=int, default=2000)
    parser.add_argument("--heat-steps", type=int, default=10000)
    parser.add_argument("--heat-start-temperature-k", type=float, default=1.0)
    parser.add_argument("--npt-equil-steps", type=int, default=100000)
    parser.add_argument("--nvt-prod-steps", type=int, default=100000)
    parser.add_argument("--production-stages", type=int, default=15)
    parser.add_argument("--timestep-ps", type=float, default=0.001)
    parser.add_argument("--dump-every-steps", type=int, default=2)
    parser.add_argument("--replicas", type=int, default=15)
    parser.add_argument("--base-seed", type=int, default=0)

    parser.add_argument("--max-memory-scaler", type=float, default=None)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--skip-md", action="store_true")
    parser.add_argument("--skip-py2pt", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")

    parser.add_argument("--py2pt-command", type=str, default="py2pt")
    parser.add_argument("--py2pt-workers", type=int, default=8)

    parser.add_argument("--slurm-account", type=str, default="<account>")
    parser.add_argument("--slurm-queue", type=str, default="regular")
    parser.add_argument("--slurm-time", type=str, default="04:00:00")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.resolve()

    run_root = args.output_dir / "uma_torchsim_screen"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.manifest_path is not None:
        manifest_path = args.manifest_path.resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        manifest_rows = _read_manifest_rows(manifest_path)
    else:
        if args.input_dir is None:
            raise ValueError("Provide --input-dir when --manifest-path is not set")
        tp_combos = _parse_tp_combos(args.tp_combos)
        prepared_root = run_root / "prepared_structures" / args.system_type
        prepared = _prepare_structures(args.system_type, args.input_dir.resolve(), prepared_root)
        manifest_rows = _build_manifest_rows(args.system_type, prepared, tp_combos)
        manifest_path = run_root / f"{args.system_type}_manifest.workflow.csv"
        _write_manifest(args.system_type, manifest_rows, manifest_path)

    if args.execution_mode == "slurm":
        slurm_root = _generate_slurm_scripts(args, manifest_path, manifest_rows)
        summary = {
            "execution_mode": "slurm",
            "system_type": args.system_type,
            "manifest_path": str(manifest_path),
            "slurm_root": str(slurm_root),
            "n_conditions": len(manifest_rows),
        }
        (run_root / "chem_potential_workflow_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return

    if not args.skip_md:
        cfg = _build_config(args, args.system_type, manifest_path)
        run_one_phase(args.system_type, cfg)
        _export_2pt_for_phase(cfg, args.system_type)

    py2pt_results: list[Py2PTResult] = []
    if not args.skip_py2pt:
        py2pt_results = _run_py2pt_phase(args, manifest_rows)

    if not args.skip_plots:
        phase_root = run_root / args.system_type
        thermo_summary_rows, _series = _plot_master_thermo(
            args.system_type,
            phase_root,
            manifest_rows,
            run_root / "master_thermo_evolution_all_runs.png",
        )
        _write_thermo_summary_csv(run_root / "thermo_summary_all_runs.csv", thermo_summary_rows)

        _plot_final_stats(
            args.system_type,
            manifest_rows,
            thermo_summary_rows,
            run_root / _final_stats_name(args.system_type),
        )

        _plot_li_chemical_potential(
            args.system_type,
            manifest_rows,
            py2pt_results,
            run_root / _li_mu_name(args.system_type),
            run_root / "li_chemical_potential_summary.csv",
        )

    summary = {
        "execution_mode": "local",
        "system_type": args.system_type,
        "manifest_path": str(manifest_path),
        "output_dir": str(args.output_dir),
        "n_conditions": len(manifest_rows),
        "skip_md": bool(args.skip_md),
        "skip_py2pt": bool(args.skip_py2pt),
        "skip_plots": bool(args.skip_plots),
    }
    (run_root / "chem_potential_workflow_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
