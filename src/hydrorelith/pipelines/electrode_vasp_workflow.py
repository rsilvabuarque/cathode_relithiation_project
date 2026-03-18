from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPStaticSet


@dataclass(slots=True)
class CaseContext:
    case_id: str
    case_dir: Path
    source_structure: Path
    temperature_k: int | None
    pressure_mpa: float | None
    lithiation_pct: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-electrode-vasp-workflow",
        description="Prepare, submit, monitor, and analyze VASP SCF jobs for DIRECT-filtered electrode structures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-inputs", help="Prepare VASP case directories from best_training_set structures.")
    prepare.add_argument("--structures-root", type=Path, required=True, help="Path to best_training_set or parent structure_generation directory.")
    prepare.add_argument("--output-dir", type=Path, required=True, help="Root where VASP case folders are written.")
    prepare.add_argument("--template-dir", type=Path, default=None, help="Optional directory containing INCAR/KPOINTS/POTCAR/run_vasp.slurm templates.")
    prepare.add_argument("--max-cases", type=int, default=None, help="Optional cap on number of structures to prepare.")
    prepare.add_argument("--temperatures", type=int, nargs="*", default=[393, 433, 473, 493])
    prepare.add_argument("--pressures-mpa", type=float, nargs="*", default=[0.08, 0.46, 1.32, 2.02])
    prepare.add_argument("--potcar-spec", action="store_true", help="When using MPStaticSet without templates, write POTCAR.spec instead of POTCAR.")
    prepare.add_argument("--perlmutter-account", type=str, default="m4537")
    prepare.add_argument("--perlmutter-queue", type=str, default="regular")
    prepare.add_argument("--slurm-time", type=str, default="4:00:00")
    prepare.add_argument("--slurm-nodes", type=int, default=1)
    prepare.add_argument("--slurm-ntasks-per-node", type=int, default=128)
    prepare.add_argument("--slurm-cpus-per-task", type=int, default=2)
    prepare.add_argument("--slurm-gpus", type=int, default=0)
    prepare.add_argument("--vasp-module", type=str, default="vasp/6.4.3-cpu")
    prepare.add_argument("--vasp-exe", type=str, default="vasp_std")
    prepare.add_argument("--job-name-prefix", type=str, default="vasp_scf")

    submit = subparsers.add_parser("submit", help="Submit pending/fizzled VASP jobs in prepared case directories.")
    submit.add_argument("--cases-root", type=Path, required=True, help="Path to workflow output root or directly to cases/.")
    submit.add_argument("--user", type=str, default=None)
    submit.add_argument("--resubmit-fizzled", action="store_true")
    submit.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status", help="Report completed/running/fizzled/pending VASP jobs.")
    status.add_argument("--cases-root", type=Path, required=True)
    status.add_argument("--user", type=str, default=None)
    status.add_argument("--output-json", type=Path, default=None)
    status.add_argument(
        "--list-states",
        nargs="+",
        choices=["completed", "running", "fizzled", "pending"],
        default=[],
        help="Print case directories for the requested states.",
    )

    analyze = subparsers.add_parser("analyze-uma-vs-vasp", help="Batch UMA-vs-VASP comparison and lithiation-dependent plots.")
    analyze.add_argument("--cases-root", type=Path, required=True)
    analyze.add_argument("--analysis-dir", type=Path, default=None)
    analyze.add_argument("--model", type=str, default="uma-s-1p2")
    analyze.add_argument("--task-name", type=str, default="omat")
    analyze.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    return parser


def _resolve_best_training_set_root(path: Path) -> Path:
    if (path / "best_training_set").exists():
        return path / "best_training_set"
    if (path / "structure_generation" / "best_training_set").exists():
        return path / "structure_generation" / "best_training_set"
    if path.name == "best_training_set" and path.exists():
        return path
    raise FileNotFoundError(
        f"Could not find best_training_set under '{path}'. Expected best_training_set/ or structure_generation/best_training_set/."
    )


def _discover_structure_files(root: Path) -> list[Path]:
    files = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and (p.name.startswith("POSCAR") or p.suffix.lower() == ".cif")
        ]
    )
    if not files:
        raise FileNotFoundError(f"No structure files found in {root}")
    return files


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _parse_temperature_lithiation(path: Path) -> tuple[int | None, float | None]:
    temperature = None
    lithiation = None
    for token in path.parts:
        t_match = re.match(r"T_(\d+)K$", token)
        if t_match:
            temperature = int(t_match.group(1))
        l_match = re.match(r"lith_([0-9]*\.?[0-9]+)pct$", token)
        if l_match:
            lithiation = float(l_match.group(1))
    return temperature, lithiation


def _pressure_map(temperatures: list[int], pressures: list[float]) -> dict[int, float]:
    if len(temperatures) != len(pressures):
        raise ValueError("--pressures-mpa must have same length as --temperatures")
    return {int(t): float(p) for t, p in zip(temperatures, pressures)}


def _replace_composition_tokens(template_text: str, structure: Structure) -> str:
    counts = structure.composition.get_el_amt_dict()
    text = template_text
    for el, value in counts.items():
        token = f"N_{el}"
        text = re.sub(rf"\b{re.escape(token)}\b", str(int(round(float(value)))), text)
    return text


def _build_perlmutter_slurm(
    account: str,
    queue: str,
    walltime: str,
    nodes: int,
    ntasks_per_node: int,
    cpus_per_task: int,
    gpus: int,
    module: str,
    vasp_exe: str,
    job_name: str,
) -> str:
    constraint = "gpu" if gpus > 0 else "cpu"
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH -A {account}",
        f"#SBATCH -C {constraint}",
        f"#SBATCH -q {queue}",
        f"#SBATCH -N {nodes}",
        f"#SBATCH --ntasks-per-node={ntasks_per_node}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH -t {walltime}",
        f"#SBATCH -J {job_name}",
        "#SBATCH -o %x-%j.out",
        "#SBATCH -e %x-%j.err",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gpus {gpus}")

    lines.extend(
        [
            "",
            f"module load {module}",
            "which " + vasp_exe,
            "",
            "export OMP_NUM_THREADS=1",
            "export OMP_PLACES=threads",
            "export OMP_PROC_BIND=spread",
            "",
            f"srun -n {ntasks_per_node} -c {cpus_per_task} --cpu-bind=cores {vasp_exe}",
            "",
        ]
    )
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_case_manifest(case: CaseContext, manifest_path: Path, extra: dict[str, object]) -> None:
    payload = {
        "case_id": case.case_id,
        "case_dir": str(case.case_dir),
        "source_structure": str(case.source_structure),
        "temperature_k": case.temperature_k,
        "pressure_mpa": case.pressure_mpa,
        "lithiation_pct": case.lithiation_pct,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_case(
    structure_path: Path,
    best_root: Path,
    cases_root: Path,
    pressure_lookup: dict[int, float],
    args: argparse.Namespace,
) -> CaseContext:
    rel = structure_path.relative_to(best_root)
    temperature_k, lithiation_pct = _parse_temperature_lithiation(rel)
    pressure_mpa = pressure_lookup.get(temperature_k) if temperature_k is not None else None

    case_id = _safe_id("__".join(rel.parts))
    case_dir = cases_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    structure = Structure.from_file(str(structure_path))
    structure.to(fmt="poscar", filename=str(case_dir / "POSCAR"))

    template_dir = args.template_dir
    template_used = bool(template_dir is not None)
    if template_dir is not None:
        incar_tpl = template_dir / "INCAR"
        if not incar_tpl.exists():
            raise FileNotFoundError(f"Template INCAR not found at {incar_tpl}")
        incar_text = _replace_composition_tokens(_read_text(incar_tpl), structure)
        (case_dir / "INCAR").write_text(incar_text, encoding="utf-8")

        kpoints_tpl = template_dir / "KPOINTS"
        if kpoints_tpl.exists():
            (case_dir / "KPOINTS").write_text(_read_text(kpoints_tpl), encoding="utf-8")
        else:
            Kpoints.gamma_automatic(kpts=(1, 1, 1)).write_file(str(case_dir / "KPOINTS"))

        potcar_tpl = template_dir / "POTCAR"
        if potcar_tpl.exists():
            (case_dir / "POTCAR").write_bytes(potcar_tpl.read_bytes())
    else:
        mpset = MPStaticSet(structure)
        mpset.write_input(str(case_dir), potcar_spec=bool(args.potcar_spec))

    slurm_tpl = (template_dir / "run_vasp.slurm") if template_dir is not None else None
    slurm_path = case_dir / "run_vasp.slurm"
    if slurm_tpl is not None and slurm_tpl.exists():
        slurm_text = _read_text(slurm_tpl)
    else:
        slurm_text = _build_perlmutter_slurm(
            account=args.perlmutter_account,
            queue=args.perlmutter_queue,
            walltime=args.slurm_time,
            nodes=args.slurm_nodes,
            ntasks_per_node=args.slurm_ntasks_per_node,
            cpus_per_task=args.slurm_cpus_per_task,
            gpus=args.slurm_gpus,
            module=args.vasp_module,
            vasp_exe=args.vasp_exe,
            job_name=f"{args.job_name_prefix}_{case_id[:30]}",
        )
    slurm_path.write_text(slurm_text, encoding="utf-8")

    context = CaseContext(
        case_id=case_id,
        case_dir=case_dir,
        source_structure=structure_path,
        temperature_k=temperature_k,
        pressure_mpa=pressure_mpa,
        lithiation_pct=lithiation_pct,
    )
    _write_case_manifest(
        context,
        case_dir / "run_manifest.json",
        {
            "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "template_mode": template_used,
            "submitted_job_ids": [],
        },
    )
    return context


def cmd_prepare_inputs(args: argparse.Namespace) -> None:
    best_root = _resolve_best_training_set_root(args.structures_root)
    structures = _discover_structure_files(best_root)
    if args.max_cases is not None:
        structures = structures[: max(int(args.max_cases), 0)]

    output_root = args.output_dir
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    pressure_lookup = _pressure_map(args.temperatures, args.pressures_mpa)
    prepared: list[CaseContext] = []
    for structure_path in structures:
        prepared.append(_prepare_case(structure_path, best_root, cases_root, pressure_lookup, args))

    summary = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "structures_root": str(best_root),
        "cases_root": str(cases_root),
        "n_cases": len(prepared),
        "template_dir": str(args.template_dir) if args.template_dir else None,
        "cases": [
            {
                "case_id": case.case_id,
                "case_dir": str(case.case_dir),
                "source_structure": str(case.source_structure),
                "temperature_k": case.temperature_k,
                "pressure_mpa": case.pressure_mpa,
                "lithiation_pct": case.lithiation_pct,
            }
            for case in prepared
        ],
    }
    (output_root / "prepared_cases_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Prepared {len(prepared)} VASP cases under {cases_root}")


def _resolve_cases_root(path: Path) -> Path:
    if (path / "cases").exists():
        return path / "cases"
    return path


def _discover_case_dirs(cases_root: Path) -> list[Path]:
    case_dirs = sorted({p.parent for p in cases_root.rglob("run_vasp.slurm")})
    if case_dirs:
        return case_dirs
    fallback = sorted({p.parent for p in cases_root.rglob("POSCAR")})
    return fallback


def _read_manifest(case_dir: Path) -> dict[str, object]:
    path = case_dir / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest_dict(case_dir: Path, manifest: dict[str, object]) -> None:
    manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    (case_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _tail_text(path: Path, max_bytes: int = 64_000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-max_bytes:].decode("utf-8", errors="ignore")


def _extract_job_ids(case_dir: Path, manifest: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    queued = manifest.get("submitted_job_ids", [])
    if isinstance(queued, list):
        ids.update(str(x) for x in queued if str(x).strip())
    for path in case_dir.glob("*.out"):
        match = re.search(r"-(\d+)\.out$", path.name)
        if match:
            ids.add(match.group(1))
    for path in case_dir.glob("*.err"):
        match = re.search(r"-(\d+)\.err$", path.name)
        if match:
            ids.add(match.group(1))
    return ids


def _active_job_ids(user: str) -> set[str]:
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-u", user, "-o", "%A"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return set()
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _is_completed(case_dir: Path) -> bool:
    outcar = case_dir / "OUTCAR"
    text = _tail_text(outcar)
    if not text:
        return False
    markers = [
        "General timing and accounting informations for this job",
        "Voluntary context switches",
        "reached required accuracy - stopping structural energy minimisation",
    ]
    return any(marker in text for marker in markers)


def _has_runtime_outputs(case_dir: Path) -> bool:
    candidates = ["OUTCAR", "OSZICAR", "vasprun.xml"]
    if any((case_dir / name).exists() for name in candidates):
        return True
    if any(case_dir.glob("*.out")) or any(case_dir.glob("*.err")):
        return True
    return False


def _case_status(case_dir: Path, active_jobs: set[str], user: str) -> str:
    del user
    if _is_completed(case_dir):
        return "completed"
    manifest = _read_manifest(case_dir)
    ids = _extract_job_ids(case_dir, manifest)
    if ids & active_jobs:
        return "running"
    if _has_runtime_outputs(case_dir):
        return "fizzled"
    return "pending"


def _summarize_status(cases_root: Path, user: str) -> dict[str, object]:
    case_dirs = _discover_case_dirs(cases_root)
    active_jobs = _active_job_ids(user)
    by_state: dict[str, list[str]] = defaultdict(list)
    for case_dir in case_dirs:
        state = _case_status(case_dir, active_jobs, user)
        by_state[state].append(str(case_dir))

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cases_root": str(cases_root),
        "counts": {key: len(value) for key, value in by_state.items()},
        "fizzled_cases": sorted(by_state.get("fizzled", [])),
        "running_cases": sorted(by_state.get("running", [])),
        "completed_cases": sorted(by_state.get("completed", [])),
        "pending_cases": sorted(by_state.get("pending", [])),
    }
    return summary


def cmd_status(args: argparse.Namespace) -> None:
    user = args.user or os.environ.get("USER", "")
    if not user:
        raise RuntimeError("Could not determine username. Pass --user explicitly.")
    cases_root = _resolve_cases_root(args.cases_root)
    summary = _summarize_status(cases_root, user)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))

    state_key_map = {
        "completed": "completed_cases",
        "running": "running_cases",
        "fizzled": "fizzled_cases",
        "pending": "pending_cases",
    }
    requested_states = list(dict.fromkeys(args.list_states))
    if not requested_states and summary["fizzled_cases"]:
        requested_states = ["fizzled"]

    for state in requested_states:
        case_paths = summary[state_key_map[state]]
        print(f"{state.capitalize()} case directories ({len(case_paths)}):")
        for path in case_paths:
            print(path)


def cmd_submit(args: argparse.Namespace) -> None:
    user = args.user or os.environ.get("USER", "")
    if not user:
        raise RuntimeError("Could not determine username. Pass --user explicitly.")

    cases_root = _resolve_cases_root(args.cases_root)
    case_dirs = _discover_case_dirs(cases_root)
    active_jobs = _active_job_ids(user)

    submitted = 0
    skipped = 0
    for case_dir in case_dirs:
        state = _case_status(case_dir, active_jobs, user)
        if state in {"completed", "running"}:
            skipped += 1
            continue
        if state == "fizzled" and not args.resubmit_fizzled:
            skipped += 1
            continue

        script = case_dir / "run_vasp.slurm"
        if not script.exists():
            skipped += 1
            continue

        if args.dry_run:
            print(f"[dry-run] would submit {script}")
            submitted += 1
            continue

        proc = subprocess.run(["sbatch", script.name], cwd=str(case_dir), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(f"[submit-error] {case_dir}: {proc.stderr.strip()}")
            continue

        out = proc.stdout.strip()
        match = re.search(r"Submitted batch job\s+(\d+)", out)
        job_id = match.group(1) if match else None

        manifest = _read_manifest(case_dir)
        existing = manifest.get("submitted_job_ids", [])
        submitted_ids = list(existing) if isinstance(existing, list) else []
        if job_id is not None:
            submitted_ids.append(job_id)
            active_jobs.add(job_id)
        manifest["submitted_job_ids"] = submitted_ids
        manifest["last_submit_stdout"] = out
        _write_manifest_dict(case_dir, manifest)
        submitted += 1

    print(f"Submitted: {submitted}; skipped: {skipped}")


def _read_vasp_images(case_dir: Path) -> list:
    if (case_dir / "vasprun.xml").exists():
        images = ase.io.read(str(case_dir / "vasprun.xml"), index=":", format="vasp-xml")
        if isinstance(images, list) and images:
            return images
        if images is not None:
            return [images]
    if (case_dir / "OUTCAR").exists():
        images = ase.io.read(str(case_dir / "OUTCAR"), index=":", format="vasp-out")
        if isinstance(images, list) and images:
            return images
        if images is not None:
            return [images]
    raise FileNotFoundError(f"No readable VASP trajectory in {case_dir}")


def _uma_single_point(final_atoms, model: str, task_name: str, device: str):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    import torch

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    predictor = pretrained_mlip.get_predict_unit(model, device=resolved_device)
    calc = FAIRChemCalculator(predictor, task_name=task_name)

    atoms = final_atoms.copy()
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return energy, forces, resolved_device


def _force_metrics(vasp_forces: np.ndarray, uma_forces: np.ndarray) -> tuple[float, float, float]:
    delta = uma_forces - vasp_forces
    norms = np.linalg.norm(delta, axis=1)
    return float(np.mean(norms)), float(np.sqrt(np.mean(norms * norms))), float(np.max(norms))


def _parse_case_meta(case_dir: Path, manifest: dict[str, object]) -> tuple[float | None, int | None, float | None]:
    lith = manifest.get("lithiation_pct")
    temp = manifest.get("temperature_k")
    pressure = manifest.get("pressure_mpa")

    lithiation = float(lith) if lith is not None else None
    temperature = int(temp) if temp is not None else None
    pressure_mpa = float(pressure) if pressure is not None else None

    if lithiation is None or temperature is None:
        for token in case_dir.parts:
            l_match = re.match(r"lith_([0-9]*\.?[0-9]+)pct$", token)
            if l_match:
                lithiation = float(l_match.group(1))
            t_match = re.match(r"T_(\d+)K$", token)
            if t_match:
                temperature = int(t_match.group(1))
    return lithiation, temperature, pressure_mpa


def _write_analysis_plots(records: list[dict[str, object]], analysis_dir: Path) -> None:
    if not records:
        return

    x = np.array([float(r.get("lithiation_pct", np.nan)) for r in records], dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return

    metrics = [
        ("delta_energy_per_atom_eV", "ΔE per atom (eV)", "uma_vs_vasp_delta_energy_per_atom_vs_lithiation.png"),
        ("mean_force_difference_eV_per_A", "Mean |ΔF| (eV/Å)", "uma_vs_vasp_mean_force_vs_lithiation.png"),
        ("rms_force_difference_eV_per_A", "RMS |ΔF| (eV/Å)", "uma_vs_vasp_rms_force_vs_lithiation.png"),
    ]

    for key, ylabel, filename in metrics:
        y = np.array([float(r.get(key, np.nan)) for r in records], dtype=float)
        mask = valid & np.isfinite(y)
        if not np.any(mask):
            continue

        plt.figure(figsize=(8, 5))
        plt.scatter(x[mask], y[mask], alpha=0.25, s=20, label="All cases")

        grouped: dict[float, list[float]] = defaultdict(list)
        for xi, yi in zip(x[mask], y[mask]):
            grouped[float(xi)].append(float(yi))
        xs = sorted(grouped.keys())
        means = np.array([np.mean(grouped[val]) for val in xs], dtype=float)
        stds = np.array([np.std(grouped[val]) for val in xs], dtype=float)

        plt.errorbar(xs, means, yerr=stds, fmt="o-", capsize=4, linewidth=1.5, label="Mean ± std")
        plt.xlabel("Lithiation (%)")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.2)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(analysis_dir / filename, dpi=220)
        plt.close()


def cmd_analyze_uma_vs_vasp(args: argparse.Namespace) -> None:
    cases_root = _resolve_cases_root(args.cases_root)
    case_dirs = _discover_case_dirs(cases_root)

    analysis_dir = args.analysis_dir or (cases_root.parent / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir = analysis_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for case_dir in case_dirs:
        if not _is_completed(case_dir):
            continue

        manifest = _read_manifest(case_dir)
        try:
            images = _read_vasp_images(case_dir)
        except Exception:
            continue

        final_atoms = images[-1].copy()
        final_atoms.calc = None
        try:
            vasp_energy = float(final_atoms.get_potential_energy())
            vasp_forces = np.asarray(final_atoms.get_forces(), dtype=float)
        except Exception:
            # fallback through outcar single image parser with calculator attached
            final_atoms = ase.io.read(str(case_dir / "OUTCAR"), index=-1, format="vasp-out")
            vasp_energy = float(final_atoms.get_potential_energy())
            vasp_forces = np.asarray(final_atoms.get_forces(), dtype=float)

        uma_energy, uma_forces, resolved_device = _uma_single_point(
            final_atoms,
            model=args.model,
            task_name=args.task_name,
            device=args.device,
        )
        mean_df, rms_df, max_df = _force_metrics(vasp_forces, uma_forces)
        natoms = len(final_atoms)

        lithiation_pct, temperature_k, pressure_mpa = _parse_case_meta(case_dir, manifest)
        record = {
            "case_id": manifest.get("case_id", case_dir.name),
            "case_dir": str(case_dir),
            "natoms": int(natoms),
            "lithiation_pct": lithiation_pct,
            "temperature_k": temperature_k,
            "pressure_mpa": pressure_mpa,
            "vasp_energy_eV": float(vasp_energy),
            "uma_energy_eV": float(uma_energy),
            "delta_energy_eV": float(uma_energy - vasp_energy),
            "delta_energy_per_atom_eV": float((uma_energy - vasp_energy) / max(natoms, 1)),
            "mean_force_difference_eV_per_A": mean_df,
            "rms_force_difference_eV_per_A": rms_df,
            "max_force_difference_eV_per_A": max_df,
            "uma_model": args.model,
            "uma_task_name": args.task_name,
            "uma_device": resolved_device,
        }
        records.append(record)

        (per_case_dir / f"{_safe_id(str(record['case_id']))}_summary.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

    csv_path = analysis_dir / "uma_vs_vasp_all_cases.csv"
    if records:
        fields = list(records[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow(row)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cases_root": str(cases_root),
        "analysis_dir": str(analysis_dir),
        "n_analyzed_cases": len(records),
        "csv": str(csv_path),
    }
    (analysis_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_analysis_plots(records, analysis_dir)
    print(f"Analyzed {len(records)} completed cases. Outputs in {analysis_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-inputs":
        cmd_prepare_inputs(args)
        return
    if args.command == "submit":
        cmd_submit(args)
        return
    if args.command == "status":
        cmd_status(args)
        return
    if args.command == "analyze-uma-vs-vasp":
        cmd_analyze_uma_vs_vasp(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
