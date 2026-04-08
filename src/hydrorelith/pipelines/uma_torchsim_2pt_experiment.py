from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from hydrorelith.pipelines.uma_torchsim_screen_config import ScreenConfig, default_config


DEFAULT_TP_MAP = "393:0.08,433:0.46,473:1.32,493:2.02"


@dataclass(slots=True)
class SelectedStructure:
    condition_id: str
    structure_path: Path
    temperature_k: float
    pressure_mpa: float
    lithiation_fraction: float
    vacancy_config_id: str


@dataclass(slots=True)
class SelectedElectrolyteStructure:
    condition_id: str
    structure_path: Path
    temperature_k: float
    pressure_mpa: float
    liOH_M: float
    kOH_M: float


def _default_tiny_root() -> Path:
    return Path(
        "results/publication/default_systems/electrode/"
        "LCO_mp-22526/structure_generation_tiny/best_training_set"
    )


def _default_electrolyte_best_training_root() -> Path:
    return Path(
        "results/publication/default_systems/electrolyte/"
        "LiOH_KOH_H2O/structure_generation/best_training_set"
    )


def _default_omat_finetune_root() -> Path:
    return Path(
        "results/publication/default_systems/electrode/LCO_mp-22526/"
        "structure_generation_tiny/uma_finetune_workflow"
    )


def _format_num_token(value: float, ndigits: int = 2) -> str:
    return f"{value:.{ndigits}f}".replace(".", "p")


def _parse_temperature_k(dirname: str) -> float:
    match = re.fullmatch(r"T_(\d+(?:\.\d+)?)K", dirname)
    if match is None:
        raise ValueError(f"Expected temperature directory like T_393K, got: {dirname}")
    return float(match.group(1))


def _parse_lithiation_fraction(dirname: str) -> float:
    match = re.fullmatch(r"lith_(\d+(?:\.\d+)?)pct", dirname)
    if match is None:
        raise ValueError(f"Expected lithiation directory like lith_75.00pct, got: {dirname}")
    return float(match.group(1)) / 100.0


def _parse_tp_map(raw: str) -> dict[float, float]:
    out: dict[float, float] = {}
    entries = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not entries:
        raise ValueError("Temperature/pressure map cannot be empty")
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid T/P entry '{entry}'. Use format 393:0.08")
        t_raw, p_raw = [part.strip() for part in entry.split(":", maxsplit=1)]
        out[float(t_raw)] = float(p_raw)
    return out


def _parse_li_k_concentration(dirname: str) -> tuple[float, float]:
    match = re.fullmatch(r"LiOH_(\d+(?:\.\d+)?)_KOH_(\d+(?:\.\d+)?)", dirname)
    if match is None:
        raise ValueError(
            f"Expected concentration directory like LiOH_2.00_KOH_2.00, got: {dirname}"
        )
    return float(match.group(1)), float(match.group(2))


def _candidate_structure_files(root: Path) -> list[Path]:
    supported = {".xyz", ".extxyz", ".cif", ".pdb", ".vasp"}
    out = [
        path
        for path in sorted(root.iterdir())
        if path.is_file() and (path.suffix.lower() in supported or path.name.upper().startswith("POSCAR"))
    ]
    return out


def _find_latest_omat_checkpoint(finetune_root: Path) -> Path:
    if not finetune_root.exists():
        raise FileNotFoundError(f"Fine-tune workflow root not found: {finetune_root}")
    candidates = [
        path
        for path in finetune_root.rglob("inference_ckpt.pt")
        if "checkpoints" in path.as_posix()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No inference_ckpt.pt files found under fine-tune workflow root: "
            f"{finetune_root}"
        )

    def _rank(path: Path) -> tuple[int, float]:
        is_final = 1 if "/checkpoints/final/" in path.as_posix() else 0
        return (is_final, path.stat().st_mtime)

    return max(candidates, key=_rank)


def _select_direct_tiny_structures(
    tiny_root: Path,
    pressure_by_temp_k: dict[float, float],
    selection_index: int,
) -> list[SelectedStructure]:
    if not tiny_root.exists():
        raise FileNotFoundError(f"Tiny best_training_set root not found: {tiny_root}")
    if selection_index < 0:
        raise ValueError("selection_index must be >= 0")

    selected: list[SelectedStructure] = []
    temp_dirs = sorted([path for path in tiny_root.iterdir() if path.is_dir() and path.name.startswith("T_")])
    if not temp_dirs:
        raise ValueError(f"No T_*K directories found under {tiny_root}")

    for temp_dir in temp_dirs:
        temperature_k = _parse_temperature_k(temp_dir.name)
        if temperature_k not in pressure_by_temp_k:
            raise ValueError(
                f"No pressure specified for temperature {temperature_k:g} K. "
                f"Provide it in --tp-map."
            )
        pressure_mpa = pressure_by_temp_k[temperature_k]

        lith_dirs = sorted([path for path in temp_dir.iterdir() if path.is_dir() and path.name.startswith("lith_")])
        if not lith_dirs:
            raise ValueError(f"No lith_* directories found under {temp_dir}")

        for lith_dir in lith_dirs:
            lith_fraction = _parse_lithiation_fraction(lith_dir.name)
            candidates = sorted([path for path in lith_dir.iterdir() if path.is_file() and path.name.upper().startswith("POSCAR")])
            if not candidates:
                raise ValueError(f"No POSCAR* files found under {lith_dir}")
            if selection_index >= len(candidates):
                raise ValueError(
                    f"Selection index {selection_index} out of range for {lith_dir}; "
                    f"available count={len(candidates)}"
                )
            chosen = candidates[selection_index]
            condition_id = (
                f"ed_t{int(round(temperature_k)):03d}k_"
                f"p{_format_num_token(pressure_mpa, ndigits=2)}mpa_"
                f"lith{_format_num_token(lith_fraction * 100.0, ndigits=2)}pct"
            )
            selected.append(
                SelectedStructure(
                    condition_id=condition_id,
                    structure_path=chosen,
                    temperature_k=temperature_k,
                    pressure_mpa=pressure_mpa,
                    lithiation_fraction=lith_fraction,
                    vacancy_config_id=chosen.stem,
                )
            )

    if not selected:
        raise ValueError("No structures selected from tiny DIRECT best_training_set")
    return selected


def _select_electrolyte_structures(
    electrolyte_best_training_root: Path,
    pressure_by_temp_k: dict[float, float],
    selection_index: int,
) -> list[SelectedElectrolyteStructure]:
    if not electrolyte_best_training_root.exists():
        raise FileNotFoundError(
            f"Electrolyte best_training_set root not found: {electrolyte_best_training_root}"
        )
    if selection_index < 0:
        raise ValueError("electrolyte_selection_index must be >= 0")

    selected: list[SelectedElectrolyteStructure] = []
    temp_dirs = sorted(
        [path for path in electrolyte_best_training_root.iterdir() if path.is_dir() and path.name.startswith("T_")]
    )
    if not temp_dirs:
        raise ValueError(f"No T_*K directories found under {electrolyte_best_training_root}")

    for temp_dir in temp_dirs:
        temperature_k = _parse_temperature_k(temp_dir.name)
        if temperature_k not in pressure_by_temp_k:
            raise ValueError(
                f"No pressure specified for temperature {temperature_k:g} K. "
                f"Provide it in --tp-map."
            )
        pressure_mpa = pressure_by_temp_k[temperature_k]

        concentration_dirs = sorted([path for path in temp_dir.iterdir() if path.is_dir() and path.name.startswith("LiOH_")])
        if not concentration_dirs:
            raise ValueError(f"No LiOH_* concentration directories found under {temp_dir}")

        for concentration_dir in concentration_dirs:
            li_m, k_m = _parse_li_k_concentration(concentration_dir.name)
            candidates = _candidate_structure_files(concentration_dir)
            if not candidates:
                raise ValueError(f"No structure files found under {concentration_dir}")
            if selection_index >= len(candidates):
                raise ValueError(
                    f"Selection index {selection_index} out of range for {concentration_dir}; "
                    f"available count={len(candidates)}"
                )
            chosen = candidates[selection_index]
            condition_id = (
                f"el_t{int(round(temperature_k)):03d}k_"
                f"p{_format_num_token(pressure_mpa, ndigits=2)}mpa_"
                f"li{_format_num_token(li_m, ndigits=2)}_k{_format_num_token(k_m, ndigits=2)}"
            )
            selected.append(
                SelectedElectrolyteStructure(
                    condition_id=condition_id,
                    structure_path=chosen,
                    temperature_k=temperature_k,
                    pressure_mpa=pressure_mpa,
                    liOH_M=li_m,
                    kOH_M=k_m,
                )
            )

    if not selected:
        raise ValueError("No electrolyte structures selected")
    return selected


def _write_electrode_manifest(rows: list[SelectedStructure], out_csv: Path, task_name: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )
        for row in rows:
            writer.writerow(
                [
                    row.condition_id,
                    str(row.structure_path),
                    f"{row.temperature_k - 273.15:.2f}",
                    f"{row.pressure_mpa:.6g}",
                    f"{row.lithiation_fraction:.8f}",
                    row.vacancy_config_id,
                    "electrode",
                    task_name,
                    "",
                    "selected_from_structure_generation_tiny_best_training_set",
                ]
            )


def _write_electrolyte_manifest(
    rows: list[SelectedElectrolyteStructure],
    out_csv: Path,
    task_name: str,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )
        for row in rows:
            writer.writerow(
                [
                    row.condition_id,
                    str(row.structure_path),
                    f"{row.temperature_k - 273.15:.2f}",
                    f"{row.pressure_mpa:.6g}",
                    f"{row.liOH_M:.6g}",
                    f"{row.kOH_M:.6g}",
                    "electrolyte",
                    task_name,
                    "0",
                    "1",
                    "selected_from_electrolyte_best_training_set",
                ]
            )


def _export_2pt_for_phase(config: ScreenConfig, phase: str) -> Path:
    from hydrorelith.io.torchsim_export2pt import (
        export_h5md_to_lammps_dump,
        write_2pt_metadata,
        write_lammps_eng_from_thermo_csv,
    )

    phase_root = config.output_dir / "uma_torchsim_screen" / phase
    out_root = config.output_dir / "uma_torchsim_screen" / "export2pt" / phase
    cond_dirs = sorted([path for path in phase_root.iterdir() if path.is_dir()] if phase_root.exists() else [])
    for cond_dir in tqdm(cond_dirs, desc=f"export-2pt {phase}", unit="condition"):
        rep_dirs = sorted([path for path in cond_dir.iterdir() if path.is_dir() and path.name.startswith("replica_")])
        for rep_dir in rep_dirs:
            prod_h5 = rep_dir / "prod.h5md"
            thermo = rep_dir / "prod_thermo.csv"
            if not (prod_h5.exists() and thermo.exists()):
                continue
            out_dir = out_root / cond_dir.name / rep_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            export_h5md_to_lammps_dump(
                prod_h5,
                out_dir / "prod.lammpstrj",
                unwrap=True,
                include_ke_atom=True,
                lammps_template_style=True,
            )
            export_h5md_to_lammps_dump(
                prod_h5,
                out_dir / "prod.lammps",
                unwrap=True,
                include_ke_atom=True,
                lammps_template_style=True,
            )
            write_lammps_eng_from_thermo_csv(thermo, out_dir / "prod.eng")
            write_2pt_metadata(
                thermo,
                out_dir / "2pt_metadata.json",
                timestep_ps=config.timestep_ps,
                dump_every_steps=config.dump_every_steps,
            )
    return out_root


def _iter_export_replica_dirs(export_root: Path) -> list[Path]:
    cond_dirs = sorted([path for path in export_root.iterdir() if path.is_dir()] if export_root.exists() else [])
    rep_dirs: list[Path] = []
    for cond_dir in cond_dirs:
        rep_dirs.extend(sorted([path for path in cond_dir.iterdir() if path.is_dir() and path.name.startswith("replica_")]))
    return rep_dirs


def _run_2pt_backend(backend_name: str, command_template: str, export_root: Path) -> None:
    replica_dirs = _iter_export_replica_dirs(export_root)
    if not replica_dirs:
        raise ValueError(f"No exported replica directories found under {export_root}")

    failures: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    bar = tqdm(replica_dirs, desc=f"2PT {backend_name}", unit="replica")
    for rep_dir in bar:
        placeholders = {
            "export_dir": str(rep_dir),
            "traj": str(rep_dir / "prod.lammpstrj"),
            "metadata": str(rep_dir / "2pt_metadata.json"),
            "type_map": str(rep_dir / "type_map.json"),
        }
        command = command_template.format(**placeholders)
        t0 = time.perf_counter()
        proc = subprocess.run(
            shlex.split(command),
            cwd=rep_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_s = time.perf_counter() - t0

        (rep_dir / f"{backend_name}.stdout.log").write_text(proc.stdout, encoding="utf-8")
        (rep_dir / f"{backend_name}.stderr.log").write_text(proc.stderr, encoding="utf-8")

        row = {
            "replica_dir": str(rep_dir),
            "backend": backend_name,
            "command": command,
            "returncode": int(proc.returncode),
            "elapsed_s": float(elapsed_s),
        }
        summary_rows.append(row)
        if proc.returncode != 0:
            failures.append(row)
            bar.set_postfix_str("failure detected")

    summary_path = export_root / f"{backend_name}_run_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(
            f"{backend_name} failed for {len(failures)} replica(s). "
            f"See logs under export dirs and {summary_path}."
        )


def _steps_from_fs(duration_fs: int, timestep_ps: float) -> int:
    timestep_fs = float(timestep_ps) * 1000.0
    if timestep_fs <= 0:
        raise ValueError("timestep_ps must be > 0")
    steps = int(round(float(duration_fs) / timestep_fs))
    if steps <= 0:
        raise ValueError("Computed step count must be > 0")
    return steps


def _build_config(args, phase: str, manifest_path: Path, model_name: str) -> ScreenConfig:
    config = default_config()
    config.electrode_manifest = manifest_path if phase == "electrode" else None
    config.electrolyte_manifest = manifest_path if phase == "electrolyte" else None
    config.electrode_root = None
    config.electrolyte_root = None
    config.output_dir = args.output_dir
    config.phase = phase
    config.stages = ("md", "export-2pt")
    config.model_name = model_name
    config.device = args.device
    config.ensemble = "nvt"
    config.timestep_ps = args.timestep_ps
    config.dump_every_steps = args.dump_every_steps
    config.retherm_steps_electrode = _steps_from_fs(args.npt_equil_fs, args.timestep_ps)
    config.retherm_steps_electrolyte = config.retherm_steps_electrode
    config.prod_steps = _steps_from_fs(args.nvt_prod_fs, args.timestep_ps)
    config.replicas = args.replicas
    config.base_seed = args.base_seed
    config.compute_stress = True
    config.skip_batch_benchmark = args.skip_batch_benchmark
    config.max_memory_scaler = args.max_memory_scaler
    config.precision = args.precision
    config.debug = args.debug
    return config


def _resolve_electrode_model_name(args) -> str:
    if args.electrode_model_source == "model-name":
        return args.electrode_model_name
    if args.electrode_model_source == "checkpoint-path":
        if args.electrode_checkpoint_path is None:
            raise ValueError(
                "--electrode-checkpoint-path is required when "
                "--electrode-model-source=checkpoint-path"
            )
        if not args.electrode_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Electrode checkpoint path not found: {args.electrode_checkpoint_path}"
            )
        return str(args.electrode_checkpoint_path)
    latest = _find_latest_omat_checkpoint(args.omat_finetune_root)
    return str(latest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-uma-torchsim-2pt-experiment",
        description=(
            "Run TorchSim UMA NPT->NVT experiments for electrode/electrolyte datasets and "
            "prepare full 2PT-ready outputs with optional backend execution."
        ),
    )
    parser.add_argument("--campaign", choices=["electrode", "electrolyte", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tp-map",
        type=str,
        default=DEFAULT_TP_MAP,
        help="Comma-separated K:MPa map, e.g. 393:0.08,433:0.46,473:1.32,493:2.02",
    )

    parser.add_argument("--tiny-root", type=Path, default=_default_tiny_root())
    parser.add_argument("--electrode-selection-index", type=int, default=0)
    parser.add_argument(
        "--electrode-model-source",
        choices=["finetuned-latest", "model-name", "checkpoint-path"],
        default="finetuned-latest",
    )
    parser.add_argument("--electrode-model-name", type=str, default="uma-s-1p2")
    parser.add_argument("--electrode-checkpoint-path", type=Path, default=None)
    parser.add_argument("--omat-finetune-root", type=Path, default=_default_omat_finetune_root())

    parser.add_argument("--electrolyte-root", type=Path, default=_default_electrolyte_best_training_root())
    parser.add_argument("--electrolyte-selection-index", type=int, default=0)
    parser.add_argument("--electrolyte-model-name", type=str, default="uma-s-1p2")
    parser.add_argument("--electrolyte-task-name", type=str, default="omol")

    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")

    parser.add_argument("--npt-equil-fs", type=int, default=500000)
    parser.add_argument("--nvt-prod-fs", type=int, default=100000)
    parser.add_argument("--timestep-ps", type=float, default=0.001)
    parser.add_argument("--dump-every-steps", type=int, default=2)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)

    parser.add_argument("--max-memory-scaler", type=float, default=None)
    parser.add_argument("--skip-batch-benchmark", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--run-2pt-backends", choices=["none", "cpp", "python", "both"], default="none")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare manifests and selection summary; skip MD, export, and 2PT backend execution.",
    )
    parser.add_argument(
        "--two-pt-cpp-cmd-template",
        type=str,
        default=None,
        help=(
            "Command template to run C++ 2PT per replica export. Supported placeholders: "
            "{export_dir}, {traj}, {metadata}, {type_map}"
        ),
    )
    parser.add_argument(
        "--two-pt-python-cmd-template",
        type=str,
        default=None,
        help=(
            "Command template to run 2pt_python per replica export. Supported placeholders: "
            "{export_dir}, {traj}, {metadata}, {type_map}"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pressure_by_temp_k = _parse_tp_map(args.tp_map)

    run_root = args.output_dir / "uma_torchsim_screen"
    run_root.mkdir(parents=True, exist_ok=True)
    phase_export_roots: dict[str, Path] = {}
    selection_summary: dict[str, Any] = {"campaign": args.campaign, "phases": {}}

    run_summary: dict[str, Any] = {
        "campaign": args.campaign,
        "prepare_only": bool(args.prepare_only),
        "device": args.device,
        "precision": args.precision,
        "npt_equil_fs": int(args.npt_equil_fs),
        "nvt_prod_fs": int(args.nvt_prod_fs),
        "timestep_ps": float(args.timestep_ps),
        "dump_every_steps": int(args.dump_every_steps),
        "replicas": int(args.replicas),
        "run_2pt_backends": args.run_2pt_backends,
    }

    if args.campaign in {"electrode", "both"}:
        selected = _select_direct_tiny_structures(
            args.tiny_root,
            pressure_by_temp_k=pressure_by_temp_k,
            selection_index=args.electrode_selection_index,
        )
        manifest_path = run_root / "electrode_manifest.auto.csv"
        _write_electrode_manifest(selected, manifest_path, task_name="omat")
        electrode_model_name = _resolve_electrode_model_name(args)

        selection_summary["phases"]["electrode"] = {
            "manifest_path": str(manifest_path),
            "model_name": electrode_model_name,
            "n_conditions": len(selected),
            "selection": [
                {
                    "condition_id": item.condition_id,
                    "structure_path": str(item.structure_path),
                    "temperature_k": item.temperature_k,
                    "pressure_mpa": item.pressure_mpa,
                    "lithiation_fraction": item.lithiation_fraction,
                }
                for item in selected
            ],
        }

        config = _build_config(args, phase="electrode", manifest_path=manifest_path, model_name=electrode_model_name)
        run_summary["electrode_model_name"] = electrode_model_name
        if not args.prepare_only:
            from hydrorelith.pipelines.uma_torchsim_screen_run import run_one_phase

            run_one_phase("electrode", config)
            phase_export_roots["electrode"] = _export_2pt_for_phase(config, "electrode")

    if args.campaign in {"electrolyte", "both"}:
        selected_el = _select_electrolyte_structures(
            args.electrolyte_root,
            pressure_by_temp_k=pressure_by_temp_k,
            selection_index=args.electrolyte_selection_index,
        )
        manifest_path = run_root / "electrolyte_manifest.auto.csv"
        _write_electrolyte_manifest(selected_el, manifest_path, task_name=args.electrolyte_task_name)

        selection_summary["phases"]["electrolyte"] = {
            "manifest_path": str(manifest_path),
            "model_name": args.electrolyte_model_name,
            "task_name": args.electrolyte_task_name,
            "n_conditions": len(selected_el),
            "selection": [
                {
                    "condition_id": item.condition_id,
                    "structure_path": str(item.structure_path),
                    "temperature_k": item.temperature_k,
                    "pressure_mpa": item.pressure_mpa,
                    "liOH_M": item.liOH_M,
                    "kOH_M": item.kOH_M,
                }
                for item in selected_el
            ],
        }

        config = _build_config(
            args,
            phase="electrolyte",
            manifest_path=manifest_path,
            model_name=args.electrolyte_model_name,
        )
        if not args.prepare_only:
            from hydrorelith.pipelines.uma_torchsim_screen_run import run_one_phase

            run_one_phase("electrolyte", config)
            phase_export_roots["electrolyte"] = _export_2pt_for_phase(config, "electrolyte")

    (run_root / "experiment_selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2),
        encoding="utf-8",
    )

    if args.prepare_only:
        (run_root / "experiment_run_summary.json").write_text(
            json.dumps(run_summary, indent=2),
            encoding="utf-8",
        )
        return

    for phase_name, export_root in phase_export_roots.items():
        if args.run_2pt_backends in {"cpp", "both"}:
            if not args.two_pt_cpp_cmd_template:
                raise ValueError(
                    "--two-pt-cpp-cmd-template is required when --run-2pt-backends includes cpp"
                )
            _run_2pt_backend(f"2pt_cpp_{phase_name}", args.two_pt_cpp_cmd_template, export_root)
        if args.run_2pt_backends in {"python", "both"}:
            if not args.two_pt_python_cmd_template:
                raise ValueError(
                    "--two-pt-python-cmd-template is required when --run-2pt-backends includes python"
                )
            _run_2pt_backend(
                f"2pt_python_{phase_name}",
                args.two_pt_python_cmd_template,
                export_root,
            )

    (run_root / "experiment_run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
