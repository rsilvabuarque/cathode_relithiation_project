from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from hydrorelith.io.torchsim_export2pt import (
    ATM_PER_MPA,
    export_h5md_to_lammps_dump,
    write_2pt_metadata,
    write_lammps_data_full_from_structure,
    write_lammps_eng_from_thermo_csv,
)
from hydrorelith.pipelines.uma_torchsim_screen_config import default_config
from hydrorelith.pipelines.uma_torchsim_screen_run import run_one_phase


def _steps_from_duration_ps(duration_ps: float, timestep_fs: float) -> int:
    if duration_ps <= 0:
        raise ValueError("Duration must be > 0 ps")
    if timestep_fs <= 0:
        raise ValueError("Timestep must be > 0 fs")
    steps = int(round((float(duration_ps) * 1000.0) / float(timestep_fs)))
    if steps <= 0:
        raise ValueError("Computed number of steps must be > 0")
    return steps


def _write_single_electrolyte_manifest(
    out_csv: Path,
    *,
    condition_id: str,
    structure_path: Path,
    temperature_k: float,
    pressure_atm: float,
    task_name: str,
    charge: int,
    spin: int,
) -> None:
    temperature_c = float(temperature_k) - 273.15
    pressure_mpa = float(pressure_atm) / ATM_PER_MPA

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
        writer.writerow(
            [
                condition_id,
                str(structure_path),
                f"{temperature_c:.6f}",
                f"{pressure_mpa:.10f}",
                "",
                "",
                "electrolyte",
                task_name,
                str(int(charge)),
                str(int(spin)),
                "single-water-box-2pt-benchmark",
            ]
        )


def _export_lammps_artifacts(
    output_dir: Path,
    *,
    sname: str,
    structure_path: Path,
    timestep_ps: float,
    dump_every_steps: int,
) -> list[Path]:
    phase_root = output_dir / "uma_torchsim_screen" / "electrolyte"
    export_root = output_dir / "uma_torchsim_screen" / "export2pt" / "electrolyte"
    exported_dirs: list[Path] = []

    cond_dirs = sorted([path for path in phase_root.iterdir() if path.is_dir()] if phase_root.exists() else [])
    for cond_dir in cond_dirs:
        rep_dirs = sorted([path for path in cond_dir.iterdir() if path.is_dir() and path.name.startswith("replica_")])
        for rep_dir in rep_dirs:
            prod_h5 = rep_dir / "prod.h5md"
            prod_thermo = rep_dir / "prod_thermo.csv"
            if not (prod_h5.exists() and prod_thermo.exists()):
                continue

            out_dir = export_root / cond_dir.name / rep_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            traj_generic = out_dir / "prod.lammpstrj"
            eng_generic = out_dir / "prod.eng"
            data_generic = out_dir / "prod.data"
            export_h5md_to_lammps_dump(
                prod_h5,
                traj_generic,
                unwrap=True,
                include_ke_atom=True,
                lammps_template_style=True,
            )
            write_lammps_eng_from_thermo_csv(prod_thermo, eng_generic)
            write_lammps_data_full_from_structure(structure_path, data_generic)
            write_2pt_metadata(
                prod_thermo,
                out_dir / "2pt_metadata.json",
                timestep_ps=timestep_ps,
                dump_every_steps=dump_every_steps,
            )

            # Export both generic and ${sname}_* aliases for compatibility with existing 2PT scripts.
            shutil.copy2(traj_generic, out_dir / "prod.lammps")
            shutil.copy2(traj_generic, out_dir / f"{sname}_prod.lammpstrj")
            shutil.copy2(traj_generic, out_dir / f"{sname}_prod.lammps")
            shutil.copy2(eng_generic, out_dir / f"{sname}_prod.eng")
            shutil.copy2(data_generic, out_dir / f"{sname}_prod.data")

            exported_dirs.append(out_dir)

    if not exported_dirs:
        raise RuntimeError(
            "No production trajectories were found for export. Expected files like "
            "<output_dir>/uma_torchsim_screen/electrolyte/<condition_id>/replica_000/prod.h5md"
        )
    return exported_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-uma-water-2pt-experiment",
        description=(
            "Run a single UMA TorchSim electrolyte benchmark from CIF with NPT->NVT "
            "and export LAMMPS-style 2PT inputs (.lammps/.lammpstrj and .eng)."
        ),
    )
    parser.add_argument("--cif-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-id", type=str, default="water_298k_1atm")
    parser.add_argument("--sname", type=str, default="electrolyte")

    parser.add_argument("--model-name", type=str, default="uma-s-1p2")
    parser.add_argument("--task-name", type=str, default="omol")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")

    parser.add_argument("--temperature-k", type=float, default=298.0)
    parser.add_argument("--pressure-atm", type=float, default=1.0)
    parser.add_argument("--npt-ps", type=float, default=10.0)
    parser.add_argument("--nvt-ps", type=float, default=50.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--dump-every-steps", type=int, default=4)

    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-memory-scaler", type=float, default=None)
    parser.add_argument("--skip-batch-benchmark", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cif_path = args.cif_path.resolve()
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    output_dir = args.output_dir.resolve()
    run_root = output_dir / "uma_torchsim_screen"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest_path = run_root / "electrolyte_manifest.water_2pt.csv"
    _write_single_electrolyte_manifest(
        manifest_path,
        condition_id=args.condition_id,
        structure_path=cif_path,
        temperature_k=float(args.temperature_k),
        pressure_atm=float(args.pressure_atm),
        task_name=args.task_name,
        charge=int(args.charge),
        spin=int(args.spin),
    )

    timestep_ps = float(args.timestep_fs) / 1000.0
    npt_steps = _steps_from_duration_ps(float(args.npt_ps), float(args.timestep_fs))
    nvt_steps = _steps_from_duration_ps(float(args.nvt_ps), float(args.timestep_fs))

    config = default_config()
    config.electrode_manifest = None
    config.electrolyte_manifest = manifest_path
    config.electrode_root = None
    config.electrolyte_root = None
    config.output_dir = output_dir
    config.phase = "electrolyte"
    config.stages = ("md", "export-2pt")
    config.model_name = args.model_name
    config.device = args.device
    config.ensemble = "nvt"
    config.timestep_ps = timestep_ps
    config.dump_every_steps = int(args.dump_every_steps)
    config.retherm_steps_electrode = npt_steps
    config.retherm_steps_electrolyte = npt_steps
    config.prod_steps = nvt_steps
    config.replicas = 1
    config.base_seed = int(args.seed)
    config.compute_stress = True
    config.skip_batch_benchmark = bool(args.skip_batch_benchmark)
    config.max_memory_scaler = args.max_memory_scaler
    config.precision = args.precision
    config.debug = bool(args.debug)

    run_one_phase("electrolyte", config)
    exported_dirs = _export_lammps_artifacts(
        output_dir,
        sname=args.sname,
        structure_path=cif_path,
        timestep_ps=timestep_ps,
        dump_every_steps=int(args.dump_every_steps),
    )

    summary = {
        "cif_path": str(cif_path),
        "manifest_path": str(manifest_path),
        "model_name": args.model_name,
        "task_name": args.task_name,
        "temperature_k": float(args.temperature_k),
        "pressure_atm": float(args.pressure_atm),
        "pressure_mpa": float(args.pressure_atm) / ATM_PER_MPA,
        "npt_ps": float(args.npt_ps),
        "nvt_ps": float(args.nvt_ps),
        "timestep_fs": float(args.timestep_fs),
        "npt_steps": int(npt_steps),
        "nvt_steps": int(nvt_steps),
        "device": args.device,
        "precision": args.precision,
        "exported_replica_dirs": [str(path) for path in exported_dirs],
    }
    (run_root / "water_2pt_experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
