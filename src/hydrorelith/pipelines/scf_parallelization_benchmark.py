from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints


@dataclass(slots=True)
class ParallelConfig:
    config_id: str
    family: str
    mpi_ranks: int
    cpus_per_task: int
    omp_threads: int
    kpar: int
    ncore: int | None

    @property
    def active_cores(self) -> int:
        return self.mpi_ranks * self.omp_threads


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    system: str
    structure_source: str
    structure_name: str
    case_dir: Path
    config: ParallelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-scf-parallel-benchmark",
        description=(
            "Create and analyze Perlmutter CPU-node VASP SCF parallelization benchmarks "
            "for DIRECT-selected electrode and electrolyte structures."
        ),
    )
    parser.add_argument("--electrode-structures-dir", type=Path, default=None)
    parser.add_argument("--electrolyte-structures-dir", type=Path, default=None)
    parser.add_argument(
        "--template-input-root",
        type=Path,
        default=Path("for_chat_gpt/performance_analysis_input_files"),
        help="Directory containing electrode/ and electrolyte/ INCAR templates (and optional KPOINTS/POTCAR templates).",
    )
    parser.add_argument("--n-structures", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scf-steps", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/performance/scf_parallelization"),
    )
    parser.add_argument("--account", type=str, default="m4537")
    parser.add_argument("--qos", type=str, default="premium")
    parser.add_argument("--walltime", type=str, default="4:00:00")
    parser.add_argument("--job-name", type=str, default="vasp_parallel_bench")
    parser.add_argument("--module", type=str, default="vasp/6.4.3-cpu")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--benchmark-root", type=Path, default=None)
    return parser


def default_parallel_configs() -> list[ParallelConfig]:
    return [
        ParallelConfig("mpi32_ncore4_kpar1", "mpi", 32, 2, 1, 1, 4),
        ParallelConfig("mpi64_ncore8_kpar1", "mpi", 64, 2, 1, 1, 8),
        ParallelConfig("mpi96_ncore8_kpar1", "mpi", 96, 2, 1, 1, 8),
        ParallelConfig("mpi128_ncore8_kpar1", "mpi", 128, 2, 1, 1, 8),
        ParallelConfig("hybrid64x2_kpar1", "hybrid", 64, 4, 2, 1, None),
        ParallelConfig("hybrid32x4_kpar1", "hybrid", 32, 8, 4, 1, None),
        ParallelConfig("kpar2_mpi128", "kpar", 128, 2, 1, 2, 8),
        ParallelConfig("kpar4_mpi128", "kpar", 128, 2, 1, 4, 8),
        ParallelConfig("ncore4_mpi128", "ncore", 128, 2, 1, 1, 4),
        ParallelConfig("ncore16_mpi128", "ncore", 128, 2, 1, 1, 16),
    ]


def _resolve_best_training_set_root(path: Path) -> Path:
    if (path / "best_training_set").exists():
        return path / "best_training_set"
    if (path / "structure_generation" / "best_training_set").exists():
        return path / "structure_generation" / "best_training_set"
    raise FileNotFoundError(
        f"Could not find DIRECT-filtered structures under '{path}'. Expected best_training_set/ or structure_generation/best_training_set/."
    )


def _discover_structure_files(root: Path) -> list[Path]:
    files = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and (p.suffix.lower() == ".cif" or p.name.startswith("POSCAR"))
        ]
    )
    if not files:
        raise FileNotFoundError(f"No structure files found in {root}")
    return files


def _sample_structure_files(paths: list[Path], n_structures: int, seed: int) -> list[Path]:
    if n_structures <= 0:
        raise ValueError("--n-structures must be > 0")
    if len(paths) <= n_structures:
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, n_structures))


def _structure_name_from_path(path: Path) -> str:
    tokens = []
    for parent in path.parents:
        if parent.name.startswith("T_") or parent.name.startswith("lith_") or parent.name.startswith("LiOH_"):
            tokens.append(parent.name)
        if parent.name == "best_training_set":
            break
    tokens = list(reversed(tokens))
    tokens.append(path.stem)
    raw = "__".join(tokens)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _read_template_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _set_incar_tag(incar_text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$", re.MULTILINE)
    replacement = f"{key} = {value}"
    if pattern.search(incar_text):
        return pattern.sub(replacement, incar_text)
    return incar_text.rstrip() + f"\n{replacement}\n"


def _apply_electrode_magmom_tokens(incar_text: str, structure: Structure) -> str:
    counts = structure.composition.get_el_amt_dict()
    mapping = {
        "N_Li": str(int(round(float(counts.get("Li", 0.0))))),
        "N_Co": str(int(round(float(counts.get("Co", 0.0))))),
        "N_O": str(int(round(float(counts.get("O", 0.0))))),
    }
    for token, replacement in mapping.items():
        incar_text = re.sub(rf"\b{re.escape(token)}\b", replacement, incar_text)
    return incar_text


def _prepare_incar(
    template_text: str,
    structure: Structure,
    system: str,
    scf_steps: int,
    config: ParallelConfig,
) -> str:
    if scf_steps <= 0:
        raise ValueError("--scf-steps must be > 0")
    incar_text = template_text
    if system == "electrode":
        incar_text = _apply_electrode_magmom_tokens(incar_text, structure)
    incar_text = _set_incar_tag(incar_text, "NELM", str(scf_steps))
    incar_text = _set_incar_tag(incar_text, "KPAR", str(config.kpar))
    if config.omp_threads > 1:
        incar_text = _set_incar_tag(incar_text, "NCORE", "1")
    elif config.ncore is not None:
        incar_text = _set_incar_tag(incar_text, "NCORE", str(config.ncore))
    return incar_text


def _resolve_template_file(system_template_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = system_template_dir / name
        if candidate.exists():
            return candidate
    return None


def _write_kpoints(target_path: Path, template_path: Path | None) -> None:
    if template_path is not None:
        target_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        return
    kpoints = Kpoints.gamma_automatic(kpts=(1, 1, 1))
    kpoints.write_file(str(target_path))


def _build_cases(
    electrode_root: Path,
    electrolyte_root: Path,
    template_input_root: Path,
    output_dir: Path,
    n_structures: int,
    seed: int,
    scf_steps: int,
    configs: list[ParallelConfig],
) -> tuple[list[BenchmarkCase], dict[str, object]]:
    electrode_best = _resolve_best_training_set_root(electrode_root)
    electrolyte_best = _resolve_best_training_set_root(electrolyte_root)

    electrode_structures = _sample_structure_files(_discover_structure_files(electrode_best), n_structures, seed)
    electrolyte_structures = _sample_structure_files(_discover_structure_files(electrolyte_best), n_structures, seed + 1000)

    case_root = output_dir / "cases"
    case_root.mkdir(parents=True, exist_ok=True)

    templates = {
        "electrode": template_input_root / "electrode",
        "electrolyte": template_input_root / "electrolyte",
    }

    template_files: dict[str, dict[str, Path | None]] = {}
    for system, base in templates.items():
        template_files[system] = {
            "incar": _resolve_template_file(base, [f"INCAR_{system.upper()}", "INCAR"]),
            "kpoints": _resolve_template_file(base, [f"KPOINTS_{system.upper()}", "KPOINTS"]),
            "potcar": _resolve_template_file(base, [f"POTCAR_{system.upper()}", "POTCAR"]),
        }
        if template_files[system]["incar"] is None:
            raise FileNotFoundError(f"Missing INCAR template in {base}")

    cases: list[BenchmarkCase] = []
    missing_potcar_systems: list[str] = []
    if template_files["electrode"]["potcar"] is None:
        missing_potcar_systems.append("electrode")
    if template_files["electrolyte"]["potcar"] is None:
        missing_potcar_systems.append("electrolyte")

    all_selected = {
        "electrode": electrode_structures,
        "electrolyte": electrolyte_structures,
    }

    for system, selected_paths in all_selected.items():
        incar_template = _read_template_text(template_files[system]["incar"])  # type: ignore[arg-type]
        for structure_path in selected_paths:
            structure = Structure.from_file(str(structure_path))
            structure_name = _structure_name_from_path(structure_path)
            for config in configs:
                case_id = f"{system}__{structure_name}__{config.config_id}"
                case_dir = case_root / system / structure_name / config.config_id
                case_dir.mkdir(parents=True, exist_ok=True)

                prepared_incar = _prepare_incar(
                    template_text=incar_template,
                    structure=structure,
                    system=system,
                    scf_steps=scf_steps,
                    config=config,
                )
                (case_dir / "INCAR").write_text(prepared_incar, encoding="utf-8")
                structure.to(fmt="poscar", filename=str(case_dir / "POSCAR"))
                _write_kpoints(case_dir / "KPOINTS", template_files[system]["kpoints"])

                potcar_template = template_files[system]["potcar"]
                if potcar_template is not None:
                    (case_dir / "POTCAR").write_text(
                        potcar_template.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )

                case = BenchmarkCase(
                    case_id=case_id,
                    system=system,
                    structure_source=str(structure_path),
                    structure_name=structure_name,
                    case_dir=case_dir,
                    config=config,
                )
                (case_dir / "benchmark_case.json").write_text(
                    json.dumps(
                        {
                            "case_id": case.case_id,
                            "system": case.system,
                            "structure_source": case.structure_source,
                            "structure_name": case.structure_name,
                            "config": asdict(case.config),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                cases.append(case)

    plan = {
        "electrode_best_training_set": str(electrode_best),
        "electrolyte_best_training_set": str(electrolyte_best),
        "n_structures_per_system": n_structures,
        "selected_structures": {
            "electrode": [str(p) for p in electrode_structures],
            "electrolyte": [str(p) for p in electrolyte_structures],
        },
        "parallel_configs": [asdict(cfg) for cfg in configs],
        "missing_potcar_systems": missing_potcar_systems,
        "total_cases": len(cases),
    }
    return cases, plan


def _write_run_matrix(output_dir: Path, cases: list[BenchmarkCase]) -> Path:
    matrix_path = output_dir / "run_matrix.tsv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "case_id",
                "relative_case_dir",
                "system",
                "structure_name",
                "config_id",
                "family",
                "mpi_ranks",
                "cpus_per_task",
                "omp_threads",
                "active_cores",
                "kpar",
                "ncore",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.case_id,
                    str(case.case_dir.relative_to(output_dir)),
                    case.system,
                    case.structure_name,
                    case.config.config_id,
                    case.config.family,
                    case.config.mpi_ranks,
                    case.config.cpus_per_task,
                    case.config.omp_threads,
                    case.config.active_cores,
                    case.config.kpar,
                    "" if case.config.ncore is None else case.config.ncore,
                ]
            )
    return matrix_path


def _write_slurm_script(
    output_dir: Path,
    matrix_path: Path,
    account: str,
    qos: str,
    walltime: str,
    module_name: str,
    job_name: str,
) -> Path:
    script = f"""#!/bin/bash -l
#SBATCH -A {account}
#SBATCH -C cpu
#SBATCH -q {qos}
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t {walltime}
#SBATCH -J {job_name}
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

module load {module_name}
which vasp_std

export OMP_PLACES=threads
export OMP_PROC_BIND=spread

BENCH_ROOT=\"{output_dir}\"
RUN_MATRIX=\"{matrix_path}\"
RUN_LOG=\"$BENCH_ROOT/run_log.tsv\"

printf \"case_id\\tsystem\\tstructure\\tconfig_id\\tstatus\\telapsed_sec\\n\" > \"$RUN_LOG\"

while IFS=$'\\t' read -r case_id rel_dir system structure_name config_id family mpi_ranks cpus_per_task omp_threads active_cores kpar ncore; do
  if [[ \"$case_id\" == \"case_id\" ]]; then
    continue
  fi

  case_dir=\"$BENCH_ROOT/$rel_dir\"
  if [[ ! -d \"$case_dir\" ]]; then
    printf \"%s\\t%s\\t%s\\t%s\\tmissing_case_dir\\t\\n\" \"$case_id\" \"$system\" \"$structure_name\" \"$config_id\" >> \"$RUN_LOG\"
    continue
  fi

  if [[ ! -f \"$case_dir/POTCAR\" ]]; then
    printf \"%s\\t%s\\t%s\\t%s\\tmissing_potcar\\t\\n\" \"$case_id\" \"$system\" \"$structure_name\" \"$config_id\" >> \"$RUN_LOG\"
    continue
  fi

  export OMP_NUM_THREADS=$omp_threads
  pushd \"$case_dir\" > /dev/null
  srun -N 1 -n $mpi_ranks -c $cpus_per_task --cpu-bind=cores vasp_std > vasp_stdout.log 2> vasp_stderr.log

  elapsed_sec=$(python - <<'PY'
import re
from pathlib import Path
outcar = Path("OUTCAR")
if not outcar.exists():
    print("")
else:
    text = outcar.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Elapsed time \(sec\):\\s*([0-9.]+)", text)
    print(m.group(1) if m else "")
PY
)

  if [[ -n \"$elapsed_sec\" ]]; then
    status=ok
  else
    status=failed
  fi
  popd > /dev/null

  printf \"%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n\" \"$case_id\" \"$system\" \"$structure_name\" \"$config_id\" \"$status\" \"$elapsed_sec\" >> \"$RUN_LOG\"
done < \"$RUN_MATRIX\"

echo \"Benchmark runs complete. Analyze with:\"
echo \"python -m hydrorelith.pipelines.scf_parallelization_benchmark --analyze-only --benchmark-root $BENCH_ROOT\"
"""
    slurm_path = output_dir / "run_all_parallel_benchmarks.slurm"
    slurm_path.write_text(script, encoding="utf-8")
    return slurm_path


def _parse_vasp_elapsed_seconds(outcar_path: Path) -> float | None:
    if not outcar_path.exists():
        return None
    text = outcar_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Elapsed time \(sec\):\s*([0-9.]+)", text)
    if not match:
        return None
    return float(match.group(1))


def _parse_loop_times_seconds(outcar_path: Path) -> list[float]:
    if not outcar_path.exists():
        return []
    text = outcar_path.read_text(encoding="utf-8", errors="ignore")
    values: list[float] = []
    for match in re.finditer(r"LOOP:\s+cpu time\s+[0-9.]+:\s+real time\s+([0-9.]+)", text):
        values.append(float(match.group(1)))
    return values


def _parse_oszicar_scf_steps(oszicar_path: Path) -> int:
    if not oszicar_path.exists():
        return 0
    nsteps = 0
    for line in oszicar_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "DAV:" in line or "RMM:" in line:
            nsteps += 1
    return nsteps


def _load_case_metadata(case_json_path: Path) -> dict[str, object]:
    return json.loads(case_json_path.read_text(encoding="utf-8"))


def _collect_records(benchmark_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for case_json in sorted(benchmark_root.rglob("benchmark_case.json")):
        case_dir = case_json.parent
        metadata = _load_case_metadata(case_json)
        cfg = metadata["config"]

        outcar = case_dir / "OUTCAR"
        oszicar = case_dir / "OSZICAR"
        elapsed = _parse_vasp_elapsed_seconds(outcar)
        loop_times = _parse_loop_times_seconds(outcar)
        scf_steps = len(loop_times)
        if scf_steps == 0:
            scf_steps = _parse_oszicar_scf_steps(oszicar)

        sec_per_scf = None
        if elapsed is not None and scf_steps > 0:
            sec_per_scf = elapsed / float(scf_steps)

        records.append(
            {
                "case_id": metadata["case_id"],
                "system": metadata["system"],
                "structure_name": metadata["structure_name"],
                "structure_source": metadata["structure_source"],
                "config_id": cfg["config_id"],
                "family": cfg["family"],
                "mpi_ranks": int(cfg["mpi_ranks"]),
                "cpus_per_task": int(cfg["cpus_per_task"]),
                "omp_threads": int(cfg["omp_threads"]),
                "active_cores": int(cfg["active_cores"]),
                "kpar": int(cfg["kpar"]),
                "ncore": None if cfg["ncore"] is None else int(cfg["ncore"]),
                "elapsed_sec": elapsed,
                "scf_steps": scf_steps,
                "sec_per_scf": sec_per_scf,
                "status": "ok" if elapsed is not None else "missing_outcar",
                "case_dir": str(case_dir),
            }
        )
    return records


def _baseline_key(record: dict[str, object]) -> tuple[str, str]:
    return str(record["system"]), str(record["structure_name"])


def _compute_speedup_efficiency(records: list[dict[str, object]]) -> None:
    baseline_by_structure: dict[tuple[str, str], tuple[int, float]] = {}
    for rec in records:
        elapsed = rec["elapsed_sec"]
        if elapsed is None:
            continue
        key = _baseline_key(rec)
        cores = int(rec["active_cores"])
        is_scaling_family = str(rec["family"]) in {"mpi", "hybrid"}
        if not is_scaling_family:
            continue
        current = baseline_by_structure.get(key)
        if current is None or cores < current[0]:
            baseline_by_structure[key] = (cores, float(elapsed))

    for rec in records:
        rec["speedup"] = None
        rec["parallel_efficiency"] = None
        elapsed = rec["elapsed_sec"]
        if elapsed is None:
            continue
        baseline = baseline_by_structure.get(_baseline_key(rec))
        if baseline is None:
            continue
        base_cores, base_time = baseline
        cur_cores = int(rec["active_cores"])
        if cur_cores <= 0:
            continue
        speedup = float(base_time) / float(elapsed)
        efficiency = speedup / (float(cur_cores) / float(base_cores))
        rec["speedup"] = speedup
        rec["parallel_efficiency"] = efficiency


def _write_records_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "case_id",
        "system",
        "structure_name",
        "config_id",
        "family",
        "mpi_ranks",
        "cpus_per_task",
        "omp_threads",
        "active_cores",
        "kpar",
        "ncore",
        "elapsed_sec",
        "scf_steps",
        "sec_per_scf",
        "speedup",
        "parallel_efficiency",
        "status",
        "case_dir",
        "structure_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in headers})


def _average_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for rec in records:
        if rec["elapsed_sec"] is None:
            continue
        key = (str(rec["system"]), str(rec["config_id"]))
        grouped.setdefault(key, []).append(rec)

    avg_records: list[dict[str, object]] = []
    for (system, config_id), rows in grouped.items():
        elapsed_values = [float(r["elapsed_sec"]) for r in rows if r["elapsed_sec"] is not None]
        sec_per_scf_values = [float(r["sec_per_scf"]) for r in rows if r["sec_per_scf"] is not None]
        speedup_values = [float(r["speedup"]) for r in rows if r["speedup"] is not None]
        eff_values = [float(r["parallel_efficiency"]) for r in rows if r["parallel_efficiency"] is not None]

        first = rows[0]
        avg_records.append(
            {
                "system": system,
                "config_id": config_id,
                "family": first["family"],
                "mpi_ranks": first["mpi_ranks"],
                "cpus_per_task": first["cpus_per_task"],
                "omp_threads": first["omp_threads"],
                "active_cores": first["active_cores"],
                "kpar": first["kpar"],
                "ncore": first["ncore"],
                "n_structures": len(rows),
                "avg_elapsed_sec": sum(elapsed_values) / len(elapsed_values),
                "avg_sec_per_scf": (sum(sec_per_scf_values) / len(sec_per_scf_values)) if sec_per_scf_values else None,
                "avg_speedup": (sum(speedup_values) / len(speedup_values)) if speedup_values else None,
                "avg_parallel_efficiency": (sum(eff_values) / len(eff_values)) if eff_values else None,
            }
        )
    return sorted(avg_records, key=lambda x: (x["system"], x["active_cores"], x["config_id"]))


def _write_average_csv(path: Path, avg_records: list[dict[str, object]]) -> None:
    headers = [
        "system",
        "config_id",
        "family",
        "mpi_ranks",
        "cpus_per_task",
        "omp_threads",
        "active_cores",
        "kpar",
        "ncore",
        "n_structures",
        "avg_elapsed_sec",
        "avg_sec_per_scf",
        "avg_speedup",
        "avg_parallel_efficiency",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in avg_records:
            writer.writerow({k: row.get(k) for k in headers})


def _plot_mechanism_vs_cores(
    records: list[dict[str, object]],
    avg_records: list[dict[str, object]],
    system: str,
    output_dir: Path,
) -> None:
    filtered = [
        r
        for r in records
        if r["system"] == system and r["elapsed_sec"] is not None and r["family"] in {"mpi", "hybrid"}
    ]
    if not filtered:
        return

    plt.figure(figsize=(9, 6))
    structures = sorted({str(r["structure_name"]) for r in filtered})
    for name in structures:
        rows = sorted(
            [r for r in filtered if r["structure_name"] == name],
            key=lambda x: (int(x["active_cores"]), str(x["config_id"])),
        )
        x = [int(r["active_cores"]) for r in rows]
        y = [float(r["elapsed_sec"]) for r in rows]
        plt.plot(x, y, marker="o", alpha=0.35, linewidth=1)

    avg_filtered = [
        r for r in avg_records if r["system"] == system and r["family"] in {"mpi", "hybrid"}
    ]
    avg_filtered = sorted(avg_filtered, key=lambda x: (int(x["active_cores"]), str(x["config_id"])))
    x_avg = [int(r["active_cores"]) for r in avg_filtered]
    y_avg = [float(r["avg_elapsed_sec"]) for r in avg_filtered]
    plt.plot(x_avg, y_avg, marker="o", linewidth=2.5, color="black", label="average")

    plt.xlabel("Active cores (MPI ranks × OMP threads)")
    plt.ylabel("Elapsed time (s)")
    plt.title(f"{system.capitalize()} SCF benchmark: MPI/OMP vs active cores")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{system}_mpi_omp_vs_cores.png", dpi=200)
    plt.close()


def _plot_kpar_sweep(
    records: list[dict[str, object]],
    avg_records: list[dict[str, object]],
    system: str,
    output_dir: Path,
) -> None:
    filtered = [
        r
        for r in records
        if r["system"] == system
        and r["elapsed_sec"] is not None
        and int(r["active_cores"]) == 128
        and int(r["omp_threads"]) == 1
        and int(r["mpi_ranks"]) == 128
        and (str(r["family"]) == "kpar" or (str(r["family"]) == "mpi" and int(r["kpar"]) == 1))
    ]
    if not filtered:
        return

    kpar_values = sorted({int(r["kpar"]) for r in filtered})
    x_map = {k: i for i, k in enumerate(kpar_values)}

    plt.figure(figsize=(8, 5))
    structures = sorted({str(r["structure_name"]) for r in filtered})
    for name in structures:
        rows = sorted([r for r in filtered if r["structure_name"] == name], key=lambda x: int(x["kpar"]))
        x = [x_map[int(r["kpar"])] for r in rows]
        y = [float(r["elapsed_sec"]) for r in rows]
        plt.plot(x, y, marker="o", alpha=0.35, linewidth=1)

    avg_filtered = [
        r for r in avg_records if r["system"] == system and int(r["active_cores"]) == 128 and int(r["mpi_ranks"]) == 128
    ]
    avg_filtered = [
        r for r in avg_filtered if str(r["family"]) == "kpar" or (str(r["family"]) == "mpi" and int(r["kpar"]) == 1)
    ]
    avg_filtered = sorted(avg_filtered, key=lambda x: int(x["kpar"]))
    x_avg = [x_map[int(r["kpar"])] for r in avg_filtered]
    y_avg = [float(r["avg_elapsed_sec"]) for r in avg_filtered]
    plt.plot(x_avg, y_avg, marker="o", linewidth=2.5, color="black", label="average")

    plt.xticks(list(x_map.values()), [str(k) for k in kpar_values])
    plt.xlabel("KPAR")
    plt.ylabel("Elapsed time (s)")
    plt.title(f"{system.capitalize()} SCF benchmark: KPAR sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{system}_kpar_sweep.png", dpi=200)
    plt.close()


def _plot_ncore_sweep(
    records: list[dict[str, object]],
    avg_records: list[dict[str, object]],
    system: str,
    output_dir: Path,
) -> None:
    filtered = [
        r
        for r in records
        if r["system"] == system
        and r["elapsed_sec"] is not None
        and int(r["active_cores"]) == 128
        and int(r["omp_threads"]) == 1
        and int(r["kpar"]) == 1
        and r["ncore"] is not None
        and (str(r["family"]) == "ncore" or str(r["family"]) == "mpi")
    ]
    if not filtered:
        return

    ncore_values = sorted({int(r["ncore"]) for r in filtered if r["ncore"] is not None})
    x_map = {k: i for i, k in enumerate(ncore_values)}

    plt.figure(figsize=(8, 5))
    structures = sorted({str(r["structure_name"]) for r in filtered})
    for name in structures:
        rows = sorted(
            [r for r in filtered if r["structure_name"] == name and r["ncore"] is not None],
            key=lambda x: int(x["ncore"]),
        )
        x = [x_map[int(r["ncore"])] for r in rows]
        y = [float(r["elapsed_sec"]) for r in rows]
        plt.plot(x, y, marker="o", alpha=0.35, linewidth=1)

    avg_filtered = [
        r
        for r in avg_records
        if r["system"] == system
        and int(r["active_cores"]) == 128
        and int(r["omp_threads"]) == 1
        and int(r["kpar"]) == 1
        and r["ncore"] is not None
        and (str(r["family"]) == "ncore" or str(r["family"]) == "mpi")
    ]
    avg_filtered = sorted(avg_filtered, key=lambda x: int(x["ncore"]))
    x_avg = [x_map[int(r["ncore"])] for r in avg_filtered]
    y_avg = [float(r["avg_elapsed_sec"]) for r in avg_filtered]
    plt.plot(x_avg, y_avg, marker="o", linewidth=2.5, color="black", label="average")

    plt.xticks(list(x_map.values()), [str(v) for v in ncore_values])
    plt.xlabel("NCORE")
    plt.ylabel("Elapsed time (s)")
    plt.title(f"{system.capitalize()} SCF benchmark: NCORE sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{system}_ncore_sweep.png", dpi=200)
    plt.close()


def _plot_parallel_efficiency(avg_records: list[dict[str, object]], system: str, output_dir: Path) -> None:
    filtered = [
        r
        for r in avg_records
        if r["system"] == system and r["avg_parallel_efficiency"] is not None and r["family"] in {"mpi", "hybrid"}
    ]
    if not filtered:
        return

    filtered = sorted(filtered, key=lambda x: (int(x["active_cores"]), str(x["config_id"])))
    x = [int(r["active_cores"]) for r in filtered]
    y = [float(r["avg_parallel_efficiency"]) for r in filtered]
    labels = [str(r["config_id"]) for r in filtered]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    for xi, yi, label in zip(x, y, labels):
        plt.text(xi, yi, label, fontsize=7, ha="left", va="bottom")
    plt.xlabel("Active cores")
    plt.ylabel("Average parallel efficiency")
    plt.title(f"{system.capitalize()} SCF benchmark: parallel efficiency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{system}_parallel_efficiency.png", dpi=200)
    plt.close()


def _recommend_configs(avg_records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for system in {str(r["system"]) for r in avg_records}:
        rows = [r for r in avg_records if r["system"] == system and r["avg_elapsed_sec"] is not None]
        if not rows:
            continue
        fastest = min(rows, key=lambda x: float(x["avg_elapsed_sec"]))
        efficiency_rows = [
            r for r in rows if r["avg_parallel_efficiency"] is not None and r["family"] in {"mpi", "hybrid"}
        ]
        best_eff = max(efficiency_rows, key=lambda x: float(x["avg_parallel_efficiency"])) if efficiency_rows else None
        out[system] = {
            "fastest": fastest,
            "best_time_per_performance": best_eff if best_eff is not None else fastest,
        }
    return out


def analyze_benchmark_results(benchmark_root: Path) -> dict[str, object]:
    analysis_dir = benchmark_root / "analysis"
    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = _collect_records(benchmark_root)
    _compute_speedup_efficiency(records)
    _write_records_csv(analysis_dir / "records_per_case.csv", records)

    avg_records = _average_records(records)
    _write_average_csv(analysis_dir / "records_average.csv", avg_records)

    systems = sorted({str(r["system"]) for r in records})
    for system in systems:
        _plot_mechanism_vs_cores(records, avg_records, system, plot_dir)
        _plot_kpar_sweep(records, avg_records, system, plot_dir)
        _plot_ncore_sweep(records, avg_records, system, plot_dir)
        _plot_parallel_efficiency(avg_records, system, plot_dir)

    completed = sum(1 for r in records if r["elapsed_sec"] is not None)
    recommendations = _recommend_configs(avg_records)

    summary = {
        "benchmark_root": str(benchmark_root),
        "n_cases": len(records),
        "n_completed": completed,
        "completion_fraction": (completed / len(records)) if records else 0.0,
        "recommendations": recommendations,
        "analysis_files": {
            "records_per_case_csv": str(analysis_dir / "records_per_case.csv"),
            "records_average_csv": str(analysis_dir / "records_average.csv"),
            "plots_dir": str(plot_dir),
        },
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def generate_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.electrode_structures_dir is None or args.electrolyte_structures_dir is None:
        raise ValueError(
            "Both --electrode-structures-dir and --electrolyte-structures-dir are required unless --analyze-only is used."
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = default_parallel_configs()
    cases, plan = _build_cases(
        electrode_root=args.electrode_structures_dir,
        electrolyte_root=args.electrolyte_structures_dir,
        template_input_root=args.template_input_root,
        output_dir=output_dir,
        n_structures=args.n_structures,
        seed=args.seed,
        scf_steps=args.scf_steps,
        configs=configs,
    )

    plan["slurm"] = {
        "account": args.account,
        "qos": args.qos,
        "walltime": args.walltime,
        "job_name": args.job_name,
        "module": args.module,
        "perlmutter_cpu_node": {
            "nodes": 1,
            "ntasks_per_node": 128,
            "cpus_per_task": 2,
            "constraint": "cpu",
        },
    }

    matrix_path = _write_run_matrix(output_dir, cases)
    slurm_path = _write_slurm_script(
        output_dir=output_dir,
        matrix_path=matrix_path,
        account=args.account,
        qos=args.qos,
        walltime=args.walltime,
        module_name=args.module,
        job_name=args.job_name,
    )

    (output_dir / "benchmark_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "n_cases": len(cases),
        "run_matrix": str(matrix_path),
        "slurm_script": str(slurm_path),
        "missing_potcar_systems": plan.get("missing_potcar_systems", []),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.analyze_only:
        benchmark_root = args.benchmark_root or args.output_dir
        summary = analyze_benchmark_results(benchmark_root)
        print(json.dumps(summary, indent=2))
        return

    generation_summary = generate_benchmark(args)
    print(json.dumps(generation_summary, indent=2))


if __name__ == "__main__":
    main()
