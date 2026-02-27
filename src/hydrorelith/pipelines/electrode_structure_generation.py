from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from hydrorelith.config.defaults import default_electrode_generation_config
from hydrorelith.config.schemas import ElectrodeGenerationConfig

MATGL_RATTLING_ENABLED = False


@dataclass(slots=True)
class GeneratedStructure:
    structure: Structure
    lithiation_fraction: float
    temperature_k: int | None = None
    candidate_index: int = 0
    source_engine: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-electrode-generate",
        description="Scaffold for electrode structure generation workflow.",
    )
    parser.add_argument("--pristine-structure", type=Path, default=None)
    parser.add_argument("--mpid", type=str, default=None)
    parser.add_argument("--target-ion", type=str, default="Li")
    parser.add_argument("--supercell", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("--output-dir", type=Path, default=Path("electrode_structures"))
    parser.add_argument("--output-format", choices=["poscar", "cif"], default="poscar")
    parser.add_argument("--max-structures", type=int, default=600)
    parser.add_argument("--oversampling-factor", type=int, default=10)
    parser.add_argument("--min-lithiation-fraction", type=float, default=0.75)
    parser.add_argument("--lithiation-step", type=float, default=0.05)
    parser.add_argument("--max-removal-combinations-per-fraction", type=int, default=200)
    parser.add_argument("--rattle-method", choices=["mc", "gaussian", "phonon"], default="mc")
    parser.add_argument(
        "--rattle-engine",
        choices=["hiphive", "uma", "matgl", "all"],
        default="hiphive",
    )
    parser.add_argument(
        "--md-execution",
        choices=["run", "slurm"],
        default="run",
        help="Run MLFF MD immediately or only create SLURM scripts.",
    )
    parser.add_argument("--md-ensemble", choices=["nvt", "npt"], default="nvt")
    parser.add_argument("--md-timestep-fs", type=float, default=1.0)
    parser.add_argument("--md-steps", type=int, default=500)
    parser.add_argument("--md-sample-interval", type=int, default=10)
    parser.add_argument("--md-friction-per-fs", type=float, default=0.001)
    parser.add_argument("--uma-model-name", type=str, default="uma-s-1p1")
    parser.add_argument("--uma-task-id", type=str, default="omat")
    parser.add_argument(
        "--uma-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="UMA compute device (defaults to cuda when not provided).",
    )
    parser.add_argument(
        "--matgl-model-name",
        type=str,
        default="CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
        help="MatGL pretrained potential to use for MD-based rattling.",
    )
    parser.add_argument(
        "--matgl-backend",
        choices=["auto", "dgl", "pyg"],
        default="dgl",
        help="MatGL backend. Use DGL for CHGNet/QET models.",
    )
    parser.add_argument("--rattles-per-structure", type=int, default=1)
    parser.add_argument("--rattle-std-300k", type=float, default=0.01)
    parser.add_argument("--rattle-d-min", type=float, default=1.5)
    parser.add_argument("--rattle-n-iter", type=int, default=10)
    parser.add_argument("--max-base-structures-per-bin", type=int, default=25)
    parser.add_argument("--phonon-fc2-path", type=Path, default=None)
    parser.add_argument("--phonon-qm-statistics", action="store_true")
    parser.add_argument("--phonon-imag-freq-factor", type=float, default=1.0)
    parser.add_argument("--direct-threshold-init", type=float, default=0.05)
    parser.add_argument("--disable-direct-metric-plots", action="store_true")
    parser.add_argument("--slurm-generate-only", action="store_true")
    parser.add_argument("--slurm-dir", type=Path, default=None)
    parser.add_argument("--slurm-mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--slurm-queue", type=str, default="regular")
    parser.add_argument("--slurm-allocation", type=str, default=None)
    parser.add_argument("--slurm-time", type=str, default="1:00:00")
    parser.add_argument("--slurm-ntasks-per-node", type=int, default=128)
    parser.add_argument("--slurm-cpus-per-task", type=int, default=2)
    parser.add_argument("--slurm-gpus", type=int, default=1)
    parser.add_argument("--slurm-combined-jobs", action="store_true")
    parser.add_argument("--only-temperature", type=int, default=None)
    parser.add_argument("--only-lithiation-fraction", type=float, default=None)
    parser.add_argument("--target-rattle-count", type=int, default=None)
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Skip DIRECT selection and emit the full pre-DIRECT pool.",
    )
    parser.add_argument("--temperature-strategy", choices=["fixed", "auto"], default="fixed")
    parser.add_argument(
        "--temperatures",
        type=int,
        nargs="*",
        default=[250, 300, 600, 900, 1200],
    )
    parser.add_argument(
        "--pressures-mpa",
        type=float,
        nargs="*",
        default=None,
        help="Optional pressure values (MPa), matched by order with --temperatures.",
    )
    parser.add_argument("--auto-n-temperatures", type=int, default=5)
    parser.add_argument("--auto-include-300k", action="store_true")
    parser.add_argument("--auto-melting-margin", type=float, default=1.10)
    parser.add_argument(
        "--bootstrap-output-tree",
        action="store_true",
        help="Create output folder tree (T/lithiation) and exit.",
    )
    parser.add_argument(
        "--stop-after-delithiation",
        action="store_true",
        help="Run through pristine loading and delithiation generation only.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ElectrodeGenerationConfig:
    config = default_electrode_generation_config()
    config.source.pristine_structure_path = args.pristine_structure
    config.source.mpid = args.mpid
    config.source.target_ion = args.target_ion
    config.source.supercell = tuple(args.supercell)

    config.output.output_dir = args.output_dir
    config.output.output_format = args.output_format

    config.sampling.max_structures = args.max_structures
    config.sampling.oversampling_factor = args.oversampling_factor
    config.sampling.min_lithiation_fraction = args.min_lithiation_fraction
    config.sampling.lithiation_step = args.lithiation_step
    config.sampling.max_removal_combinations_per_fraction = args.max_removal_combinations_per_fraction
    config.sampling.rattle_engine = args.rattle_engine
    config.sampling.md_execution = args.md_execution
    config.sampling.md_ensemble = args.md_ensemble
    config.sampling.md_timestep_fs = args.md_timestep_fs
    config.sampling.md_steps = args.md_steps
    config.sampling.md_sample_interval = args.md_sample_interval
    config.sampling.md_friction_per_fs = args.md_friction_per_fs
    config.sampling.uma_model_name = args.uma_model_name
    config.sampling.uma_task_id = args.uma_task_id
    config.sampling.uma_device = args.uma_device or "cuda"
    config.sampling.matgl_model_name = args.matgl_model_name
    config.sampling.matgl_backend = args.matgl_backend
    config.sampling.rattle_method = args.rattle_method
    config.sampling.rattles_per_structure = args.rattles_per_structure
    config.sampling.rattle_std_300k = args.rattle_std_300k
    config.sampling.rattle_d_min = args.rattle_d_min
    config.sampling.rattle_n_iter = args.rattle_n_iter
    config.sampling.max_base_structures_per_bin = args.max_base_structures_per_bin
    config.sampling.phonon_fc2_path = args.phonon_fc2_path
    config.sampling.phonon_qm_statistics = bool(args.phonon_qm_statistics)
    config.sampling.phonon_imag_freq_factor = args.phonon_imag_freq_factor
    config.sampling.direct_threshold_init = args.direct_threshold_init
    config.sampling.direct_plot_metrics = not bool(args.disable_direct_metric_plots)

    config.temperature.strategy = args.temperature_strategy
    config.temperature.values = tuple(args.temperatures)
    if args.pressures_mpa is not None:
        if len(args.pressures_mpa) != len(args.temperatures):
            raise ValueError("--pressures-mpa must have same length as --temperatures")
        config.temperature.pressures_mpa = {
            temp: pressure
            for temp, pressure in zip(args.temperatures, args.pressures_mpa)
        }
    config.temperature.auto.n_points = args.auto_n_temperatures
    config.temperature.auto.include_300k = bool(args.auto_include_300k)
    config.temperature.auto.melting_temperature_margin = args.auto_melting_margin
    return config


class ElectrodeStructureGenerationPipeline:
    def __init__(self, config: ElectrodeGenerationConfig) -> None:
        self.config = config

    def run(self) -> None:
        self.validate_inputs()
        self.prepare_output_layout()
        base_structure = self.load_pristine_structure()
        delithiation_candidates = self.generate_delithiation_candidates(base_structure)
        target_temperatures = self.resolve_target_temperatures(base_structure)
        overview = self._build_generation_overview(
            base_structure=base_structure,
            delithiation_candidates=delithiation_candidates,
            temperatures=target_temperatures,
        )
        self._write_generation_overview(overview)
        if getattr(self, "stop_after_delithiation", False):
            self.write_structures(delithiation_candidates)
            return

        if getattr(self, "slurm_generate_only", False) and self.config.sampling.rattle_engine in {"uma", "matgl", "all"}:
            self.write_structures(delithiation_candidates)
            self.generate_slurm_files(target_temperatures, delithiation_candidates)
            return

        rattled_candidates = self.generate_rattled_candidates(delithiation_candidates, target_temperatures)
        if getattr(self, "skip_direct", False):
            output_structures = rattled_candidates
        else:
            output_structures = self.select_with_direct(rattled_candidates)
            if self.config.sampling.direct_plot_metrics:
                self.plot_direct_metrics(rattled_candidates, output_structures)
        self.write_structures(output_structures)

    def bootstrap_output_tree(self) -> None:
        self.prepare_output_layout()
        temperatures = list(self.config.temperature.values)
        lithiation_fractions = self._build_lithiation_grid()
        self._create_output_tree(
            temperatures=temperatures,
            lithiation_fractions=lithiation_fractions,
        )
        self._write_bootstrap_manifest(temperatures=temperatures, lithiation_fractions=lithiation_fractions)

    def validate_inputs(self) -> None:
        has_pristine = self.config.source.pristine_structure_path is not None
        has_mpid = self.config.source.mpid is not None
        if has_pristine == has_mpid:
            raise ValueError("Provide exactly one of --pristine-structure or --mpid")

        if self.config.source.pristine_structure_path is not None and not self.config.source.pristine_structure_path.exists():
            raise FileNotFoundError(f"Pristine structure not found: {self.config.source.pristine_structure_path}")

        if self.config.sampling.max_structures <= 0:
            raise ValueError("--max-structures must be > 0")

        if self.config.sampling.oversampling_factor <= 0:
            raise ValueError("--oversampling-factor must be > 0")

        if not (0.0 < self.config.sampling.min_lithiation_fraction <= 1.0):
            raise ValueError("--min-lithiation-fraction must be in (0, 1]")

        if not (0.0 < self.config.sampling.lithiation_step < 1.0):
            raise ValueError("--lithiation-step must be in (0, 1)")

        if self.config.sampling.max_removal_combinations_per_fraction <= 0:
            raise ValueError("--max-removal-combinations-per-fraction must be > 0")

        if self.config.sampling.rattles_per_structure <= 0:
            raise ValueError("--rattles-per-structure must be > 0")

        if self.config.sampling.rattle_std_300k <= 0.0:
            raise ValueError("--rattle-std-300k must be > 0")

        if self.config.sampling.rattle_d_min <= 0.0:
            raise ValueError("--rattle-d-min must be > 0")

        if self.config.sampling.rattle_n_iter <= 0:
            raise ValueError("--rattle-n-iter must be > 0")

        if self.config.sampling.max_base_structures_per_bin <= 0:
            raise ValueError("--max-base-structures-per-bin must be > 0")

        if self.config.sampling.phonon_imag_freq_factor <= 0.0:
            raise ValueError("--phonon-imag-freq-factor must be > 0")

        if self.config.sampling.md_timestep_fs <= 0:
            raise ValueError("--md-timestep-fs must be > 0")

        if self.config.sampling.md_steps <= 0:
            raise ValueError("--md-steps must be > 0")

        if self.config.sampling.md_sample_interval <= 0:
            raise ValueError("--md-sample-interval must be > 0")

        if self.config.sampling.direct_threshold_init <= 0:
            raise ValueError("--direct-threshold-init must be > 0")

        if (not MATGL_RATTLING_ENABLED) and self.config.sampling.rattle_engine == "matgl":
            raise RuntimeError(
                "MatGL rattling is temporarily disabled due environment compatibility constraints "
                "with fairchem-core>=2.15. Use --rattle-engine uma or --rattle-engine all "
                "(which currently runs hiPhive+UMA)."
            )

    def _active_rattle_engines(self) -> list[str]:
        engine = self.config.sampling.rattle_engine
        if engine == "all":
            return ["hiphive", "uma"] if not MATGL_RATTLING_ENABLED else ["hiphive", "uma", "matgl"]
        return [engine]

    def prepare_output_layout(self) -> None:
        self.config.output.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_lithiation_grid(self) -> list[float]:
        fractions = [1.0]
        current = 1.0
        while current - self.config.sampling.lithiation_step >= self.config.sampling.min_lithiation_fraction:
            current = round(current - self.config.sampling.lithiation_step, 6)
            fractions.append(current)
        if fractions[-1] > self.config.sampling.min_lithiation_fraction:
            fractions.append(round(self.config.sampling.min_lithiation_fraction, 6))
        return fractions

    def _create_output_tree(
        self,
        temperatures: list[int],
        lithiation_fractions: list[float],
    ) -> None:
        for temperature in temperatures:
            temp_dir = self.config.output.output_dir / f"T_{temperature}K"
            temp_dir.mkdir(parents=True, exist_ok=True)
            for fraction in lithiation_fractions:
                lith_dir = temp_dir / self._format_lithiation_dir(fraction)
                lith_dir.mkdir(parents=True, exist_ok=True)

    def _format_lithiation_dir(self, lithiation_fraction: float) -> str:
        lith_percent = 100.0 * lithiation_fraction
        return f"lith_{lith_percent:.2f}pct"

    def _write_bootstrap_manifest(
        self,
        temperatures: list[int],
        lithiation_fractions: list[float],
    ) -> None:
        manifest = {
            "source": {
                "mpid": self.config.source.mpid,
                "pristine_structure_path": (
                    str(self.config.source.pristine_structure_path)
                    if self.config.source.pristine_structure_path is not None
                    else None
                ),
                "target_ion": self.config.source.target_ion,
                "supercell": list(self.config.source.supercell),
            },
            "sampling": {
                "max_structures": self.config.sampling.max_structures,
                "oversampling_factor": self.config.sampling.oversampling_factor,
                "min_lithiation_fraction": self.config.sampling.min_lithiation_fraction,
                "lithiation_step": self.config.sampling.lithiation_step,
                "max_removal_combinations_per_fraction": self.config.sampling.max_removal_combinations_per_fraction,
                "rattle_engine": self.config.sampling.rattle_engine,
                "md_execution": self.config.sampling.md_execution,
                "md_ensemble": self.config.sampling.md_ensemble,
                "md_timestep_fs": self.config.sampling.md_timestep_fs,
                "md_steps": self.config.sampling.md_steps,
                "md_sample_interval": self.config.sampling.md_sample_interval,
                "md_friction_per_fs": self.config.sampling.md_friction_per_fs,
                "uma_model_name": self.config.sampling.uma_model_name,
                "uma_task_id": self.config.sampling.uma_task_id,
                "uma_device": self.config.sampling.uma_device,
                "matgl_model_name": self.config.sampling.matgl_model_name,
                "matgl_backend": self.config.sampling.matgl_backend,
                "rattle_method": self.config.sampling.rattle_method,
                "rattles_per_structure": self.config.sampling.rattles_per_structure,
                "rattle_std_300k": self.config.sampling.rattle_std_300k,
                "rattle_d_min": self.config.sampling.rattle_d_min,
                "rattle_n_iter": self.config.sampling.rattle_n_iter,
                "max_base_structures_per_bin": self.config.sampling.max_base_structures_per_bin,
                "phonon_fc2_path": (
                    str(self.config.sampling.phonon_fc2_path)
                    if self.config.sampling.phonon_fc2_path is not None
                    else None
                ),
                "phonon_qm_statistics": self.config.sampling.phonon_qm_statistics,
                "phonon_imag_freq_factor": self.config.sampling.phonon_imag_freq_factor,
                "direct_threshold_init": self.config.sampling.direct_threshold_init,
                "direct_plot_metrics": self.config.sampling.direct_plot_metrics,
            },
            "conditions": {
                "temperatures_k": temperatures,
                "pressures_mpa": self.config.temperature.pressures_mpa,
                "lithiation_fractions": lithiation_fractions,
            },
            "output": {
                "directory_layout": "T_<temp>K/lith_<percent>pct",
                "output_format": self.config.output.output_format,
            },
            "notes": [
                "Bootstrap mode creates directory and metadata scaffold only.",
                "For UMA MD with --md-ensemble npt, pressures_mpa are used as target external pressure per temperature (default 0.1 MPa if omitted).",
                "Without --skip-direct, this stage applies DIRECT selection to max_structures.",
            ],
        }

        manifest_path = self.config.output.output_dir / "structure_generation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load_pristine_structure(self) -> Structure:
        if self.config.source.pristine_structure_path is not None:
            structure = Structure.from_file(str(self.config.source.pristine_structure_path))
        else:
            mpid = self.config.source.mpid
            if mpid is None:
                raise ValueError("MPID is required when pristine structure path is not provided")
            api_key = os.environ.get("MP_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "MP_API_KEY is required to fetch structures from Materials Project"
                )
            with MPRester(api_key) as rester:
                structure = rester.get_structure_by_material_id(mpid)

        structure.make_supercell(self.config.source.supercell)
        return structure

    def resolve_target_temperatures(self, structure: Structure) -> list[int]:
        del structure
        if self.config.temperature.strategy == "fixed":
            return list(self.config.temperature.values)

        if not self.config.temperature.values:
            lower = 250
            upper = 1200
            n_points = max(self.config.temperature.auto.n_points, 2)
            points = [round(lower + idx * (upper - lower) / (n_points - 1)) for idx in range(n_points)]
        else:
            points = sorted({int(value) for value in self.config.temperature.values})

        if self.config.temperature.auto.include_300k and 300 not in points:
            points.append(300)
            points = sorted(points)
        return points

    def generate_delithiation_candidates(self, structure: Structure) -> list[GeneratedStructure]:
        ion_symbol = self.config.source.target_ion
        ion_indices = [idx for idx, site in enumerate(structure) if site.specie.symbol == ion_symbol]
        if not ion_indices:
            raise ValueError(f"No {ion_symbol} sites found in loaded structure")

        target_count = self.config.sampling.max_structures * self.config.sampling.oversampling_factor
        delithiation_targets = self._build_delithiation_targets(total_ion_sites=len(ion_indices))
        per_fraction_count = max(1, math.ceil(target_count / len(delithiation_targets)))
        base_seed = 11
        cap_per_fraction = self.config.sampling.max_removal_combinations_per_fraction

        generated: list[GeneratedStructure] = []
        for fraction_idx, (remove_count, actual_fraction) in enumerate(delithiation_targets):

            max_possible = math.comb(len(ion_indices), remove_count)
            n_requested = min(per_fraction_count, cap_per_fraction, max_possible)
            combo_list = self._sample_delithiation_combinations(
                ion_indices=ion_indices,
                remove_count=remove_count,
                n_requested=n_requested,
                seed=base_seed + fraction_idx * 100_003,
            )
            for combo_idx, combo in enumerate(combo_list):
                candidate = structure.copy()
                if combo:
                    candidate.remove_sites(list(sorted(combo, reverse=True)))
                generated.append(
                    GeneratedStructure(
                        structure=candidate,
                        lithiation_fraction=actual_fraction,
                        candidate_index=combo_idx,
                    )
                )
        return generated

    def _count_target_ion_sites(self, structure: Structure) -> int:
        ion_symbol = self.config.source.target_ion
        return sum(1 for site in structure if site.specie.symbol == ion_symbol)

    def _planned_engine_targets(self, total_target_count: int) -> dict[str, int]:
        engines = self._active_rattle_engines()
        if len(engines) > 1:
            return self._split_counts(engines, total_target_count)
        return {engines[0]: total_target_count}

    def _planned_bin_targets(
        self,
        delithiation_candidates: list[GeneratedStructure],
        temperatures: list[int],
        target_count: int,
    ) -> dict[tuple[int, float], int]:
        lithiation_levels = sorted({item.lithiation_fraction for item in delithiation_candidates}, reverse=True)
        bins = [(temperature, lithiation) for temperature in temperatures for lithiation in lithiation_levels]
        bins = self._apply_bin_filters(bins)
        per_bin_base = target_count // len(bins)
        remainder = target_count % len(bins)
        target_per_bin: dict[tuple[int, float], int] = {}
        for idx, key in enumerate(bins):
            target_per_bin[key] = per_bin_base + (1 if idx < remainder else 0)
        return target_per_bin

    def _build_generation_overview(
        self,
        base_structure: Structure,
        delithiation_candidates: list[GeneratedStructure],
        temperatures: list[int],
    ) -> dict:
        total_target_ions = self._count_target_ion_sites(base_structure)
        missing_li_counts: dict[int, int] = defaultdict(int)
        for item in delithiation_candidates:
            present = self._count_target_ion_sites(item.structure)
            missing = total_target_ions - present
            missing_li_counts[int(missing)] += 1

        target_total = getattr(self, "target_rattle_count", None)
        if target_total is None:
            target_total = self.config.sampling.max_structures * self.config.sampling.oversampling_factor

        engine_targets = self._planned_engine_targets(target_total)
        rattling_by_engine: dict[str, dict] = {}
        for engine_name, engine_target in engine_targets.items():
            bin_targets = self._planned_bin_targets(
                delithiation_candidates=delithiation_candidates,
                temperatures=temperatures,
                target_count=engine_target,
            )
            rattling_by_engine[engine_name] = {
                "target_structures": int(engine_target),
                "n_bins": int(len(bin_targets)),
                "per_bin": [
                    {
                        "temperature_k": int(t),
                        "lithiation_fraction": float(l),
                        "target_structures": int(n),
                    }
                    for (t, l), n in sorted(bin_targets.items(), key=lambda x: (x[0][0], x[0][1]))
                ],
            }

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_ion": self.config.source.target_ion,
            "n_target_ion_sites_pristine": int(total_target_ions),
            "delithiation": {
                "total_candidates": int(len(delithiation_candidates)),
                "unique_missing_li_levels": int(len(missing_li_counts)),
                "by_missing_li": [
                    {
                        "missing_li": int(missing),
                        "candidate_count": int(count),
                    }
                    for missing, count in sorted(missing_li_counts.items(), key=lambda x: x[0])
                ],
            },
            "rattling_plan": {
                "rattle_engine": self.config.sampling.rattle_engine,
                "md_execution": self.config.sampling.md_execution,
                "target_pool_structures": int(target_total),
                "engines": rattling_by_engine,
            },
            "temperatures_k": [int(t) for t in temperatures],
        }

    def _write_generation_overview(self, overview: dict) -> None:
        overview_path = self.config.output.output_dir / "generation_overview.json"
        overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")

        delith_total = overview["delithiation"]["total_candidates"]
        unique_missing = overview["delithiation"]["unique_missing_li_levels"]
        print(
            f"[overview] delithiation candidates={delith_total}, unique missing-Li levels={unique_missing}"
        )
        for engine_name, plan in overview["rattling_plan"]["engines"].items():
            print(
                f"[overview] {engine_name} target rattled structures={plan['target_structures']} over {plan['n_bins']} bins"
            )

    def _build_delithiation_targets(self, total_ion_sites: int) -> list[tuple[int, float]]:
        desired_fractions = self._build_lithiation_grid()
        unique_remove_counts: set[int] = set()
        targets: list[tuple[int, float]] = []
        for desired_fraction in desired_fractions:
            remove_count = int(round((1.0 - desired_fraction) * total_ion_sites))
            remove_count = max(0, min(remove_count, total_ion_sites))
            if remove_count in unique_remove_counts:
                continue
            unique_remove_counts.add(remove_count)
            actual_fraction = (total_ion_sites - remove_count) / total_ion_sites
            targets.append((remove_count, actual_fraction))
        return targets

    def _sample_delithiation_combinations(
        self,
        ion_indices: list[int],
        remove_count: int,
        n_requested: int,
        seed: int,
    ) -> list[tuple[int, ...]]:
        if n_requested <= 0:
            return []

        if remove_count == 0:
            return [tuple()]

        if remove_count == 1:
            return [(min(ion_indices),)]

        max_possible = math.comb(len(ion_indices), remove_count)
        if max_possible <= n_requested and max_possible <= 50_000:
            all_combos = [tuple(c) for c in combinations(sorted(ion_indices), remove_count)]
            return all_combos[:n_requested]

        rng = random.Random(seed)
        unique_combos: set[tuple[int, ...]] = set()
        attempts = 0
        max_attempts = max(1000, n_requested * 50)
        while len(unique_combos) < n_requested and attempts < max_attempts:
            sampled = tuple(sorted(rng.sample(ion_indices, remove_count)))
            unique_combos.add(sampled)
            attempts += 1

        combos = list(unique_combos)
        combos.sort()
        return combos[:n_requested]

    def generate_rattled_candidates(
        self,
        structures: list[GeneratedStructure],
        temperatures: list[int],
    ) -> list[GeneratedStructure]:
        target_count = getattr(self, "target_rattle_count", None)
        if target_count is None:
            target_count = self.config.sampling.max_structures * self.config.sampling.oversampling_factor
        engine = self.config.sampling.rattle_engine
        if engine == "all" and not MATGL_RATTLING_ENABLED:
            counts = self._split_counts(["hiphive", "uma"], target_count)
            combined: list[GeneratedStructure] = []
            combined.extend(self._generate_rattled_candidates_hiphive(structures, temperatures, counts["hiphive"]))
            combined.extend(self._generate_rattled_candidates_mlff_md(structures, temperatures, counts["uma"], backend="uma"))
            return combined

        if engine == "hiphive":
            return self._generate_rattled_candidates_hiphive(structures, temperatures, target_count)
        if engine == "uma":
            return self._generate_rattled_candidates_mlff_md(structures, temperatures, target_count, backend="uma")
        if engine == "matgl":
            return self._generate_rattled_candidates_mlff_md(structures, temperatures, target_count, backend="matgl")
        if engine == "all":
            counts = self._split_counts(["hiphive", "uma", "matgl"], target_count)
            combined: list[GeneratedStructure] = []
            combined.extend(self._generate_rattled_candidates_hiphive(structures, temperatures, counts["hiphive"]))
            combined.extend(self._generate_rattled_candidates_mlff_md(structures, temperatures, counts["uma"], backend="uma"))
            combined.extend(self._generate_rattled_candidates_mlff_md(structures, temperatures, counts["matgl"], backend="matgl"))
            return combined

        raise ValueError(f"Unsupported rattle engine: {engine}")

    def _generate_rattled_candidates_hiphive(
        self,
        structures: list[GeneratedStructure],
        temperatures: list[int],
        target_count: int,
    ) -> list[GeneratedStructure]:
        try:
            from hiphive.structure_generation import (
                generate_mc_rattled_structures,
                generate_phonon_rattled_structures,
                generate_rattled_structures,
            )
        except Exception as exc:
            raise RuntimeError(
                "hiPhive is required for the rattling stage. Install hiphive to continue."
            ) from exc

        adaptor = AseAtomsAdaptor()
        expanded: list[GeneratedStructure] = []
        by_lithiation: dict[float, list[GeneratedStructure]] = defaultdict(list)
        for item in structures:
            by_lithiation[item.lithiation_fraction].append(item)

        lithiation_levels = sorted(by_lithiation.keys(), reverse=True)
        bins = [(temperature, lithiation) for temperature in temperatures for lithiation in lithiation_levels]
        bins = self._apply_bin_filters(bins)
        per_bin_base = target_count // len(bins)
        remainder = target_count % len(bins)
        target_per_bin: dict[tuple[int, float], int] = {}
        for idx, key in enumerate(bins):
            target_per_bin[key] = per_bin_base + (1 if idx < remainder else 0)

        phonon_fc2 = None
        use_phonon = self.config.sampling.rattle_method == "phonon"
        if use_phonon:
            fc2_path = self.config.sampling.phonon_fc2_path
            if fc2_path is None or not fc2_path.exists():
                print(
                    "[WARN] rattle_method=phonon requested but --phonon-fc2-path missing/unavailable; falling back to MC-rattle."
                )
                use_phonon = False
            else:
                phonon_fc2 = np.load(fc2_path)

        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None

        for temperature in sorted({b[0] for b in bins}):
            for lithiation in sorted({b[1] for b in bins if b[0] == temperature}, reverse=True):
                n_bin = target_per_bin[(temperature, lithiation)]
                base_candidates = list(by_lithiation[lithiation])
                max_bases = min(self.config.sampling.max_base_structures_per_bin, len(base_candidates))
                rng = random.Random(2_000_003 * temperature + int(round(lithiation * 10000)))
                rng.shuffle(base_candidates)
                selected_bases = base_candidates[:max_bases]

                base_work = [n_bin // max_bases] * max_bases
                for idx in range(n_bin % max_bases):
                    base_work[idx] += 1

                progress_desc = f"T={temperature}K lith={lithiation*100:.2f}%"
                pbar = tqdm(total=n_bin, desc=progress_desc, leave=False) if tqdm else None

                for base_idx, base_item in enumerate(selected_bases):
                    n_structures = base_work[base_idx]
                    if n_structures <= 0:
                        continue

                    atoms = adaptor.get_atoms(base_item.structure)
                    seed = 1_000_003 * temperature + 101 * base_item.candidate_index + 17

                    if use_phonon and phonon_fc2 is not None:
                        rattled_atoms_list = generate_phonon_rattled_structures(
                            atoms=atoms,
                            fc2=phonon_fc2,
                            n_structures=n_structures,
                            temperature=float(temperature),
                            QM_statistics=self.config.sampling.phonon_qm_statistics,
                            imag_freq_factor=self.config.sampling.phonon_imag_freq_factor,
                        )
                    elif self.config.sampling.rattle_method == "gaussian":
                        scaled_std = self.config.sampling.rattle_std_300k * math.sqrt(temperature / 300.0)
                        rattled_atoms_list = generate_rattled_structures(
                            atoms=atoms,
                            n_structures=n_structures,
                            rattle_std=scaled_std,
                            seed=seed,
                        )
                    else:
                        scaled_std = self.config.sampling.rattle_std_300k * math.sqrt(temperature / 300.0)
                        rattled_atoms_list = generate_mc_rattled_structures(
                            atoms=atoms,
                            n_structures=n_structures,
                            rattle_std=scaled_std,
                            d_min=self.config.sampling.rattle_d_min,
                            seed=seed,
                            n_iter=self.config.sampling.rattle_n_iter,
                        )

                    for rattle_idx, rattled_atoms in enumerate(rattled_atoms_list):
                        expanded.append(
                            GeneratedStructure(
                                structure=adaptor.get_structure(rattled_atoms),
                                lithiation_fraction=lithiation,
                                temperature_k=temperature,
                                candidate_index=base_item.candidate_index * 10_000 + rattle_idx,
                                source_engine="hiphive",
                            )
                        )
                    if pbar is not None:
                        pbar.update(len(rattled_atoms_list))

                if pbar is not None:
                    pbar.close()
                else:
                    print(f"[progress] generated {n_bin} structures for {progress_desc}")

        if len(expanded) != target_count:
            raise RuntimeError(
                f"Generated {len(expanded)} rattled structures, expected exactly {target_count}"
            )
        return expanded

    def _generate_rattled_candidates_mlff_md(
        self,
        structures: list[GeneratedStructure],
        temperatures: list[int],
        target_count: int,
        backend: str,
    ) -> list[GeneratedStructure]:
        if self.config.sampling.md_execution != "run":
            raise RuntimeError(
                "MLFF MD generation requested but md_execution is not 'run'. Use --slurm-generate-only for script creation."
            )

        adaptor = AseAtomsAdaptor()
        expanded: list[GeneratedStructure] = []
        by_lithiation: dict[float, list[GeneratedStructure]] = defaultdict(list)
        for item in structures:
            by_lithiation[item.lithiation_fraction].append(item)

        lithiation_levels = sorted(by_lithiation.keys(), reverse=True)
        bins = [(temperature, lithiation) for temperature in temperatures for lithiation in lithiation_levels]
        bins = self._apply_bin_filters(bins)
        per_bin_base = target_count // len(bins)
        remainder = target_count % len(bins)
        target_per_bin: dict[tuple[int, float], int] = {}
        for idx, key in enumerate(bins):
            target_per_bin[key] = per_bin_base + (1 if idx < remainder else 0)

        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None

        pressure_per_bin: dict[tuple[int, float], float | None] | None = None
        if backend == "uma" and self.config.sampling.md_ensemble == "npt":
            pressure_per_bin = {
                (temperature, lithiation): self._pressure_mpa_for_temperature(temperature)
                for (temperature, lithiation) in bins
            }

        self._init_md_runtime_stats(
            backend=backend,
            target_count=target_count,
            target_per_bin=target_per_bin,
            pressure_per_bin=pressure_per_bin,
        )

        for temperature in sorted({b[0] for b in bins}):
            for lithiation in sorted({b[1] for b in bins if b[0] == temperature}, reverse=True):
                n_bin = target_per_bin[(temperature, lithiation)]
                base_candidates = list(by_lithiation[lithiation])
                max_bases = min(self.config.sampling.max_base_structures_per_bin, len(base_candidates))
                rng = random.Random(9_000_001 * temperature + int(round(lithiation * 10000)))
                rng.shuffle(base_candidates)
                selected_bases = base_candidates[:max_bases]

                base_work = [n_bin // max_bases] * max_bases
                for idx in range(n_bin % max_bases):
                    base_work[idx] += 1

                progress_desc = f"{backend.upper()} MD T={temperature}K lith={lithiation*100:.2f}%"
                pbar = tqdm(total=n_bin, desc=progress_desc, leave=False) if tqdm else None

                for base_idx, base_item in enumerate(selected_bases):
                    n_structures = base_work[base_idx]
                    if n_structures <= 0:
                        continue

                    atoms = adaptor.get_atoms(base_item.structure)
                    seed = 7_000_001 * temperature + 151 * base_item.candidate_index + 19

                    def _progress_callback(stage: str, captured: int, md_steps_done: int, md_steps_total: int) -> None:
                        self._update_md_runtime_stats(
                            backend=backend,
                            temperature=temperature,
                            lithiation=lithiation,
                            stage=stage,
                            completed_delta=0,
                            base_index=base_idx + 1,
                            base_total=max_bases,
                            md_steps_done=md_steps_done,
                            md_steps_total=md_steps_total,
                            snapshots_captured=captured,
                        )

                    self._update_md_runtime_stats(
                        backend=backend,
                        temperature=temperature,
                        lithiation=lithiation,
                        stage="base_start",
                        completed_delta=0,
                        base_index=base_idx + 1,
                        base_total=max_bases,
                        md_steps_done=0,
                        md_steps_total=0,
                        snapshots_captured=0,
                    )

                    try:
                        if backend == "uma":
                            pressure_mpa = (
                                self._pressure_mpa_for_temperature(temperature)
                                if self.config.sampling.md_ensemble == "npt"
                                else None
                            )
                            snapshots = self._run_uma_md_snapshots(
                                atoms,
                                temperature,
                                n_structures,
                                seed,
                                pressure_mpa=pressure_mpa,
                                progress_callback=_progress_callback,
                            )
                        elif backend == "matgl":
                            snapshots = self._run_matgl_md_snapshots(
                                atoms,
                                temperature,
                                n_structures,
                                seed,
                                progress_callback=_progress_callback,
                            )
                        else:
                            raise ValueError(f"Unsupported MLFF backend: {backend}")
                    except Exception as exc:
                        self._fail_md_runtime_stats(
                            backend=backend,
                            error_message=str(exc),
                            temperature=temperature,
                            lithiation=lithiation,
                        )
                        raise

                    for snap_idx, snap_atoms in enumerate(snapshots):
                        expanded.append(
                            GeneratedStructure(
                                structure=adaptor.get_structure(snap_atoms),
                                lithiation_fraction=lithiation,
                                temperature_k=temperature,
                                candidate_index=base_item.candidate_index * 100_000 + snap_idx,
                                source_engine=backend,
                            )
                        )
                    if pbar is not None:
                        pbar.update(len(snapshots))

                    self._update_md_runtime_stats(
                        backend=backend,
                        temperature=temperature,
                        lithiation=lithiation,
                        stage="base_complete",
                        completed_delta=len(snapshots),
                        base_index=base_idx + 1,
                        base_total=max_bases,
                        md_steps_done=self.config.sampling.md_steps,
                        md_steps_total=self.config.sampling.md_steps,
                        snapshots_captured=len(snapshots),
                    )

                if pbar is not None:
                    pbar.close()
                else:
                    print(f"[progress] generated {n_bin} {backend} MD structures for {progress_desc}")

        self._finalize_md_runtime_stats(backend=backend, final_count=len(expanded), target_count=target_count)

        if len(expanded) != target_count:
            raise RuntimeError(
                f"Generated {len(expanded)} structures from {backend} MD, expected exactly {target_count}"
            )
        return expanded

    def _apply_bin_filters(self, bins: list[tuple[int, float]]) -> list[tuple[int, float]]:
        filtered = bins
        only_t = getattr(self, "only_temperature", None)
        only_l = getattr(self, "only_lithiation_fraction", None)
        if only_t is not None:
            filtered = [b for b in filtered if b[0] == only_t]
        if only_l is not None:
            filtered = [b for b in filtered if abs(b[1] - only_l) <= 1e-6]
        if not filtered:
            raise ValueError("Bin filters produced no temperature/lithiation combinations.")
        return filtered

    def _pressure_mpa_for_temperature(self, temperature: int) -> float:
        pressures = self.config.temperature.pressures_mpa
        if not pressures:
            return 0.1
        if temperature not in pressures:
            raise ValueError(
                f"No pressure provided for temperature {temperature} K. "
                "Provide --pressures-mpa aligned with --temperatures for NPT runs."
            )
        return float(pressures[temperature])

    def _run_uma_md_snapshots(
        self,
        atoms,
        temperature: int,
        n_structures: int,
        seed: int,
        pressure_mpa: float | None = None,
        progress_callback=None,
    ):
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.nptberendsen import NPTBerendsen
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        if not hasattr(self, "_uma_predictor"):
            self._uma_predictor = pretrained_mlip.get_predict_unit(
                self.config.sampling.uma_model_name,
                device=self.config.sampling.uma_device,
            )

        atoms = atoms.copy()
        atoms.calc = FAIRChemCalculator(self._uma_predictor, task_name=self.config.sampling.uma_task_id)
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(temperature), rng=np.random.default_rng(seed))
        Stationary(atoms)

        timestep = self.config.sampling.md_timestep_fs * units.fs
        if self.config.sampling.md_ensemble == "npt":
            target_pressure_mpa = float(
                pressure_mpa if pressure_mpa is not None else self._pressure_mpa_for_temperature(temperature)
            )
            target_pressure_au = target_pressure_mpa * 1e6 * units.Pascal
            dyn = NPTBerendsen(
                atoms,
                timestep=timestep,
                temperature_K=float(temperature),
                pressure_au=target_pressure_au,
                taut=25.0 * units.fs,
                taup=75.0 * units.fs,
                compressibility=4.57e-5,
            )
        else:
            dyn = Langevin(
                atoms,
                timestep=timestep,
                temperature_K=float(temperature),
                friction=self.config.sampling.md_friction_per_fs / units.fs,
            )

        snapshots = []

        def _capture():
            snapshots.append(atoms.copy())

        dyn.attach(_capture, interval=self.config.sampling.md_sample_interval)
        n_steps = max(self.config.sampling.md_steps, n_structures * self.config.sampling.md_sample_interval)

        if progress_callback is not None:
            sample_interval = self.config.sampling.md_sample_interval

            def _progress_hook():
                captured = len(snapshots)
                done_steps = min(n_steps, captured * sample_interval)
                progress_callback(
                    stage="running",
                    captured=captured,
                    md_steps_done=done_steps,
                    md_steps_total=n_steps,
                )

            dyn.attach(_progress_hook, interval=sample_interval)

        dyn.run(steps=n_steps)
        return self._select_even_snapshots(snapshots, n_structures)

    def _run_matgl_md_snapshots(self, atoms, temperature: int, n_structures: int, seed: int, progress_callback=None):
        import matgl
        from ase.io.trajectory import Trajectory

        self._ensure_matgl_backend(matgl)
        backend_setting = self.config.sampling.matgl_backend.lower()
        if backend_setting == "auto":
            model_name = self.config.sampling.matgl_model_name
            backend_setting = "dgl" if any(
                token in model_name
                for token in ("CHGNet", "QET", "TensorNetDGL")
            ) else "pyg"

        if backend_setting == "dgl":
            from matgl.ext._ase_dgl import MolecularDynamics
        else:
            from matgl.ext.ase import MolecularDynamics

        potential = self._load_matgl_potential(matgl)

        del seed
        tmp_dir = self.config.output.output_dir / "_tmp_md"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        token = random.randint(10_000, 99_999)
        traj_path = tmp_dir / f"matgl_T{temperature}_{token}.traj"
        log_path = tmp_dir / f"matgl_T{temperature}_{token}.log"

        n_steps = max(self.config.sampling.md_steps, n_structures * self.config.sampling.md_sample_interval)
        md = MolecularDynamics(
            atoms=atoms.copy(),
            potential=potential,
            ensemble=self.config.sampling.md_ensemble,
            temperature=float(temperature),
            timestep=float(self.config.sampling.md_timestep_fs),
            friction=float(self.config.sampling.md_friction_per_fs),
            trajectory=str(traj_path),
            logfile=str(log_path),
            loginterval=int(self.config.sampling.md_sample_interval),
        )
        if progress_callback is None:
            md.run(steps=n_steps)
        else:
            done_steps = 0
            chunk = max(1, int(self.config.sampling.md_sample_interval))
            while done_steps < n_steps:
                run_steps = min(chunk, n_steps - done_steps)
                md.run(steps=run_steps)
                done_steps += run_steps
                captured = done_steps // chunk
                progress_callback(
                    stage="running",
                    captured=captured,
                    md_steps_done=done_steps,
                    md_steps_total=n_steps,
                )

        with Trajectory(str(traj_path)) as traj:
            snapshots = [frame.copy() for frame in traj]

        return self._select_even_snapshots(snapshots, n_structures)

    def _ensure_matgl_backend(self, matgl_module) -> None:
        backend_setting = self.config.sampling.matgl_backend.lower()
        model_name = self.config.sampling.matgl_model_name

        if backend_setting == "auto":
            backend_setting = "dgl" if any(
                token in model_name
                for token in ("CHGNet", "QET", "TensorNetDGL")
            ) else "pyg"

        matgl_module.set_backend(backend_setting.upper())

    def _load_matgl_potential(self, matgl_module):
        if hasattr(self, "_matgl_potential") and self._matgl_potential is not None:
            return self._matgl_potential

        model_name = self.config.sampling.matgl_model_name
        try:
            self._matgl_potential = matgl_module.load_model(model_name)
            return self._matgl_potential
        except Exception as exc:
            message = str(exc)
            if "Bad serialized model or bad model name" in message:
                raise RuntimeError(
                    f"Failed to load MatGL model '{model_name}'. Ensure the model name is valid, "
                    "the selected backend matches the model (DGL required for CHGNet/QET), and clear stale cache with "
                    "`python -c 'import matgl; matgl.clear_cache()'` if needed."
                ) from exc
            raise

    def _select_even_snapshots(self, snapshots, n_keep: int):
        if len(snapshots) <= n_keep:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, n_keep, dtype=int)
        return [snapshots[i] for i in indices]

    def _split_counts(self, keys: list[str], total: int) -> dict[str, int]:
        base = total // len(keys)
        rem = total % len(keys)
        out = {key: base for key in keys}
        for idx, key in enumerate(keys):
            if idx < rem:
                out[key] += 1
        return out

    def _md_stats_dir(self) -> Path:
        stats_dir = self.config.output.output_dir / "md_runtime_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        return stats_dir

    def _md_stats_path(self, backend: str) -> Path:
        return self._md_stats_dir() / f"md_progress_{backend}.json"

    def _init_md_runtime_stats(
        self,
        backend: str,
        target_count: int,
        target_per_bin: dict[tuple[int, float], int],
        pressure_per_bin: dict[tuple[int, float], float | None] | None = None,
    ) -> None:
        now = time.time()
        self._md_runtime_state = getattr(self, "_md_runtime_state", {})
        self._md_runtime_state[backend] = {
            "backend": backend,
            "status": "running",
            "error_message": None,
            "start_epoch_s": now,
            "last_update_epoch_s": now,
            "target_structures": int(target_count),
            "completed_structures": 0,
            "rate_structures_per_min": 0.0,
            "eta_seconds": None,
            "eta_utc": None,
            "bins": {
                f"T{t}_lith_{l:.6f}": {
                    "temperature_k": int(t),
                    "lithiation_fraction": float(l),
                    "pressure_mpa": (
                        float(pressure_per_bin[(t, l)])
                        if pressure_per_bin is not None and pressure_per_bin.get((t, l)) is not None
                        else None
                    ),
                    "target_structures": int(n),
                    "completed_structures": 0,
                    "base_index": 0,
                    "base_total": 0,
                    "md_steps_done": 0,
                    "md_steps_total": 0,
                    "snapshots_captured": 0,
                    "stage": "pending",
                    "error_message": None,
                }
                for (t, l), n in sorted(target_per_bin.items(), key=lambda x: (x[0][0], x[0][1]))
            },
        }
        self._write_md_runtime_stats(backend)

    def _update_md_runtime_stats(
        self,
        backend: str,
        temperature: int,
        lithiation: float,
        stage: str,
        completed_delta: int,
        base_index: int,
        base_total: int,
        md_steps_done: int,
        md_steps_total: int,
        snapshots_captured: int,
    ) -> None:
        state = getattr(self, "_md_runtime_state", {}).get(backend)
        if state is None:
            return

        key = f"T{temperature}_lith_{lithiation:.6f}"
        bin_state = state["bins"].get(key)
        if bin_state is None:
            return

        if completed_delta:
            state["completed_structures"] += int(completed_delta)
            bin_state["completed_structures"] += int(completed_delta)

        bin_state["base_index"] = int(base_index)
        bin_state["base_total"] = int(base_total)
        bin_state["md_steps_done"] = int(md_steps_done)
        bin_state["md_steps_total"] = int(md_steps_total)
        bin_state["snapshots_captured"] = int(snapshots_captured)
        bin_state["stage"] = stage
        bin_state["error_message"] = None

        now = time.time()
        elapsed = max(now - state["start_epoch_s"], 1e-9)
        completed = int(state["completed_structures"])
        target = int(state["target_structures"])
        rate_per_sec = completed / elapsed
        state["rate_structures_per_min"] = float(rate_per_sec * 60.0)
        remaining = max(target - completed, 0)
        if rate_per_sec > 0 and remaining > 0:
            eta_seconds = remaining / rate_per_sec
            eta_epoch = now + eta_seconds
            state["eta_seconds"] = float(eta_seconds)
            state["eta_utc"] = datetime.fromtimestamp(eta_epoch, tz=timezone.utc).isoformat()
        elif remaining == 0:
            state["eta_seconds"] = 0.0
            state["eta_utc"] = datetime.now(timezone.utc).isoformat()
        else:
            state["eta_seconds"] = None
            state["eta_utc"] = None

        state["last_update_epoch_s"] = now
        state["last_update_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_md_runtime_stats(backend)

    def _finalize_md_runtime_stats(self, backend: str, final_count: int, target_count: int) -> None:
        state = getattr(self, "_md_runtime_state", {}).get(backend)
        if state is None:
            return
        state["completed_structures"] = int(final_count)
        state["target_structures"] = int(target_count)
        state["status"] = "completed"
        state["eta_seconds"] = 0.0
        state["eta_utc"] = datetime.now(timezone.utc).isoformat()
        state["last_update_epoch_s"] = time.time()
        state["last_update_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_md_runtime_stats(backend)

    def _fail_md_runtime_stats(
        self,
        backend: str,
        error_message: str,
        temperature: int | None = None,
        lithiation: float | None = None,
    ) -> None:
        state = getattr(self, "_md_runtime_state", {}).get(backend)
        if state is None:
            return

        state["status"] = "failed"
        state["error_message"] = str(error_message)
        state["eta_seconds"] = None
        state["eta_utc"] = None
        state["last_update_epoch_s"] = time.time()
        state["last_update_utc"] = datetime.now(timezone.utc).isoformat()

        if temperature is not None and lithiation is not None:
            key = f"T{temperature}_lith_{lithiation:.6f}"
            bin_state = state.get("bins", {}).get(key)
            if bin_state is not None:
                bin_state["stage"] = "failed"
                bin_state["error_message"] = str(error_message)

        self._write_md_runtime_stats(backend)

    def _write_md_runtime_stats(self, backend: str) -> None:
        state = getattr(self, "_md_runtime_state", {}).get(backend)
        if state is None:
            return
        path = self._md_stats_path(backend)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def generate_slurm_files(self, temperatures: list[int], delithiation_candidates: list[GeneratedStructure]) -> None:
        slurm_dir = getattr(self, "slurm_dir", None) or (self.config.output.output_dir / "slurm_jobs")
        slurm_dir.mkdir(parents=True, exist_ok=True)

        engines = self._active_rattle_engines()

        md_engines = [engine for engine in engines if engine in {"uma", "matgl"}]

        allocation = getattr(self, "slurm_allocation", None)
        mode = getattr(self, "slurm_mode", "cpu")
        if allocation is None:
            allocation = "m4537_g" if mode == "gpu" else "m4537"

        lith_levels = sorted({item.lithiation_fraction for item in delithiation_candidates}, reverse=True)
        bins = [(t, l) for t in temperatures for l in lith_levels]
        target_total = getattr(self, "target_rattle_count", None)
        if target_total is None:
            target_total = self.config.sampling.max_structures * self.config.sampling.oversampling_factor
        per_bin = target_total // max(len(bins), 1)

        split_jobs = not bool(getattr(self, "slurm_combined_jobs", False))

        def _md_backend_args(engine_name: str) -> str:
            if engine_name == "uma":
                return (
                    f" --uma-model-name {self.config.sampling.uma_model_name}"
                    f" --uma-task-id {self.config.sampling.uma_task_id}"
                    f" --uma-device {self.config.sampling.uma_device}"
                )
            if engine_name == "matgl":
                return (
                    f" --matgl-model-name {self.config.sampling.matgl_model_name}"
                    f" --matgl-backend {self.config.sampling.matgl_backend}"
                )
            return ""

        if split_jobs:
            for engine in md_engines:
                for temperature, lith in bins:
                    lith_pct = 100.0 * lith
                    script_path = slurm_dir / f"run_{engine}_T{temperature}_lith_{lith_pct:.2f}.slurm"
                    header = self._render_slurm_header(job_name=f"{engine}_T{temperature}_L{lith_pct:.0f}")
                    cmd = (
                        "PYTHONPATH=src python -m hydrorelith.pipelines.electrode_structure_generation "
                        f"--mpid {self.config.source.mpid} --target-ion {self.config.source.target_ion} "
                        f"--output-dir {self.config.output.output_dir} --output-format {self.config.output.output_format} "
                        f"--max-structures {self.config.sampling.max_structures} --oversampling-factor {self.config.sampling.oversampling_factor} "
                        f"--min-lithiation-fraction {self.config.sampling.min_lithiation_fraction} --lithiation-step {self.config.sampling.lithiation_step} "
                        f"--max-removal-combinations-per-fraction {self.config.sampling.max_removal_combinations_per_fraction} "
                        f"--rattle-engine {engine} --md-execution run --skip-direct "
                        f"{_md_backend_args(engine)} "
                        f"--only-temperature {temperature} --only-lithiation-fraction {lith:.8f} "
                        f"--target-rattle-count {per_bin} --temperatures {' '.join(str(t) for t in temperatures)}"
                    )
                    script_path.write_text(f"{header}\n{cmd}\n", encoding="utf-8")
        else:
            for engine in md_engines:
                script_path = slurm_dir / f"run_{engine}_rattling.slurm"
                header = self._render_slurm_header(job_name=f"rattle_{engine}")
                cmd = (
                    "PYTHONPATH=src python -m hydrorelith.pipelines.electrode_structure_generation "
                    f"--mpid {self.config.source.mpid} --target-ion {self.config.source.target_ion} "
                    f"--output-dir {self.config.output.output_dir} --output-format {self.config.output.output_format} "
                    f"--max-structures {self.config.sampling.max_structures} --oversampling-factor {self.config.sampling.oversampling_factor} "
                    f"--min-lithiation-fraction {self.config.sampling.min_lithiation_fraction} --lithiation-step {self.config.sampling.lithiation_step} "
                    f"--max-removal-combinations-per-fraction {self.config.sampling.max_removal_combinations_per_fraction} "
                    f"--rattle-engine {engine} --md-execution run --skip-direct "
                    f"{_md_backend_args(engine)} "
                    f"--temperatures {' '.join(str(t) for t in temperatures)}"
                )
                script_path.write_text(f"{header}\n{cmd}\n", encoding="utf-8")

        plot_script = slurm_dir / "plot_direct_metrics.slurm"
        plot_header = self._render_slurm_header(job_name="plot_direct_metrics")
        plot_cmd = (
            "PYTHONPATH=src python -m hydrorelith.pipelines.electrode_structure_generation "
            f"--mpid {self.config.source.mpid} --target-ion {self.config.source.target_ion} "
            f"--output-dir {self.config.output.output_dir} --stop-after-delithiation"
        )
        plot_note = "# Replace command above with your post-MD plotting workflow if running MD jobs separately."
        plot_script.write_text(f"{plot_header}\n{plot_note}\n{plot_cmd}\n", encoding="utf-8")

    def _render_slurm_header(self, job_name: str) -> str:
        mode = getattr(self, "slurm_mode", "cpu")
        queue = getattr(self, "slurm_queue", "regular")
        allocation = getattr(self, "slurm_allocation", None)
        if allocation is None:
            allocation = "m4537_g" if mode == "gpu" else "m4537"
        time_limit = getattr(self, "slurm_time", "1:00:00")

        if mode == "gpu":
            gpus = getattr(self, "slurm_gpus", 1)
            return textwrap.dedent(
                f"""#!/bin/bash
#SBATCH -C gpu
#SBATCH -q {queue}
#SBATCH -A {allocation}
#SBATCH -N 1
#SBATCH -t {time_limit}
#SBATCH -J {job_name}
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --gpus {gpus}
"""
            )

        ntasks = getattr(self, "slurm_ntasks_per_node", 128)
        cpus = getattr(self, "slurm_cpus_per_task", 2)
        return textwrap.dedent(
            f"""#!/bin/bash
#SBATCH -C cpu
#SBATCH -q {queue}
#SBATCH -A {allocation}
#SBATCH -N 1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH -t {time_limit}
#SBATCH -J {job_name}
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
"""
        )

    def select_with_direct(self, structures: list[GeneratedStructure]) -> list[GeneratedStructure]:
        if len(structures) <= self.config.sampling.max_structures:
            return structures

        quotas = self._allocate_direct_quotas(structures)
        descriptors = [self._compute_descriptor(item) for item in structures]
        grouped_indices: dict[tuple[int, float], list[int]] = defaultdict(list)
        for idx, item in enumerate(structures):
            if item.temperature_k is None:
                continue
            grouped_indices[(item.temperature_k, item.lithiation_fraction)].append(idx)

        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None

        selected_indices: list[int] = []
        groups = sorted(grouped_indices.keys(), key=lambda x: (x[0], x[1]))
        iterator = tqdm(groups, desc="DIRECT selection", leave=False) if tqdm else groups
        for group in iterator:
            candidate_indices = grouped_indices[group]
            quota = min(quotas.get(group, 0), len(candidate_indices))
            if quota <= 0:
                continue
            local_vectors = np.array([descriptors[i] for i in candidate_indices], dtype=float)
            local_selected = self._run_maml_direct(local_vectors, quota)
            selected_indices.extend(candidate_indices[i] for i in local_selected)

        if len(selected_indices) < self.config.sampling.max_structures:
            selected_set = set(selected_indices)
            remaining = [idx for idx in range(len(structures)) if idx not in selected_set]
            needed = self.config.sampling.max_structures - len(selected_indices)
            if remaining:
                rem_vectors = np.array([descriptors[i] for i in remaining], dtype=float)
                fill_selected = self._greedy_maximin_indices(rem_vectors, min(needed, len(remaining)))
                selected_indices.extend(remaining[i] for i in fill_selected)

        selected_indices = selected_indices[: self.config.sampling.max_structures]
        return [structures[i] for i in selected_indices]

    def _run_maml_direct(self, vectors: np.ndarray, n_select: int) -> list[int]:
        if len(vectors) <= n_select:
            return list(range(len(vectors)))
        from maml.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters

        sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=n_select, threshold_init=self.config.sampling.direct_threshold_init),
            select_k_from_clusters=SelectKFromClusters(k=1),
        )
        result = sampler.fit_transform(vectors)
        selected = [int(i) for i in result.get("selected_indexes", [])]
        if len(selected) < n_select:
            remaining = [i for i in range(len(vectors)) if i not in set(selected)]
            fill = self._greedy_maximin_indices(vectors[remaining], min(n_select - len(selected), len(remaining)))
            selected.extend(remaining[i] for i in fill)
        return selected[:n_select]

    def plot_direct_metrics(
        self,
        all_structures: list[GeneratedStructure],
        selected_structures: list[GeneratedStructure],
    ) -> None:
        if not all_structures or not selected_structures:
            return

        all_features = np.array([self._compute_descriptor(item) for item in all_structures], dtype=float)
        selected_features = np.array([self._compute_descriptor(item) for item in selected_structures], dtype=float)

        selected_indexes: list[int] = []
        used = set()
        for sf in selected_features:
            candidates = np.where(np.all(np.isclose(all_features, sf, atol=1e-8), axis=1))[0]
            for idx in candidates:
                if idx not in used:
                    selected_indexes.append(int(idx))
                    used.add(int(idx))
                    break

        metrics_dir = self.config.output.output_dir / "direct_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        from sklearn.decomposition import PCA

        n_components = min(30, all_features.shape[0], all_features.shape[1])
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(all_features)
        explained_variance = pca.explained_variance_ratio_

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, n_components + 1), explained_variance[:n_components] * 100, "o-")
        plt.xlabel("i-th PC")
        plt.ylabel("Explained variance")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.tight_layout()
        plt.savefig(metrics_dir / "explained_variance.png", dpi=200)
        plt.close()

        weighted = pca_features[:, : min(7, pca_features.shape[1])]
        if weighted.shape[1] >= 2:
            plt.figure(figsize=(6, 6))
            plt.plot(weighted[:, 0], weighted[:, 1], "*", alpha=0.4, label=f"All {len(weighted):,} structures")
            sel = weighted[selected_indexes]
            plt.plot(sel[:, 0], sel[:, 1], "*", alpha=0.6, label=f"DIRECT selected {len(sel):,}")
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(metrics_dir / "pca_coverage_direct.png", dpi=200)
            plt.close()

            manual_selection_index = [int(i) for i in np.linspace(0, len(weighted) - 1, len(selected_indexes))]
            plt.figure(figsize=(6, 6))
            plt.plot(weighted[:, 0], weighted[:, 1], "*", alpha=0.4, label=f"All {len(weighted):,} structures")
            man = weighted[manual_selection_index]
            plt.plot(man[:, 0], man[:, 1], "*", alpha=0.6, label=f"Manual selected {len(man):,}")
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(metrics_dir / "pca_coverage_manual.png", dpi=200)
            plt.close()

        n_pcs_score = min(7, weighted.shape[1])
        scores_direct = [
            self._coverage_score(weighted[:, i], selected_indexes, n_bins=100)
            for i in range(n_pcs_score)
        ]
        manual_selection_index = [int(i) for i in np.linspace(0, len(weighted) - 1, len(selected_indexes))]
        scores_manual = [
            self._coverage_score(weighted[:, i], manual_selection_index, n_bins=100)
            for i in range(n_pcs_score)
        ]

        x = np.arange(n_pcs_score)
        plt.figure(figsize=(10, 4))
        plt.bar(
            x + 0.2,
            scores_direct,
            width=0.35,
            label=f"DIRECT (mean={np.mean(scores_direct):.3f})",
        )
        plt.bar(
            x - 0.2,
            scores_manual,
            width=0.35,
            label=f"Manual (mean={np.mean(scores_manual):.3f})",
        )
        plt.xticks(x, [f"PC {i+1}" for i in range(n_pcs_score)])
        plt.ylim(0, 1.05)
        plt.ylabel("Coverage score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(metrics_dir / "coverage_scores.png", dpi=200)
        plt.close()

    def _coverage_score(self, values: np.ndarray, selected_indexes: list[int], n_bins: int = 100) -> float:
        selected_values = values[selected_indexes]
        bins = np.linspace(float(np.min(values)), float(np.max(values)), n_bins)
        n_all = np.count_nonzero(np.histogram(values, bins=bins)[0])
        n_sel = np.count_nonzero(np.histogram(selected_values, bins=bins)[0])
        if n_all == 0:
            return 0.0
        return n_sel / n_all

    def _allocate_direct_quotas(self, structures: list[GeneratedStructure]) -> dict[tuple[int, float], int]:
        grouped_counts: dict[tuple[int, float], int] = defaultdict(int)
        for item in structures:
            if item.temperature_k is None:
                continue
            grouped_counts[(item.temperature_k, item.lithiation_fraction)] += 1

        if not grouped_counts:
            return {}

        temp_weights_cfg = self.config.ratios.retention.temperature_weights
        lith_weights_cfg = self.config.ratios.retention.lithiation_bin_weights

        temps = sorted({key[0] for key in grouped_counts})
        lith_levels = sorted({key[1] for key in grouped_counts}, reverse=True)

        temp_weight_raw: dict[int, float] = {}
        for t in temps:
            temp_weight_raw[t] = float(temp_weights_cfg.get(t, 1.0))
        temp_norm = sum(temp_weight_raw.values()) or float(len(temps))
        temp_weights = {t: w / temp_norm for t, w in temp_weight_raw.items()}

        lith_weight_raw: dict[float, float] = {}
        for l in lith_levels:
            lith_weight_raw[l] = self._lithiation_weight_from_bins(l, lith_weights_cfg)
        lith_norm = sum(lith_weight_raw.values()) or float(len(lith_levels))
        lith_weights = {l: w / lith_norm for l, w in lith_weight_raw.items()}

        target_total = self.config.sampling.max_structures
        desired: dict[tuple[int, float], float] = {}
        for key in grouped_counts:
            t, l = key
            desired[key] = target_total * temp_weights[t] * lith_weights[l]

        floor_alloc: dict[tuple[int, float], int] = {}
        for key, value in desired.items():
            floor_alloc[key] = min(int(math.floor(value)), grouped_counts[key])

        allocated = sum(floor_alloc.values())
        remaining = target_total - allocated

        fractional = sorted(
            (
                key,
                desired[key] - math.floor(desired[key]),
            )
            for key in desired
        )
        fractional.sort(key=lambda x: x[1], reverse=True)

        idx = 0
        while remaining > 0 and fractional:
            key = fractional[idx % len(fractional)][0]
            if floor_alloc[key] < grouped_counts[key]:
                floor_alloc[key] += 1
                remaining -= 1
            idx += 1
            if idx > 10_000:
                break

        return floor_alloc

    def _lithiation_weight_from_bins(
        self,
        lithiation_fraction: float,
        lith_weights_cfg: dict[str, float],
    ) -> float:
        for key, weight in lith_weights_cfg.items():
            match = re.match(r"\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*", key)
            if not match:
                continue
            hi = float(match.group(1))
            lo = float(match.group(2))
            upper = max(hi, lo)
            lower = min(hi, lo)
            if lower - 1e-8 <= lithiation_fraction <= upper + 1e-8:
                return float(weight)
        return 1.0

    def _compute_descriptor(self, item: GeneratedStructure) -> np.ndarray:
        structure = item.structure
        lattice = structure.lattice
        natoms = len(structure)
        density = float(structure.density)
        volume_per_atom = lattice.volume / natoms
        li_fraction = float(structure.composition.get(self.config.source.target_ion, 0.0)) / natoms

        distance_matrix = structure.distance_matrix
        np.fill_diagonal(distance_matrix, np.inf)
        nearest = np.min(distance_matrix, axis=1)

        descriptor = np.array(
            [
                lattice.a,
                lattice.b,
                lattice.c,
                lattice.alpha,
                lattice.beta,
                lattice.gamma,
                density,
                volume_per_atom,
                li_fraction,
                float(item.lithiation_fraction),
                float(item.temperature_k or 0),
                float(np.mean(nearest)),
                float(np.std(nearest)),
                float(np.percentile(nearest, 10)),
                float(np.percentile(nearest, 50)),
                float(np.percentile(nearest, 90)),
            ],
            dtype=float,
        )
        return descriptor

    def _greedy_maximin_indices(self, vectors: np.ndarray, n_select: int) -> list[int]:
        if n_select <= 0:
            return []
        if len(vectors) <= n_select:
            return list(range(len(vectors)))

        mu = vectors.mean(axis=0)
        d2_center = np.sum((vectors - mu) ** 2, axis=1)
        first_idx = int(np.argmax(d2_center))
        selected = [first_idx]

        d2_min = np.sum((vectors - vectors[first_idx]) ** 2, axis=1)
        d2_min[first_idx] = -1.0
        while len(selected) < n_select:
            next_idx = int(np.argmax(d2_min))
            if d2_min[next_idx] < 0:
                break
            selected.append(next_idx)
            d2_new = np.sum((vectors - vectors[next_idx]) ** 2, axis=1)
            d2_min = np.minimum(d2_min, d2_new)
            d2_min[selected] = -1.0
        return selected

    def write_structures(self, structures: list[GeneratedStructure]) -> None:
        if not structures:
            return

        if structures[0].temperature_k is None:
            base_dir = self.config.output.output_dir / "delithiation_candidates"
        else:
            base_dir = self.config.output.output_dir

        for idx, item in enumerate(structures, start=1):
            lith_dir_name = self._format_lithiation_dir(item.lithiation_fraction)
            if item.temperature_k is None:
                target_dir = base_dir / lith_dir_name
            else:
                if self.config.sampling.rattle_engine == "all" and item.source_engine is not None:
                    target_dir = (
                        base_dir
                        / f"engine_{item.source_engine}"
                        / f"T_{item.temperature_k}K"
                        / lith_dir_name
                    )
                else:
                    target_dir = (
                        base_dir
                        / f"T_{item.temperature_k}K"
                        / lith_dir_name
                    )
            target_dir.mkdir(parents=True, exist_ok=True)

            if self.config.output.output_format == "poscar":
                out_path = target_dir / f"POSCAR_{idx:06d}"
                item.structure.to(fmt="poscar", filename=str(out_path))
            else:
                out_path = target_dir / f"structure_{idx:06d}.cif"
                item.structure.to(fmt="cif", filename=str(out_path))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    pipeline = ElectrodeStructureGenerationPipeline(config=config)
    pipeline.stop_after_delithiation = bool(args.stop_after_delithiation)
    pipeline.skip_direct = bool(args.skip_direct)
    pipeline.slurm_generate_only = bool(args.slurm_generate_only)
    pipeline.slurm_dir = args.slurm_dir
    pipeline.slurm_mode = args.slurm_mode
    pipeline.slurm_queue = args.slurm_queue
    pipeline.slurm_allocation = args.slurm_allocation
    pipeline.slurm_time = args.slurm_time
    pipeline.slurm_ntasks_per_node = args.slurm_ntasks_per_node
    pipeline.slurm_cpus_per_task = args.slurm_cpus_per_task
    pipeline.slurm_gpus = args.slurm_gpus
    pipeline.slurm_combined_jobs = bool(args.slurm_combined_jobs)
    pipeline.only_temperature = args.only_temperature
    pipeline.only_lithiation_fraction = args.only_lithiation_fraction
    pipeline.target_rattle_count = args.target_rattle_count
    if args.bootstrap_output_tree:
        pipeline.bootstrap_output_tree()
        return
    pipeline.run()


if __name__ == "__main__":
    main()
