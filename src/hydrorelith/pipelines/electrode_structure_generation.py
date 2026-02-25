from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from hydrorelith.config.defaults import default_electrode_generation_config
from hydrorelith.config.schemas import ElectrodeGenerationConfig


@dataclass(slots=True)
class GeneratedStructure:
    structure: Structure
    lithiation_fraction: float
    temperature_k: int | None = None
    candidate_index: int = 0


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
    parser.add_argument("--rattles-per-structure", type=int, default=1)
    parser.add_argument("--rattle-std-300k", type=float, default=0.01)
    parser.add_argument("--rattle-d-min", type=float, default=1.5)
    parser.add_argument("--rattle-n-iter", type=int, default=10)
    parser.add_argument("--max-base-structures-per-bin", type=int, default=25)
    parser.add_argument("--phonon-fc2-path", type=Path, default=None)
    parser.add_argument("--phonon-qm-statistics", action="store_true")
    parser.add_argument("--phonon-imag-freq-factor", type=float, default=1.0)
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
    config.sampling.rattle_method = args.rattle_method
    config.sampling.rattles_per_structure = args.rattles_per_structure
    config.sampling.rattle_std_300k = args.rattle_std_300k
    config.sampling.rattle_d_min = args.rattle_d_min
    config.sampling.rattle_n_iter = args.rattle_n_iter
    config.sampling.max_base_structures_per_bin = args.max_base_structures_per_bin
    config.sampling.phonon_fc2_path = args.phonon_fc2_path
    config.sampling.phonon_qm_statistics = bool(args.phonon_qm_statistics)
    config.sampling.phonon_imag_freq_factor = args.phonon_imag_freq_factor

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
        if getattr(self, "stop_after_delithiation", False):
            self.write_structures(delithiation_candidates)
            return
        target_temperatures = self.resolve_target_temperatures(base_structure)
        rattled_candidates = self.generate_rattled_candidates(delithiation_candidates, target_temperatures)
        if getattr(self, "skip_direct", False):
            output_structures = rattled_candidates
        else:
            output_structures = self.select_with_direct(rattled_candidates)
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
                "Pressure values are recorded for downstream MD/thermo stages and future hiPhive coupling.",
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
        target_count = self.config.sampling.max_structures * self.config.sampling.oversampling_factor
        by_lithiation: dict[float, list[GeneratedStructure]] = defaultdict(list)
        for item in structures:
            by_lithiation[item.lithiation_fraction].append(item)

        lithiation_levels = sorted(by_lithiation.keys(), reverse=True)
        bins = [(temperature, lithiation) for temperature in temperatures for lithiation in lithiation_levels]
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

        for temperature in temperatures:
            for lithiation in lithiation_levels:
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
            local_selected = self._greedy_maximin_indices(local_vectors, quota)
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
    if args.bootstrap_output_tree:
        pipeline.bootstrap_output_tree()
        return
    pipeline.run()


if __name__ == "__main__":
    main()
