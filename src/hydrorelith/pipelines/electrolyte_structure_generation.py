from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from hydrorelith.config.defaults import default_electrode_generation_config


@dataclass(slots=True)
class MoleculeTemplate:
    name: str
    species: list[str]
    coords: np.ndarray

    @property
    def n_atoms(self) -> int:
        return len(self.species)


@dataclass(slots=True)
class ElectrolyteStructure:
    structure: Structure
    concentration_label: str
    li_molality: float
    k_molality: float
    density_g_cm3: float
    temperature_k: int | None = None
    candidate_index: int = 0
    source_engine: str | None = None


@dataclass(slots=True)
class CompositionPlan:
    n_solvent: int
    n_li: int
    n_k: int
    n_oh: int
    box_length_a: float
    density_g_cm3: float
    achieved_li_molality: float
    achieved_k_molality: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-electrolyte-generate",
        description=(
            "Generate electrolyte amorphous structures from molecular templates, then optionally rattle "
            "with hiPhive/UMA-MD and select final structures with DIRECT."
        ),
    )
    parser.add_argument(
        "--solvent",
        type=str,
        required=True,
        help="Solvent template in 'name=path' format (e.g., H2O=for_chat_gpt/H2O.bgf)",
    )
    parser.add_argument(
        "--li-template",
        type=str,
        required=True,
        help="Li+ template in 'name=path' format (e.g., Li=for_chat_gpt/Li.bgf)",
    )
    parser.add_argument(
        "--k-template",
        type=str,
        required=True,
        help="K+ template in 'name=path' format (e.g., K=for_chat_gpt/K.bgf)",
    )
    parser.add_argument(
        "--oh-template",
        type=str,
        required=True,
        help="OH- template in 'name=path' format (e.g., OH=for_chat_gpt/OH.bgf)",
    )
    parser.add_argument(
        "--li-k-concentrations",
        type=str,
        required=True,
        help=(
            "Comma-separated LiOH/KOH molality pairs in mol/kg-solvent, "
            "e.g. '4/0,3.5/0.5,3/1,2.5/1.5,2/2,1.5/2.5,1/3,0.5/3.5,0/4'"
        ),
    )
    parser.add_argument("--max-atoms", type=int, default=350)
    parser.add_argument("--structures-per-concentration", type=int, default=5)
    parser.add_argument(
        "--solvent-density-g-cm3",
        type=float,
        default=None,
        help="Density of pure solvent in g/cm^3. Defaults to 1.0 for water-like solvent names.",
    )
    parser.add_argument(
        "--density-correction-per-molal",
        type=float,
        default=0.0,
        help="Optional linear density increment (g/cm^3) per total molality (LiOH+KOH).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("electrolyte_structures"))
    parser.add_argument("--output-format", choices=["poscar", "cif"], default="cif")

    parser.add_argument("--max-structures", type=int, default=600)
    parser.add_argument("--oversampling-factor", type=int, default=10)
    parser.add_argument("--rattle-engine", choices=["hiphive", "uma", "all"], default="hiphive")
    parser.add_argument("--rattle-method", choices=["mc", "gaussian"], default="mc")
    parser.add_argument("--md-ensemble", choices=["nvt", "npt"], default="npt")
    parser.add_argument("--md-steps", type=int, default=500)
    parser.add_argument("--md-sample-interval", type=int, default=10)
    parser.add_argument("--md-timestep-fs", type=float, default=1.0)
    parser.add_argument("--md-friction-per-fs", type=float, default=0.001)
    parser.add_argument("--md-frame-select-fraction", type=float, default=0.10)
    parser.add_argument("--md-min-step-multiplier", type=float, default=4.0)
    parser.add_argument("--uma-model-name", type=str, default="uma-s-1p1")
    parser.add_argument("--uma-task-id", type=str, default="omat")
    parser.add_argument("--uma-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--rattle-std-300k", type=float, default=0.01)
    parser.add_argument("--rattle-d-min", type=float, default=1.5)
    parser.add_argument("--rattle-n-iter", type=int, default=10)
    parser.add_argument("--max-base-structures-per-bin", type=int, default=25)
    parser.add_argument(
        "--hiphive-base-structures-per-concentration",
        type=int,
        default=None,
        help="Optional override for number of base amorphous structures per concentration for hiPhive path.",
    )
    parser.add_argument(
        "--uma-base-structures-per-concentration",
        type=int,
        default=None,
        help="Optional override for number of base amorphous structures per concentration for UMA MD path.",
    )
    parser.add_argument(
        "--hiphive-rattle-fraction-in-all",
        type=float,
        default=0.40,
        help="In all-mode, fraction of target pre-DIRECT pool assigned to hiPhive (UMA gets the remainder).",
    )
    parser.add_argument(
        "--temperatures",
        type=int,
        nargs="*",
        default=[250, 300, 600, 900, 1200],
        help="Default temperature set mirrors electrode generation defaults.",
    )
    parser.add_argument(
        "--pressures-mpa",
        type=float,
        nargs="*",
        default=None,
        help="Optional pressure values aligned with --temperatures; default is 0.1 MPa if omitted.",
    )
    parser.add_argument("--direct-threshold-init", type=float, default=0.05)
    parser.add_argument("--skip-rattling", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow atom overlaps during amorphous building (disabled by default).",
    )
    return parser


def _parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected name=path format, got '{spec}'")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise ValueError(f"Missing name in spec '{spec}'")
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")
    return name, path


def _parse_li_k_pairs(raw: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "/" not in token:
            raise ValueError(f"Invalid concentration token '{token}', expected LiOH/KOH")
        li_str, k_str = token.split("/", 1)
        li_m = float(li_str.strip())
        k_m = float(k_str.strip())
        if li_m < 0 or k_m < 0:
            raise ValueError("Concentrations must be non-negative")
        pairs.append((li_m, k_m))
    if not pairs:
        raise ValueError("No LiOH/KOH concentration pairs were provided")
    return pairs


def _water_like(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in {"h2o", "water", "h2o(l)", "h2o_liquid"}


class ElectrolyteStructureGenerationPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base_cfg = default_electrode_generation_config()
        self.adaptor = AseAtomsAdaptor()

        self.solvent_name, self.solvent_path = _parse_named_path(args.solvent)
        self.li_name, self.li_path = _parse_named_path(args.li_template)
        self.k_name, self.k_path = _parse_named_path(args.k_template)
        self.oh_name, self.oh_path = _parse_named_path(args.oh_template)
        self.concentration_pairs = _parse_li_k_pairs(args.li_k_concentrations)

        self.solvent_density = self._resolve_solvent_density()
        self.templates = self._load_templates()

    @staticmethod
    def _bounded_seed(seed: int) -> int:
        return int(seed % (2**32 - 1))

    def _resolve_solvent_density(self) -> float:
        if self.args.solvent_density_g_cm3 is not None:
            return float(self.args.solvent_density_g_cm3)
        if _water_like(self.solvent_name):
            return 1.0
        raise ValueError(
            "--solvent-density-g-cm3 is required for non-water solvent names."
        )

    def _load_templates(self) -> dict[str, MoleculeTemplate]:
        return {
            "solvent": self._load_template(self.solvent_name, self.solvent_path),
            "li": self._load_template(self.li_name, self.li_path),
            "k": self._load_template(self.k_name, self.k_path),
            "oh": self._load_template(self.oh_name, self.oh_path),
        }

    def _load_template(self, name: str, path: Path) -> MoleculeTemplate:
        if path.suffix.lower() == ".bgf":
            species, coords = self._read_bgf(path)
            return MoleculeTemplate(name=name, species=species, coords=coords)

        from ase.io import read

        atoms = read(str(path))
        species = atoms.get_chemical_symbols()
        coords = atoms.get_positions()
        return MoleculeTemplate(name=name, species=list(species), coords=np.array(coords, dtype=float))

    def _read_bgf(self, path: Path) -> tuple[list[str], np.ndarray]:
        species: list[str] = []
        coords: list[list[float]] = []
        pattern = re.compile(r"^(?:ATOM|HETATM)\s+")

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not pattern.match(line):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            x, y, z = float(fields[6]), float(fields[7]), float(fields[8])
            element = fields[9]
            if not re.match(r"^[A-Za-z]{1,2}$", element):
                atom_name = fields[2]
                matched = re.match(r"([A-Za-z]{1,2})", atom_name)
                if not matched:
                    raise ValueError(f"Cannot infer element from line in {path}: {line}")
                element = matched.group(1)
            species.append(element.capitalize())
            coords.append([x, y, z])

        if not species:
            raise ValueError(f"No atom lines found in BGF file: {path}")

        return species, np.array(coords, dtype=float)

    def run(self) -> None:
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        if self.args.rattle_engine == "all" and not self.args.skip_rattling:
            base_structures_hiphive = self.generate_base_structures(engine="hiphive")
            base_structures_uma = self.generate_base_structures(engine="uma")
            self.write_structures(
                base_structures_hiphive,
                base_dir=self.args.output_dir / "base_structures" / "engine_hiphive",
            )
            self.write_structures(
                base_structures_uma,
                base_dir=self.args.output_dir / "base_structures" / "engine_uma",
            )
        else:
            base_structures = self.generate_base_structures(engine=self.args.rattle_engine)
            self.write_structures(base_structures, base_dir=self.args.output_dir / "base_structures")

        if self.args.skip_rattling:
            selected = base_structures
        else:
            if self.args.rattle_engine == "all":
                rattled = self.generate_rattled_candidates_all(
                    hiphive_structures=base_structures_hiphive,
                    uma_structures=base_structures_uma,
                )
            else:
                rattled = self.generate_rattled_candidates(base_structures, engine=self.args.rattle_engine)
            if self.args.skip_direct:
                selected = rattled
            else:
                selected = self.select_with_direct(rattled)

        self.write_structures(selected, base_dir=self.args.output_dir / "best_training_set")

    def _base_structures_per_concentration(self, engine: str | None) -> int:
        if engine == "hiphive" and self.args.hiphive_base_structures_per_concentration is not None:
            return max(1, int(self.args.hiphive_base_structures_per_concentration))
        if engine == "uma" and self.args.uma_base_structures_per_concentration is not None:
            return max(1, int(self.args.uma_base_structures_per_concentration))

        base = max(1, int(self.args.structures_per_concentration))
        if engine == "uma":
            return max(1, int(math.ceil(base / 3.0)))
        if engine == "hiphive":
            return max(3, base)
        return base

    def generate_base_structures(self, engine: str | None = None) -> list[ElectrolyteStructure]:
        generated: list[ElectrolyteStructure] = []
        summary: dict[str, dict] = {}
        n_per_concentration = self._base_structures_per_concentration(engine)

        for li_m, k_m in self.concentration_pairs:
            label = self._concentration_label(li_m, k_m)
            density = self.solvent_density + self.args.density_correction_per_molal * (li_m + k_m)
            plan = self._plan_composition(li_m=li_m, k_m=k_m, density_g_cm3=density)

            summary[label] = {
                "target_li_molality": li_m,
                "target_k_molality": k_m,
                "achieved_li_molality": plan.achieved_li_molality,
                "achieved_k_molality": plan.achieved_k_molality,
                "density_g_cm3": density,
                "n_solvent": plan.n_solvent,
                "n_li": plan.n_li,
                "n_k": plan.n_k,
                "n_oh": plan.n_oh,
                "box_length_a": plan.box_length_a,
            }

            summary[label]["n_base_structures"] = int(n_per_concentration)
            summary[label]["engine_context"] = engine or "none"

            for idx in range(n_per_concentration):
                seed = self._bounded_seed(
                    1_000_003 * (idx + 1) + int(round(100 * li_m)) * 101 + int(round(100 * k_m)) * 137
                )
                structure = self._build_amorphous_structure(plan, seed=seed)
                generated.append(
                    ElectrolyteStructure(
                        structure=structure,
                        concentration_label=label,
                        li_molality=plan.achieved_li_molality,
                        k_molality=plan.achieved_k_molality,
                        density_g_cm3=density,
                        candidate_index=idx,
                    )
                )

        suffix = f"_{engine}" if engine else ""
        summary_path = self.args.output_dir / f"electrolyte_generation_overview{suffix}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return generated

    def _concentration_label(self, li_m: float, k_m: float) -> str:
        return f"LiOH_{li_m:.2f}_KOH_{k_m:.2f}"

    def _plan_composition(self, li_m: float, k_m: float, density_g_cm3: float) -> CompositionPlan:
        solvent_atoms = self.templates["solvent"].n_atoms
        li_atoms = self.templates["li"].n_atoms
        k_atoms = self.templates["k"].n_atoms
        oh_atoms = self.templates["oh"].n_atoms

        max_solvent = max(1, self.args.max_atoms // solvent_atoms)
        mw_solvent = self._molecule_mass_g_per_mol(self.templates["solvent"])  # g/mol
        factor = mw_solvent / 1000.0

        best: CompositionPlan | None = None
        best_score: tuple[int, float] | None = None

        for n_solvent in range(1, max_solvent + 1):
            li_target = li_m * factor * n_solvent
            k_target = k_m * factor * n_solvent

            li_candidates = sorted({max(0, int(math.floor(li_target))), int(round(li_target)), int(math.ceil(li_target))})
            k_candidates = sorted({max(0, int(math.floor(k_target))), int(round(k_target)), int(math.ceil(k_target))})

            for n_li in li_candidates:
                for n_k in k_candidates:
                    n_oh = n_li + n_k
                    total_atoms = (
                        n_solvent * solvent_atoms
                        + n_li * li_atoms
                        + n_k * k_atoms
                        + n_oh * oh_atoms
                    )
                    if total_atoms > self.args.max_atoms:
                        continue

                    achieved_li = n_li / (n_solvent * factor)
                    achieved_k = n_k / (n_solvent * factor)
                    error = abs(achieved_li - li_m) + abs(achieved_k - k_m)
                    score = (total_atoms, -error)

                    if best is None or score > best_score:
                        box_length = self._estimate_cubic_box_length(
                            n_solvent=n_solvent,
                            n_li=n_li,
                            n_k=n_k,
                            n_oh=n_oh,
                            density_g_cm3=density_g_cm3,
                        )
                        best = CompositionPlan(
                            n_solvent=n_solvent,
                            n_li=n_li,
                            n_k=n_k,
                            n_oh=n_oh,
                            box_length_a=box_length,
                            density_g_cm3=density_g_cm3,
                            achieved_li_molality=achieved_li,
                            achieved_k_molality=achieved_k,
                        )
                        best_score = score

        if best is None:
            raise RuntimeError(
                f"Could not satisfy composition for LiOH/KOH={li_m}/{k_m} under max-atoms={self.args.max_atoms}"
            )
        return best

    def _atomic_mass(self, symbol: str) -> float:
        from pymatgen.core import Element

        return float(Element(symbol).atomic_mass)

    def _molecule_mass_g_per_mol(self, template: MoleculeTemplate) -> float:
        return float(sum(self._atomic_mass(sp) for sp in template.species))

    def _estimate_cubic_box_length(
        self,
        n_solvent: int,
        n_li: int,
        n_k: int,
        n_oh: int,
        density_g_cm3: float,
    ) -> float:
        na = 6.02214076e23
        total_mass_g = (
            n_solvent * self._molecule_mass_g_per_mol(self.templates["solvent"])
            + n_li * self._molecule_mass_g_per_mol(self.templates["li"])
            + n_k * self._molecule_mass_g_per_mol(self.templates["k"])
            + n_oh * self._molecule_mass_g_per_mol(self.templates["oh"])
        ) / na
        volume_cm3 = total_mass_g / density_g_cm3
        volume_a3 = volume_cm3 * 1.0e24
        return float(volume_a3 ** (1.0 / 3.0))

    def _build_amorphous_structure(self, plan: CompositionPlan, seed: int) -> Structure:
        rng = np.random.default_rng(seed)
        box_len = plan.box_length_a
        lattice = Lattice.cubic(box_len)

        species: list[str] = []
        cart_coords: list[np.ndarray] = []

        insertion_plan = [
            ("solvent", plan.n_solvent),
            ("li", plan.n_li),
            ("k", plan.n_k),
            ("oh", plan.n_oh),
        ]

        for key, count in insertion_plan:
            template = self.templates[key]
            centered = template.coords - np.mean(template.coords, axis=0)
            for _ in range(count):
                placed = self._place_molecule(
                    centered_coords=centered,
                    existing=cart_coords,
                    box_len=box_len,
                    rng=rng,
                )

                for atom_symbol, atom_xyz in zip(template.species, placed):
                    species.append(atom_symbol)
                    cart_coords.append(atom_xyz)

        frac_coords = [lattice.get_fractional_coords(c) for c in cart_coords]
        return Structure(lattice=lattice, species=species, coords=frac_coords)

    def _has_overlap(self, existing: list[np.ndarray], candidate: np.ndarray, box_len: float) -> bool:
        existing_arr = np.array(existing)
        for atom in candidate:
            delta = atom - existing_arr
            delta -= box_len * np.round(delta / box_len)
            dist = np.linalg.norm(delta, axis=1)
            if np.any(dist < 0.8):
                return True
        return False

    def _place_molecule(
        self,
        centered_coords: np.ndarray,
        existing: list[np.ndarray],
        box_len: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.args.allow_overlap or not existing:
            rot = self._random_rotation_matrix(rng)
            return (centered_coords @ rot.T + rng.uniform(0.0, box_len, size=3)) % box_len

        max_attempts = 1000
        for _ in range(max_attempts):
            rot = self._random_rotation_matrix(rng)
            placed = (centered_coords @ rot.T + rng.uniform(0.0, box_len, size=3)) % box_len
            if not self._has_overlap(existing=existing, candidate=placed, box_len=box_len):
                return placed

        raise RuntimeError(
            "Unable to place molecule without overlap after many attempts. "
            "Increase max-atoms/box size or use --allow-overlap if absolutely necessary."
        )

    def _random_rotation_matrix(self, rng: np.random.Generator) -> np.ndarray:
        u1, u2, u3 = rng.random(3)
        q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
        q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
        q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
        q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)

        return np.array(
            [
                [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
                [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
                [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
            ]
        )

    def _target_count_for_engine(self, engine: str) -> int:
        base_target = int(self.args.max_structures * self.args.oversampling_factor)
        if self.args.rattle_engine != "all":
            return base_target

        hip_frac = float(self.args.hiphive_rattle_fraction_in_all)
        hip_frac = max(0.05, min(0.95, hip_frac))
        hip_target = max(1, int(round(base_target * hip_frac)))
        uma_target = max(1, base_target - hip_target)
        return hip_target if engine == "hiphive" else uma_target

    def generate_rattled_candidates(self, structures: list[ElectrolyteStructure], engine: str) -> list[ElectrolyteStructure]:
        target_count = self._target_count_for_engine(engine)
        if engine == "hiphive":
            return self._generate_rattled_hiphive(structures, target_count)
        if engine == "uma":
            return self._generate_rattled_uma(structures, target_count)
        raise ValueError(f"Unsupported rattling engine: {engine}")

    def generate_rattled_candidates_all(
        self,
        hiphive_structures: list[ElectrolyteStructure],
        uma_structures: list[ElectrolyteStructure],
    ) -> list[ElectrolyteStructure]:
        combined: list[ElectrolyteStructure] = []
        combined.extend(self.generate_rattled_candidates(hiphive_structures, engine="hiphive"))
        combined.extend(self.generate_rattled_candidates(uma_structures, engine="uma"))
        return combined

    def _generate_rattled_hiphive(self, structures: list[ElectrolyteStructure], target_count: int) -> list[ElectrolyteStructure]:
        try:
            from hiphive.structure_generation import generate_mc_rattled_structures, generate_rattled_structures
        except Exception as exc:
            raise RuntimeError("hiPhive is required for rattle-engine=hiphive") from exc

        grouped: dict[str, list[ElectrolyteStructure]] = defaultdict(list)
        for item in structures:
            grouped[item.concentration_label].append(item)

        bins = [(t, c) for t in self.args.temperatures for c in sorted(grouped.keys())]
        per_bin = target_count // len(bins)
        rem = target_count % len(bins)

        expanded: list[ElectrolyteStructure] = []
        for idx, (temperature, conc) in enumerate(bins):
            n_bin = per_bin + (1 if idx < rem else 0)
            base = grouped[conc]
            max_bases = min(self.args.max_base_structures_per_bin, len(base))
            base_work = [n_bin // max_bases] * max_bases
            for i in range(n_bin % max_bases):
                base_work[i] += 1

            for base_idx, base_item in enumerate(base[:max_bases]):
                n_structures = base_work[base_idx]
                if n_structures <= 0:
                    continue
                atoms = self.adaptor.get_atoms(base_item.structure)
                seed = self._bounded_seed(11_000_003 * temperature + 113 * base_item.candidate_index + 17)

                if self.args.rattle_method == "gaussian":
                    scaled_std = self.args.rattle_std_300k * math.sqrt(temperature / 300.0)
                    rattled = generate_rattled_structures(
                        atoms=atoms,
                        n_structures=n_structures,
                        rattle_std=scaled_std,
                        seed=seed,
                    )
                else:
                    scaled_std = self.args.rattle_std_300k * math.sqrt(temperature / 300.0)
                    try:
                        rattled = generate_mc_rattled_structures(
                            atoms=atoms,
                            n_structures=n_structures,
                            rattle_std=scaled_std,
                            d_min=self.args.rattle_d_min,
                            n_iter=self.args.rattle_n_iter,
                            seed=seed,
                        )
                    except Exception:
                        try:
                            rattled = generate_mc_rattled_structures(
                                atoms=atoms,
                                n_structures=n_structures,
                                rattle_std=scaled_std,
                                d_min=max(0.8, float(self.args.rattle_d_min) * 0.7),
                                n_iter=max(20, int(self.args.rattle_n_iter) * 2),
                                seed=seed,
                            )
                        except Exception:
                            rattled = generate_rattled_structures(
                                atoms=atoms,
                                n_structures=n_structures,
                                rattle_std=scaled_std,
                                seed=seed,
                            )

                for ridx, rattled_atoms in enumerate(rattled):
                    expanded.append(
                        ElectrolyteStructure(
                            structure=self.adaptor.get_structure(rattled_atoms),
                            concentration_label=base_item.concentration_label,
                            li_molality=base_item.li_molality,
                            k_molality=base_item.k_molality,
                            density_g_cm3=base_item.density_g_cm3,
                            temperature_k=temperature,
                            candidate_index=base_item.candidate_index * 10_000 + ridx,
                            source_engine="hiphive",
                        )
                    )

        return expanded

    def _pressure_mpa_for_temperature(self, temperature: int) -> float:
        if not self.args.pressures_mpa:
            return 0.1
        pressure_map = {t: p for t, p in zip(self.args.temperatures, self.args.pressures_mpa)}
        return float(pressure_map.get(temperature, 0.1))

    def _required_md_steps(self, n_structures: int) -> int:
        sample_interval = max(1, int(self.args.md_sample_interval))
        base_samples = max(1, int(math.ceil(n_structures)))
        fraction = float(self.args.md_frame_select_fraction)
        min_multiplier = max(1.0, float(self.args.md_min_step_multiplier))
        required_samples = max(
            int(math.ceil(base_samples / max(fraction, 1e-9))),
            int(math.ceil(base_samples * min_multiplier)),
        )
        required_by_sampling = required_samples * sample_interval
        return max(required_by_sampling, int(self.args.md_steps))

    def _select_even_snapshots(self, snapshots, n_keep: int, seed: int):
        if n_keep <= 0 or not snapshots:
            return []
        if len(snapshots) <= n_keep:
            return [snapshots[i].copy() for i in range(len(snapshots))]
        rng = random.Random(seed)
        indices = list(range(len(snapshots)))
        selected = sorted(rng.sample(indices, n_keep))
        return [snapshots[i].copy() for i in selected]

    def _md_output_paths(self, temperature: int, concentration_label: str, base_idx: int) -> tuple[Path, Path]:
        safe_label = concentration_label.replace("/", "_")
        out_dir = self.args.output_dir / "md_runs" / f"T_{temperature}K" / safe_label
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"base_{base_idx:08d}"
        return out_dir / f"{stem}.extxyz", out_dir / f"{stem}_properties.csv"

    def _run_uma_md_snapshots(
        self,
        atoms,
        temperature: int,
        concentration_label: str,
        base_candidate_index: int,
        n_structures: int,
        seed: int,
        pressure_mpa: float | None = None,
        progress_callback=None,
    ):
        from ase import units
        from ase.io import write
        from ase.md.langevin import Langevin
        from ase.md.nptberendsen import NPTBerendsen
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        if not hasattr(self, "_uma_predictor"):
            self._uma_predictor = pretrained_mlip.get_predict_unit(
                self.args.uma_model_name,
                device=self.args.uma_device,
            )

        atoms = atoms.copy()
        atoms.calc = FAIRChemCalculator(self._uma_predictor, task_name=self.args.uma_task_id)
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(temperature), rng=np.random.default_rng(seed))
        Stationary(atoms)

        traj_path, prop_log_path = self._md_output_paths(
            temperature=temperature,
            concentration_label=concentration_label,
            base_idx=base_candidate_index,
        )
        if traj_path.exists():
            traj_path.unlink()
        sample_interval = int(self.args.md_sample_interval)

        log_file = prop_log_path.open("w", encoding="utf-8")
        log_file.write("step,temperature_K,pressure_GPa,kinetic_energy_eV,potential_energy_eV,total_energy_eV\n")

        timestep = self.args.md_timestep_fs * units.fs
        if self.args.md_ensemble == "npt":
            target_pressure_mpa = float(pressure_mpa if pressure_mpa is not None else self._pressure_mpa_for_temperature(temperature))
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
                friction=self.args.md_friction_per_fs / units.fs,
            )

        snapshots = []
        step_counter = {"value": 0}

        def _capture():
            snapshots.append(atoms.copy())
            write(str(traj_path), atoms, format="extxyz", append=True)

        def _log_properties():
            try:
                stress = atoms.get_stress(voigt=True)
                pressure_gpa = -float(np.mean(stress[:3])) / units.GPa
            except Exception:
                pressure_gpa = float("nan")

            kinetic = float(atoms.get_kinetic_energy())
            potential = float(atoms.get_potential_energy())
            total = kinetic + potential
            temperature_k = float(atoms.get_temperature())
            log_file.write(
                f"{step_counter['value']},{temperature_k:.8f},{pressure_gpa:.8f},{kinetic:.8f},{potential:.8f},{total:.8f}\n"
            )
            log_file.flush()
            step_counter["value"] += sample_interval

        dyn.attach(_capture, interval=sample_interval)
        dyn.attach(_log_properties, interval=sample_interval)
        n_steps = self._required_md_steps(n_structures)

        if progress_callback is not None:
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

        try:
            dyn.run(steps=n_steps)
        finally:
            log_file.close()

        return self._select_even_snapshots(snapshots, n_structures, seed=seed)

    def _generate_rattled_uma(self, structures: list[ElectrolyteStructure], target_count: int) -> list[ElectrolyteStructure]:
        grouped: dict[str, list[ElectrolyteStructure]] = defaultdict(list)
        for item in structures:
            grouped[item.concentration_label].append(item)

        bins = [(t, c) for t in self.args.temperatures for c in sorted(grouped.keys())]
        per_bin = target_count // len(bins)
        rem = target_count % len(bins)

        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None

        expanded: list[ElectrolyteStructure] = []
        for idx, (temperature, conc) in enumerate(bins):
            n_bin = per_bin + (1 if idx < rem else 0)
            base = grouped[conc]
            max_bases = min(self.args.max_base_structures_per_bin, len(base))
            base_work = [n_bin // max_bases] * max_bases
            for i in range(n_bin % max_bases):
                base_work[i] += 1

            pbar = (
                tqdm(total=n_bin, desc=f"UMA MD T={temperature}K {conc}", leave=False)
                if tqdm is not None
                else None
            )

            for base_idx, base_item in enumerate(base[:max_bases]):
                n_structures = base_work[base_idx]
                if n_structures <= 0:
                    continue
                atoms = self.adaptor.get_atoms(base_item.structure)
                seed = self._bounded_seed(13_000_003 * temperature + 127 * base_item.candidate_index + 29)
                fraction = float(self.args.md_frame_select_fraction)
                live_progress = {"selected_equiv": 0}

                def _progress_callback(stage: str, captured: int, md_steps_done: int, md_steps_total: int) -> None:
                    if pbar is None or stage != "running":
                        return
                    estimated_selected = min(
                        n_structures,
                        int(math.floor(captured * fraction + 1e-12)),
                    )
                    delta = estimated_selected - live_progress["selected_equiv"]
                    if delta > 0:
                        pbar.update(delta)
                        live_progress["selected_equiv"] = estimated_selected

                snaps = self._run_uma_md_snapshots(
                    atoms=atoms,
                    temperature=temperature,
                    concentration_label=conc,
                    base_candidate_index=base_item.candidate_index,
                    n_structures=n_structures,
                    seed=seed,
                    pressure_mpa=self._pressure_mpa_for_temperature(temperature),
                    progress_callback=_progress_callback,
                )
                if pbar is not None and live_progress["selected_equiv"] < len(snaps):
                    pbar.update(len(snaps) - live_progress["selected_equiv"])
                for sidx, snap in enumerate(snaps):
                    expanded.append(
                        ElectrolyteStructure(
                            structure=self.adaptor.get_structure(snap),
                            concentration_label=conc,
                            li_molality=base_item.li_molality,
                            k_molality=base_item.k_molality,
                            density_g_cm3=base_item.density_g_cm3,
                            temperature_k=temperature,
                            candidate_index=base_item.candidate_index * 100_000 + sidx,
                            source_engine="uma",
                        )
                    )

            if pbar is not None:
                pbar.close()

        return expanded

    def _compute_descriptor(self, item: ElectrolyteStructure) -> np.ndarray:
        structure = item.structure
        lattice = structure.lattice
        natoms = len(structure)
        volume_per_atom = lattice.volume / natoms

        distance_matrix = structure.distance_matrix
        np.fill_diagonal(distance_matrix, np.inf)
        nearest = np.min(distance_matrix, axis=1)

        return np.array(
            [
                lattice.a,
                lattice.b,
                lattice.c,
                lattice.alpha,
                lattice.beta,
                lattice.gamma,
                float(structure.density),
                volume_per_atom,
                float(item.li_molality),
                float(item.k_molality),
                float(item.temperature_k or 0),
                float(np.mean(nearest)),
                float(np.std(nearest)),
                float(np.percentile(nearest, 10)),
                float(np.percentile(nearest, 50)),
                float(np.percentile(nearest, 90)),
            ],
            dtype=float,
        )

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

    def _run_maml_direct(self, vectors: np.ndarray, n_select: int) -> list[int]:
        if len(vectors) <= n_select:
            return list(range(len(vectors)))
        from maml.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters

        sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=n_select, threshold_init=self.args.direct_threshold_init),
            select_k_from_clusters=SelectKFromClusters(k=1),
        )
        result = sampler.fit_transform(vectors)
        selected = [int(i) for i in result.get("selected_indexes", [])]
        if len(selected) < n_select:
            remaining = [i for i in range(len(vectors)) if i not in set(selected)]
            fill = self._greedy_maximin_indices(vectors[remaining], min(n_select - len(selected), len(remaining)))
            selected.extend(remaining[i] for i in fill)
        return selected[:n_select]

    def select_with_direct(self, structures: list[ElectrolyteStructure]) -> list[ElectrolyteStructure]:
        if len(structures) <= self.args.max_structures:
            return structures

        grouped_indices: dict[tuple[int, str], list[int]] = defaultdict(list)
        for idx, item in enumerate(structures):
            temp = int(item.temperature_k or 300)
            grouped_indices[(temp, item.concentration_label)].append(idx)

        n_groups = len(grouped_indices)
        base_quota = self.args.max_structures // n_groups
        rem = self.args.max_structures % n_groups

        descriptors = [self._compute_descriptor(item) for item in structures]
        selected_indices: list[int] = []
        for gidx, group in enumerate(sorted(grouped_indices.keys())):
            indices = grouped_indices[group]
            quota = min(base_quota + (1 if gidx < rem else 0), len(indices))
            if quota <= 0:
                continue
            local = np.array([descriptors[i] for i in indices], dtype=float)
            local_sel = self._run_maml_direct(local, quota)
            selected_indices.extend(indices[i] for i in local_sel)

        if len(selected_indices) < self.args.max_structures:
            selected_set = set(selected_indices)
            remaining = [i for i in range(len(structures)) if i not in selected_set]
            need = self.args.max_structures - len(selected_indices)
            if remaining:
                rem_vectors = np.array([descriptors[i] for i in remaining], dtype=float)
                fill = self._greedy_maximin_indices(rem_vectors, min(need, len(remaining)))
                selected_indices.extend(remaining[i] for i in fill)

        selected_indices = selected_indices[: self.args.max_structures]
        return [structures[i] for i in selected_indices]

    def write_structures(self, structures: list[ElectrolyteStructure], base_dir: Path) -> None:
        if not structures:
            return
        for idx, item in enumerate(structures, start=1):
            temp_tag = f"T_{item.temperature_k}K" if item.temperature_k is not None else "T_base"
            label = item.concentration_label.replace("/", "_")
            target_dir = base_dir / temp_tag / label
            target_dir.mkdir(parents=True, exist_ok=True)
            if self.args.output_format == "poscar":
                out_path = target_dir / f"POSCAR_{idx:06d}"
                item.structure.to(fmt="poscar", filename=str(out_path))
            else:
                out_path = target_dir / f"structure_{idx:06d}.cif"
                item.structure.to(fmt="cif", filename=str(out_path))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = ElectrolyteStructureGenerationPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
