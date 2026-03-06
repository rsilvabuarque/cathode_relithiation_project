from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class StructureItem:
    condition_id: str
    structure_path: Path
    temperature_C: float | None
    pressure_MPa: float | None
    phase: str
    task_name: str
    liOH_M: float | None = None
    kOH_M: float | None = None
    lithiation_fraction: float | None = None
    vacancy_config_id: str | None = None
    charge: int | None = None
    spin: int | None = None
    n_li: int | None = None
    notes: str | None = None


def _parse_decimal_token(token: str) -> float:
    # Tree/file names often encode decimals as 2p02 instead of 2.02.
    return float(token.replace("p", "."))


def _required_columns(phase: str) -> list[str]:
    base = ["condition_id", "structure_path", "temperature_C", "pressure_MPa", "phase", "task_name"]
    if phase == "electrolyte":
        return base + ["liOH_M", "kOH_M"]
    return base + ["lithiation_fraction", "vacancy_config_id"]


def _default_values(phase: str) -> dict[str, str]:
    if phase == "electrolyte":
        return {"phase": "electrolyte", "task_name": "omol", "charge": "0", "spin": "1"}
    return {"phase": "electrode", "task_name": "omat"}


def load_manifest(path: Path, phase: str) -> list[StructureItem]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    defaults = _default_values(phase)
    required = _required_columns(phase)
    items: list[StructureItem] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest {path} missing required columns: {missing}")
        for idx, row in enumerate(reader, start=2):
            merged = {**defaults, **{k: (v if v is not None else "") for k, v in row.items()}}
            condition_id = merged["condition_id"].strip()
            structure_path = Path(merged["structure_path"].strip())
            if not condition_id:
                raise ValueError(f"{path}:{idx}: empty condition_id")
            if not structure_path.exists():
                raise FileNotFoundError(f"{path}:{idx}: structure file not found: {structure_path}")

            def _float(key: str) -> float | None:
                val = merged.get(key, "").strip()
                return float(val) if val else None

            def _int(key: str) -> int | None:
                val = merged.get(key, "").strip()
                return int(val) if val else None

            items.append(
                StructureItem(
                    condition_id=condition_id,
                    structure_path=structure_path,
                    temperature_C=_float("temperature_C"),
                    pressure_MPa=_float("pressure_MPa"),
                    phase=(merged.get("phase") or defaults["phase"]).strip() or defaults["phase"],
                    task_name=(merged.get("task_name") or defaults["task_name"]).strip() or defaults["task_name"],
                    liOH_M=_float("liOH_M"),
                    kOH_M=_float("kOH_M"),
                    lithiation_fraction=_float("lithiation_fraction"),
                    vacancy_config_id=(merged.get("vacancy_config_id") or "").strip() or None,
                    charge=_int("charge"),
                    spin=_int("spin"),
                    n_li=_int("n_li"),
                    notes=(merged.get("notes") or "").strip() or None,
                )
            )
    return items


def discover_structures(root: Path, phase: str) -> list[StructureItem]:
    if not root.exists():
        raise FileNotFoundError(f"Discovery root not found: {root}")
    supported = {".xyz", ".extxyz", ".cif", ".pdb", ".vasp"}
    items: list[StructureItem] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        is_poscar = path.name.upper().startswith("POSCAR")
        if path.suffix.lower() not in supported and not is_poscar:
            continue
        token = path.as_posix()
        temp_match = re.search(r"(?:T[_-]?|temp[_-]?)(\d+(?:\.\d+)?)", token, flags=re.IGNORECASE)
        pres_match = re.search(r"(?:P[_-]?|press(?:ure)?[_-]?)(\d+(?:\.\d+)?)", token, flags=re.IGNORECASE)
        lioh_match = re.search(r"LiOH[_-]?(\d+(?:\.\d+)?)M", token, flags=re.IGNORECASE)
        koh_match = re.search(r"KOH[_-]?(\d+(?:\.\d+)?)M", token, flags=re.IGNORECASE)
        lith_match = re.search(r"lith[_-]?(\d+(?:\.\d+)?)", token, flags=re.IGNORECASE)
        vac_match = re.search(r"vac(?:ancy)?[_-]?([A-Za-z0-9.-]+)", token, flags=re.IGNORECASE)
        condition_id = path.stem
        items.append(
            StructureItem(
                condition_id=condition_id,
                structure_path=path,
                temperature_C=float(temp_match.group(1)) if temp_match else None,
                pressure_MPa=float(pres_match.group(1)) if pres_match else None,
                phase=phase,
                task_name="omol" if phase == "electrolyte" else "omat",
                liOH_M=float(lioh_match.group(1)) if lioh_match else None,
                kOH_M=float(koh_match.group(1)) if koh_match else None,
                lithiation_fraction=(float(lith_match.group(1)) / 100.0 if lith_match else None),
                vacancy_config_id=(vac_match.group(1) if vac_match else None),
                charge=0 if phase == "electrolyte" else None,
                spin=1 if phase == "electrolyte" else None,
            )
        )
    return items


def generate_default_electrolyte_manifest_from_tree(structures_root: Path, out_manifest: Path) -> None:
    """Generate electrolyte manifest from a concentration/T/P directory tree.

    Expected best-effort layout:
      <root>/LiOH_<li>M_KOH_<k>M/T<temp>_P<press>/<structure>.cif

    Where decimal tokens can be either 2.02 or 2p02 style.
    """
    if not structures_root.exists():
        raise FileNotFoundError(f"Electrolyte structures root not found: {structures_root}")

    conc_re = re.compile(
        r"LiOH_(\d+(?:[.]\d+|p\d+)?)M_KOH_(\d+(?:[.]\d+|p\d+)?)M",
        flags=re.IGNORECASE,
    )
    tp_re = re.compile(
        r"T(\d+(?:[.]\d+|p\d+)?)_P(\d+(?:[.]\d+|p\d+)?)",
        flags=re.IGNORECASE,
    )

    rows: list[list[str]] = []
    for path in sorted(structures_root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        is_poscar = path.name.upper().startswith("POSCAR")
        if ext not in {".xyz", ".extxyz", ".cif", ".pdb", ".vasp"} and not is_poscar:
            continue

        rel_parts = path.relative_to(structures_root).parts
        conc_token = next((part for part in rel_parts if conc_re.fullmatch(part)), None)
        tp_token = next((part for part in rel_parts if tp_re.fullmatch(part)), None)
        if conc_token is None or tp_token is None:
            raise ValueError(
                "Could not parse concentration/T-P tags for file "
                f"{path}. Expected parent folders like LiOH_0.00M_KOH_4.13M/T120_P0p08/."
            )

        c_match = conc_re.fullmatch(conc_token)
        t_match = tp_re.fullmatch(tp_token)
        assert c_match is not None
        assert t_match is not None

        li_m = _parse_decimal_token(c_match.group(1))
        k_m = _parse_decimal_token(c_match.group(2))
        temp_c = _parse_decimal_token(t_match.group(1))
        pressure_mpa = _parse_decimal_token(t_match.group(2))

        condition_id = path.stem
        rows.append(
            [
                condition_id,
                str(path),
                f"{temp_c:g}",
                f"{pressure_mpa:g}",
                f"{li_m:g}",
                f"{k_m:g}",
                "electrolyte",
                "omol",
                "0",
                "1",
                "",
            ]
        )

    if not rows:
        raise ValueError(f"No structure files found under {structures_root}")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="", encoding="utf-8") as handle:
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
        writer.writerows(rows)


def write_template_manifest(path: Path, phase: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if phase == "electrolyte":
        headers = [
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
        rows = [["el_220C_2p02MPa_li4_k0", "path/to/electrolyte.xyz", "220", "2.02", "4.0", "0.0", "electrolyte", "omol", "0", "1", ""]]
    else:
        headers = [
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
        rows = [["ed_220C_2p02MPa_lith0p85_vacA", "path/to/electrode.cif", "220", "2.02", "0.85", "vacA", "electrode", "omat", "", ""]]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main_template() -> None:
    parser = argparse.ArgumentParser(
        prog="hrw-uma-torchsim-manifest-template",
        description="Write a template manifest CSV for UMA TorchSim screening.",
    )
    parser.add_argument("--phase", choices=["electrode", "electrolyte"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--electrolyte-structures-root",
        type=Path,
        default=None,
        help=(
            "If provided with --phase electrolyte, generate a full default manifest by "
            "parsing the electrolyte directory tree instead of writing a one-row template."
        ),
    )
    args = parser.parse_args()
    if args.phase == "electrolyte" and args.electrolyte_structures_root is not None:
        generate_default_electrolyte_manifest_from_tree(args.electrolyte_structures_root, args.out)
    else:
        write_template_manifest(args.out, args.phase)


if __name__ == "__main__":
    main_template()
