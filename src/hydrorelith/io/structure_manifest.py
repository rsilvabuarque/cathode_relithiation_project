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
    args = parser.parse_args()
    write_template_manifest(args.out, args.phase)


if __name__ == "__main__":
    main_template()
