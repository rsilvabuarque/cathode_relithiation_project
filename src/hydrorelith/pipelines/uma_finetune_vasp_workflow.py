from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator


@dataclass(slots=True)
class LabeledCase:
    case_id: str
    case_dir: Path
    extxyz_path: Path
    natoms: int
    temperature_k: int | None
    lithiation_pct: float | None
    delithiation_pct: float | None
    pressure_mpa: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-uma-finetune-vasp-workflow",
        description=(
            "Prepare VASP-output fine-tuning datasets, run UMA fine-tuning (omat), "
            "and analyze pre/post model quality."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-dataset",
        help="Create labeled extxyz dataset and 90/5/5 split from vasp_workflow cases.",
    )
    prepare.add_argument("--cases-root", type=Path, required=True, help="Path to workflow root or directly to cases/.")
    prepare.add_argument("--output-dir", type=Path, required=True, help="Output directory for split manifests and extxyz files.")
    prepare.add_argument("--train-frac", type=float, default=0.90)
    prepare.add_argument("--val-frac", type=float, default=0.05)
    prepare.add_argument("--test-frac", type=float, default=0.05)
    prepare.add_argument("--seed", type=int, default=7)
    prepare.add_argument("--max-cases", type=int, default=None)
    prepare.add_argument(
        "--case-list",
        type=Path,
        default=None,
        help="Optional text file with one case directory path per line; when provided, only these cases are included.",
    )
    prepare.add_argument(
        "--require-completed",
        action="store_true",
        default=True,
        help="Only include completed VASP cases (default true).",
    )
    prepare.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Include incomplete cases when final VASP energies/forces are still readable.",
    )

    finetune = subparsers.add_parser(
        "run-finetune",
        help="Run fairchem UMA fine-tuning from prepared split directories.",
    )
    finetune.add_argument("--dataset-dir", type=Path, required=True, help="Directory produced by prepare-dataset.")
    finetune.add_argument("--output-dir", type=Path, required=True, help="Output directory for LMDBs and generated YAML configs.")
    finetune.add_argument("--uma-task", type=str, default="omat")
    finetune.add_argument("--regression-tasks", choices=["e", "ef", "efs"], default="ef")
    finetune.add_argument("--base-model", type=str, default="uma-s-1p2")
    finetune.add_argument("--num-workers", type=int, default=8)
    finetune.add_argument("--epochs", type=int, default=8)
    finetune.add_argument("--batch-size", type=int, default=4)
    finetune.add_argument("--lr", type=float, default=2e-4)
    finetune.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    finetune.add_argument("--run-dir", type=Path, default=None, help="fairchem job.run_dir override.")
    finetune.add_argument("--run-id", type=str, default=None, help="fairchem timestamp id override.")
    finetune.add_argument("--dry-run", action="store_true", help="Print commands without executing training.")

    analyze = subparsers.add_parser(
        "analyze-pre-post",
        help="Compare baseline UMA vs fine-tuned checkpoint against VASP labels.",
    )
    analyze.add_argument("--dataset-dir", type=Path, required=True, help="Directory produced by prepare-dataset.")
    analyze.add_argument("--output-dir", type=Path, required=True, help="Output directory for metrics/plots.")
    analyze.add_argument("--fine-tuned-checkpoint", type=Path, required=True, help="Path to inference_ckpt.pt")
    analyze.add_argument("--base-model", type=str, default="uma-s-1p2")
    analyze.add_argument("--task-name", type=str, default="omat")
    analyze.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    analyze.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    analyze.add_argument("--max-cases", type=int, default=None)
    analyze.add_argument(
        "--training-log",
        type=Path,
        default=None,
        help="Optional fairchem/slurm log file to parse training/validation loss vs step.",
    )

    analyze_ckpt = subparsers.add_parser(
        "analyze-checkpoints",
        help="Analyze one or more checkpoints from a run directory for periodic monitoring.",
    )
    analyze_ckpt.add_argument("--dataset-dir", type=Path, required=True)
    analyze_ckpt.add_argument("--run-dir", type=Path, required=True, help="Directory containing run subdirs with checkpoints.")
    analyze_ckpt.add_argument("--output-root", type=Path, required=True, help="Root directory to write checkpoint-wise analyses.")
    analyze_ckpt.add_argument("--base-model", type=str, default="uma-s-1p2")
    analyze_ckpt.add_argument("--task-name", type=str, default="omat")
    analyze_ckpt.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    analyze_ckpt.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    analyze_ckpt.add_argument("--max-cases", type=int, default=None)
    analyze_ckpt.add_argument("--latest-only", action="store_true", help="Only analyze the newest checkpoint.")
    analyze_ckpt.add_argument("--max-checkpoints", type=int, default=None, help="If set, only analyze the newest N checkpoints.")
    analyze_ckpt.add_argument(
        "--training-log",
        type=Path,
        default=None,
        help="Optional fairchem/slurm log file to parse training/validation loss vs step.",
    )

    all_cmd = subparsers.add_parser(
        "run-all",
        help="Run prepare-dataset -> run-finetune -> analyze-pre-post in sequence.",
    )
    all_cmd.add_argument("--cases-root", type=Path, required=True)
    all_cmd.add_argument("--work-dir", type=Path, required=True, help="Root to store dataset, finetune, and analysis outputs.")
    all_cmd.add_argument("--seed", type=int, default=7)
    all_cmd.add_argument("--train-frac", type=float, default=0.90)
    all_cmd.add_argument("--val-frac", type=float, default=0.05)
    all_cmd.add_argument("--test-frac", type=float, default=0.05)
    all_cmd.add_argument("--uma-task", type=str, default="omat")
    all_cmd.add_argument("--regression-tasks", choices=["e", "ef", "efs"], default="ef")
    all_cmd.add_argument("--base-model", type=str, default="uma-s-1p2")
    all_cmd.add_argument("--num-workers", type=int, default=8)
    all_cmd.add_argument("--epochs", type=int, default=8)
    all_cmd.add_argument("--batch-size", type=int, default=4)
    all_cmd.add_argument("--lr", type=float, default=2e-4)
    all_cmd.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    all_cmd.add_argument("--run-id", type=str, default=None)
    all_cmd.add_argument("--dry-run", action="store_true")

    return parser


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


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


def _tail_text(path: Path, max_bytes: int = 64_000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-max_bytes:].decode("utf-8", errors="ignore")


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


def _parse_case_meta(case_dir: Path) -> tuple[float | None, int | None, float | None]:
    lithiation = None
    temperature = None
    pressure = None
    text = "/".join(case_dir.parts)
    case_name = case_dir.name

    l_match = re.search(r"lith_([0-9]*\.?[0-9]+)pct", case_name) or re.search(r"lith_([0-9]*\.?[0-9]+)pct", text)
    if l_match:
        lithiation = float(l_match.group(1))

    t_match = re.search(r"T_(\d+)K", case_name) or re.search(r"T_(\d+)K", text)
    if t_match:
        temperature = int(t_match.group(1))

    p_match = re.search(r"P_([0-9]+(?:p[0-9]+)?)MPa", case_name) or re.search(r"P_([0-9]+(?:p[0-9]+)?)MPa", text)
    if p_match:
        pressure = float(p_match.group(1).replace("p", "."))
    return lithiation, temperature, pressure


def _extract_labeled_case(case_dir: Path, split_extxyz_dir: Path) -> LabeledCase | None:
    try:
        images = _read_vasp_images(case_dir)
    except Exception:
        return None
    if not images:
        return None

    final_atoms = images[-1].copy()
    final_atoms.calc = None
    try:
        energy = float(final_atoms.get_potential_energy())
        forces = np.asarray(final_atoms.get_forces(), dtype=float)
    except Exception:
        try:
            final_atoms = ase.io.read(str(case_dir / "OUTCAR"), index=-1, format="vasp-out")
            energy = float(final_atoms.get_potential_energy())
            forces = np.asarray(final_atoms.get_forces(), dtype=float)
        except Exception:
            return None

    final_atoms.calc = SinglePointCalculator(final_atoms, energy=energy, forces=forces)

    lithiation_pct, temperature_k, pressure_mpa = _parse_case_meta(case_dir)
    delithiation_pct = (100.0 - float(lithiation_pct)) if lithiation_pct is not None else None
    case_id = case_dir.name
    extxyz_path = split_extxyz_dir / f"{_safe_id(case_id)}.extxyz"
    ase.io.write(str(extxyz_path), final_atoms, format="extxyz")
    return LabeledCase(
        case_id=case_id,
        case_dir=case_dir,
        extxyz_path=extxyz_path,
        natoms=len(final_atoms),
        temperature_k=temperature_k,
        lithiation_pct=lithiation_pct,
        delithiation_pct=delithiation_pct,
        pressure_mpa=pressure_mpa,
    )


def _round_split_counts(n_total: int, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    if n_total <= 0:
        return 0, 0, 0
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("train/val/test fractions must sum to 1.0")

    raw = [n_total * train_frac, n_total * val_frac, n_total * test_frac]
    floors = [int(math.floor(v)) for v in raw]
    remainder = n_total - sum(floors)
    frac = sorted(enumerate([r - f for r, f in zip(raw, floors)]), key=lambda x: x[1], reverse=True)
    for idx, _ in frac[:remainder]:
        floors[idx] += 1
    return floors[0], floors[1], floors[2]


def _split_records_stratified(
    records: list[dict[str, object]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    n_train, n_val, n_test = _round_split_counts(len(records), train_frac, val_frac, test_frac)

    grouped: dict[tuple[int | None, float | None], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        key = (
            int(rec["temperature_k"]) if rec.get("temperature_k") is not None else None,
            float(rec["delithiation_pct"]) if rec.get("delithiation_pct") is not None else None,
        )
        grouped[key].append(rec)

    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)

    allocations: dict[tuple[int | None, float | None], dict[str, int]] = {}
    for key, group in grouped.items():
        g_train, g_val, g_test = _round_split_counts(len(group), train_frac, val_frac, test_frac)
        allocations[key] = {"train": g_train, "val": g_val, "test": g_test}

    def _rebalance(target_name: str, target: int) -> None:
        current = sum(allocations[k][target_name] for k in allocations)
        if current == target:
            return
        keys = list(allocations.keys())
        rng.shuffle(keys)

        if current < target:
            need = target - current
            order = ["train", "val", "test"]
            donors = [d for d in order if d != target_name]
            for donor in donors:
                for key in keys:
                    if need <= 0:
                        break
                    if allocations[key][donor] > 0:
                        allocations[key][donor] -= 1
                        allocations[key][target_name] += 1
                        need -= 1
                if need <= 0:
                    break
        else:
            extra = current - target
            receivers = [n for n in ["train", "val", "test"] if n != target_name]
            for recv in receivers:
                for key in keys:
                    if extra <= 0:
                        break
                    if allocations[key][target_name] > 0:
                        allocations[key][target_name] -= 1
                        allocations[key][recv] += 1
                        extra -= 1
                if extra <= 0:
                    break

    _rebalance("train", n_train)
    _rebalance("val", n_val)
    _rebalance("test", n_test)

    splits = {"train": [], "val": [], "test": []}
    for key, group in grouped.items():
        cursor = 0
        for split_name in ["train", "val", "test"]:
            take = allocations[key][split_name]
            if take <= 0:
                continue
            splits[split_name].extend(group[cursor : cursor + take])
            cursor += take

    for split_name in ["train", "val", "test"]:
        rng.shuffle(splits[split_name])
    return splits


def _ensure_diversity_by_swap(
    splits: dict[str, list[dict[str, object]]],
    field: str,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    rng = random.Random(seed + 19)

    present_values = sorted({rec.get(field) for rec in splits["train"] + splits["val"] + splits["test"] if rec.get(field) is not None})
    if len(present_values) < 2:
        return splits

    def _unique_count(rows: list[dict[str, object]]) -> int:
        return len({r.get(field) for r in rows if r.get(field) is not None})

    def _swap_in(target_name: str) -> None:
        if _unique_count(splits[target_name]) >= 2:
            return

        missing = [v for v in present_values if v not in {r.get(field) for r in splits[target_name]}]
        if not missing:
            return
        rng.shuffle(missing)

        donor_rows = splits["train"] if target_name != "train" else splits["val"] + splits["test"]
        donor_idx = None
        for want in missing:
            candidates = [i for i, rec in enumerate(donor_rows) if rec.get(field) == want]
            if candidates:
                donor_idx = rng.choice(candidates)
                break
        if donor_idx is None:
            return

        donor_rec = donor_rows.pop(donor_idx)
        if target_name == "train":
            if splits["val"]:
                moved = splits["val"].pop()
                splits["train"].append(moved)
                splits["val"].append(donor_rec)
            elif splits["test"]:
                moved = splits["test"].pop()
                splits["train"].append(moved)
                splits["test"].append(donor_rec)
            else:
                splits["train"].append(donor_rec)
        else:
            if not splits[target_name]:
                splits[target_name].append(donor_rec)
                return
            target_idx = rng.randrange(len(splits[target_name]))
            moved = splits[target_name][target_idx]
            splits[target_name][target_idx] = donor_rec
            splits["train"].append(moved)

    _swap_in("train")
    _swap_in("val")
    _swap_in("test")
    return splits


def _write_split_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "case_id",
        "case_dir",
        "natoms",
        "temperature_k",
        "pressure_mpa",
        "lithiation_pct",
        "delithiation_pct",
        "extxyz_path",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _summarize_split(rows: list[dict[str, object]]) -> dict[str, object]:
    temps = sorted({int(r["temperature_k"]) for r in rows if r.get("temperature_k") is not None})
    delith = sorted({float(r["delithiation_pct"]) for r in rows if r.get("delithiation_pct") is not None})
    temp_counts: dict[str, int] = defaultdict(int)
    delith_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("temperature_k") is not None:
            temp_counts[str(int(row["temperature_k"]))] += 1
        if row.get("delithiation_pct") is not None:
            delith_counts[f"{float(row['delithiation_pct']):.2f}"] += 1
    return {
        "n_cases": len(rows),
        "unique_temperatures_k": temps,
        "n_unique_temperatures": len(temps),
        "unique_delithiation_pct": delith,
        "n_unique_delithiation": len(delith),
        "counts_by_temperature": dict(sorted(temp_counts.items())),
        "counts_by_delithiation_pct": dict(sorted(delith_counts.items())),
    }


def cmd_prepare_dataset(args: argparse.Namespace) -> None:
    if args.allow_incomplete:
        args.require_completed = False

    cases_root = _resolve_cases_root(args.cases_root)
    case_dirs = _discover_case_dirs(cases_root)

    if args.case_list is not None:
        lines = args.case_list.read_text(encoding="utf-8").splitlines()
        allowed = {
            str(Path(line.strip()).resolve())
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        }
        case_dirs = [case_dir for case_dir in case_dirs if str(case_dir.resolve()) in allowed]

    if args.max_cases is not None:
        case_dirs = case_dirs[: max(0, int(args.max_cases))]

    out_dir = args.output_dir
    split_dir = out_dir / "split"
    extxyz_root = out_dir / "extxyz"
    raw_dir = extxyz_root / "raw"
    split_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    skipped_unfinished = 0
    skipped_unreadable = 0
    for case_dir in case_dirs:
        if args.require_completed and not _is_completed(case_dir):
            skipped_unfinished += 1
            continue

        labeled = _extract_labeled_case(case_dir, raw_dir)
        if labeled is None:
            skipped_unreadable += 1
            continue

        records.append(
            {
                "case_id": labeled.case_id,
                "case_dir": str(labeled.case_dir),
                "natoms": labeled.natoms,
                "temperature_k": labeled.temperature_k,
                "pressure_mpa": labeled.pressure_mpa,
                "lithiation_pct": labeled.lithiation_pct,
                "delithiation_pct": labeled.delithiation_pct,
                "extxyz_path": str(labeled.extxyz_path),
            }
        )

    if not records:
        raise RuntimeError("No usable VASP-labeled cases found. Check --cases-root and completion/readability filters.")

    splits = _split_records_stratified(
        records,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )
    splits = _ensure_diversity_by_swap(splits, field="temperature_k", seed=int(args.seed))
    splits = _ensure_diversity_by_swap(splits, field="delithiation_pct", seed=int(args.seed))

    for split_name in ["train", "val", "test"]:
        split_extxyz_dir = extxyz_root / split_name
        split_extxyz_dir.mkdir(parents=True, exist_ok=True)
        for row in splits[split_name]:
            src = Path(str(row["extxyz_path"]))
            dst = split_extxyz_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            row["extxyz_path"] = str(dst)
            row["split"] = split_name

    for split_name in ["train", "val", "test"]:
        _write_split_manifest(split_dir / f"{split_name}.csv", splits[split_name])

    all_rows = splits["train"] + splits["val"] + splits["test"]
    _write_split_manifest(split_dir / "all.csv", all_rows)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cases_root": str(cases_root),
        "output_dir": str(out_dir),
        "fractions": {
            "train": float(args.train_frac),
            "val": float(args.val_frac),
            "test": float(args.test_frac),
        },
        "seed": int(args.seed),
        "n_total_candidates": len(case_dirs),
        "n_usable_cases": len(all_rows),
        "n_skipped_unfinished": skipped_unfinished,
        "n_skipped_unreadable": skipped_unreadable,
        "split_summary": {
            "train": _summarize_split(splits["train"]),
            "val": _summarize_split(splits["val"]),
            "test": _summarize_split(splits["test"]),
        },
        "split_manifests": {
            "train": str(split_dir / "train.csv"),
            "val": str(split_dir / "val.csv"),
            "test": str(split_dir / "test.csv"),
            "all": str(split_dir / "all.csv"),
        },
        "extxyz_dirs": {
            "train": str(extxyz_root / "train"),
            "val": str(extxyz_root / "val"),
            "test": str(extxyz_root / "test"),
        },
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["split_summary"], indent=2))
    print(f"Prepared dataset under {out_dir}")


def _load_rows(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    for row in rows:
        for k in ["natoms", "temperature_k"]:
            if row.get(k):
                row[k] = int(float(row[k]))
            else:
                row[k] = None
        for k in ["pressure_mpa", "lithiation_pct", "delithiation_pct"]:
            if row.get(k):
                row[k] = float(row[k])
            else:
                row[k] = None
    return rows


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def cmd_run_finetune(args: argparse.Namespace) -> None:
    train_dir = args.dataset_dir.resolve() / "extxyz" / "train"
    val_dir = args.dataset_dir.resolve() / "extxyz" / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Could not find split directories under {args.dataset_dir}. Expected extxyz/train and extxyz/val."
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists. Choose a fresh --output-dir as required by fairchem script.")

    py_exe = sys.executable
    create_cmd = [
        py_exe,
        "-m",
        "fairchem.core.scripts.create_uma_finetune_dataset",
        "--train-dir",
        str(train_dir),
        "--val-dir",
        str(val_dir),
        "--uma-task",
        str(args.uma_task),
        "--regression-tasks",
        str(args.regression_tasks),
        "--base-model",
        str(args.base_model),
        "--output-dir",
        str(output_dir),
        "--num-workers",
        str(int(args.num_workers)),
    ]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("finetune_%Y%m%d_%H%M%S")
    run_dir = (args.run_dir.resolve() if args.run_dir is not None else (output_dir / "runs").resolve())
    device = _resolve_device(args.device)
    device_override = "CUDA" if device == "cuda" else "CPU"

    train_yaml = output_dir / "uma_sm_finetune_template.yaml"
    train_cmd = [
        "fairchem",
        "-c",
        str(train_yaml.resolve()),
        f"job.run_dir={run_dir}",
        f"+job.timestamp_id={run_id}",
        f"epochs={int(args.epochs)}",
        f"batch_size={int(args.batch_size)}",
        f"lr={float(args.lr)}",
        f"job.device_type={device_override}",
        "job.debug=True",
    ]

    if args.dry_run:
        payload = {
            "create_dataset_cmd": create_cmd,
            "finetune_cmd": train_cmd,
            "output_dir": str(output_dir),
            "run_dir": str(run_dir),
            "run_id": run_id,
        }
        print(json.dumps(payload, indent=2))
        return

    subprocess.run(create_cmd, check=True)
    subprocess.run(train_cmd, check=True)

    checkpoint_matches = sorted((run_dir / run_id).glob("checkpoints/**/inference_ckpt.pt"))
    checkpoint_path = checkpoint_matches[-1] if checkpoint_matches else None

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "run_id": run_id,
        "resolved_device": device,
        "create_dataset_cmd": create_cmd,
        "finetune_cmd": train_cmd,
        "generated_yaml": str(train_yaml),
        "inference_checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }
    (output_dir / "finetune_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _force_metrics(vasp_forces: np.ndarray, pred_forces: np.ndarray) -> tuple[float, float, float]:
    delta = pred_forces - vasp_forces
    norms = np.linalg.norm(delta, axis=1)
    return float(np.mean(norms)), float(np.sqrt(np.mean(norms * norms))), float(np.max(norms))


def _build_calculator_pretrained(base_model: str, task_name: str, device: str):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(base_model, device=device)
    return FAIRChemCalculator(predictor, task_name=task_name)


def _build_calculator_checkpoint(ckpt: Path, task_name: str, device: str):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(str(ckpt), device=device)
    return FAIRChemCalculator(predictor, task_name=task_name)


def _predict_with_calc(calc, atoms):
    copy_atoms = atoms.copy()
    copy_atoms.calc = calc
    energy = float(copy_atoms.get_potential_energy())
    forces = np.asarray(copy_atoms.get_forces(), dtype=float)
    return energy, forces


def _metric_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}

    def _mae(key: str) -> float:
        arr = np.array([float(r[key]) for r in rows], dtype=float)
        return float(np.mean(np.abs(arr)))

    return {
        "energy_mae_eV": _mae("delta_energy_eV"),
        "energy_rmse_eV": float(
            np.sqrt(np.mean(np.square(np.array([float(r["delta_energy_eV"]) for r in rows], dtype=float))))
        ),
        "energy_per_atom_mae_eV": _mae("delta_energy_per_atom_eV"),
        "energy_per_atom_rmse_eV": float(
            np.sqrt(np.mean(np.square(np.array([float(r["delta_energy_per_atom_eV"]) for r in rows], dtype=float))))
        ),
        "mean_force_difference_mae_eV_per_A": _mae("mean_force_difference_eV_per_A"),
        "mean_force_difference_rmse_eV_per_A": float(
            np.sqrt(np.mean(np.square(np.array([float(r["mean_force_difference_eV_per_A"]) for r in rows], dtype=float))))
        ),
        "rms_force_difference_mae_eV_per_A": _mae("rms_force_difference_eV_per_A"),
        "rms_force_difference_rmse_eV_per_A": float(
            np.sqrt(np.mean(np.square(np.array([float(r["rms_force_difference_eV_per_A"]) for r in rows], dtype=float))))
        ),
        "max_force_difference_mae_eV_per_A": _mae("max_force_difference_eV_per_A"),
        "max_force_difference_rmse_eV_per_A": float(
            np.sqrt(np.mean(np.square(np.array([float(r["max_force_difference_eV_per_A"]) for r in rows], dtype=float))))
        ),
    }


def _compare_records(
    rows: list[dict[str, object]],
    base_calc,
    ft_calc,
    max_cases: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    eval_rows = rows[: max_cases] if max_cases is not None else rows
    out: list[dict[str, object]] = []

    for row in eval_rows:
        extxyz_path = Path(str(row["extxyz_path"]))
        atoms = ase.io.read(str(extxyz_path), index=-1, format="extxyz")

        vasp_energy = float(atoms.get_potential_energy())
        vasp_forces = np.asarray(atoms.get_forces(), dtype=float)
        natoms = int(len(atoms))

        base_energy, base_forces = _predict_with_calc(base_calc, atoms)
        base_mean_df, base_rms_df, base_max_df = _force_metrics(vasp_forces, base_forces)

        ft_energy, ft_forces = _predict_with_calc(ft_calc, atoms)
        ft_mean_df, ft_rms_df, ft_max_df = _force_metrics(vasp_forces, ft_forces)
        vasp_mean_force_abs = float(np.mean(np.linalg.norm(vasp_forces, axis=1)))
        base_mean_force_abs = float(np.mean(np.linalg.norm(base_forces, axis=1)))
        ft_mean_force_abs = float(np.mean(np.linalg.norm(ft_forces, axis=1)))

        out.append(
            {
                "case_id": row["case_id"],
                "case_dir": row["case_dir"],
                "split": row["split"],
                "natoms": natoms,
                "temperature_k": row.get("temperature_k"),
                "lithiation_pct": row.get("lithiation_pct"),
                "delithiation_pct": row.get("delithiation_pct"),
                "vasp_energy_eV": vasp_energy,
                "base_energy_eV": base_energy,
                "ft_energy_eV": ft_energy,
                "base_delta_energy_eV": base_energy - vasp_energy,
                "ft_delta_energy_eV": ft_energy - vasp_energy,
                "base_delta_energy_per_atom_eV": (base_energy - vasp_energy) / max(1, natoms),
                "ft_delta_energy_per_atom_eV": (ft_energy - vasp_energy) / max(1, natoms),
                "vasp_mean_force_abs_eV_per_A": vasp_mean_force_abs,
                "base_mean_force_abs_eV_per_A": base_mean_force_abs,
                "ft_mean_force_abs_eV_per_A": ft_mean_force_abs,
                "base_mean_force_difference_eV_per_A": base_mean_df,
                "ft_mean_force_difference_eV_per_A": ft_mean_df,
                "base_rms_force_difference_eV_per_A": base_rms_df,
                "ft_rms_force_difference_eV_per_A": ft_rms_df,
                "base_max_force_difference_eV_per_A": base_max_df,
                "ft_max_force_difference_eV_per_A": ft_max_df,
            }
        )

    base_metrics = _metric_summary(
        [
            {
                "delta_energy_eV": r["base_delta_energy_eV"],
                "delta_energy_per_atom_eV": r["base_delta_energy_per_atom_eV"],
                "mean_force_difference_eV_per_A": r["base_mean_force_difference_eV_per_A"],
                "rms_force_difference_eV_per_A": r["base_rms_force_difference_eV_per_A"],
                "max_force_difference_eV_per_A": r["base_max_force_difference_eV_per_A"],
            }
            for r in out
        ]
    )
    ft_metrics = _metric_summary(
        [
            {
                "delta_energy_eV": r["ft_delta_energy_eV"],
                "delta_energy_per_atom_eV": r["ft_delta_energy_per_atom_eV"],
                "mean_force_difference_eV_per_A": r["ft_mean_force_difference_eV_per_A"],
                "rms_force_difference_eV_per_A": r["ft_rms_force_difference_eV_per_A"],
                "max_force_difference_eV_per_A": r["ft_max_force_difference_eV_per_A"],
            }
            for r in out
        ]
    )
    improvement = {
        key: (base_metrics[key] - ft_metrics.get(key, np.nan)) if key in base_metrics else np.nan
        for key in base_metrics
    }
    return out, {"baseline": base_metrics, "finetuned": ft_metrics, "improvement": improvement}


def _parse_training_history(log_path: Path) -> list[dict[str, float]]:
    if not log_path.exists():
        return []

    rows: list[dict[str, float]] = []
    current_step: float | None = None
    train_pattern = re.compile(r"\{.*'train/loss'.*\}")
    val_pattern = re.compile(r"val/loss:\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m_train = train_pattern.search(line)
        if m_train:
            try:
                payload = ast.literal_eval(m_train.group(0))
                step = float(payload.get("train/step"))
                loss = float(payload.get("train/loss"))
                current_step = step
                rows.append({"step": step, "train_loss": loss})
            except Exception:
                pass
            continue

        m_val = val_pattern.search(line)
        if m_val and current_step is not None:
            try:
                rows.append({"step": float(current_step), "val_loss": float(m_val.group(1))})
            except Exception:
                pass

    return rows


def _write_analysis_plots(rows: list[dict[str, object]], out_dir: Path, training_log: Path | None = None) -> None:
    if not rows:
        return

    x = np.array([float(r.get("delithiation_pct", np.nan)) for r in rows], dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        x = np.array([float(r.get("temperature_k", np.nan)) for r in rows], dtype=float)
        valid = np.isfinite(x)
        xlabel = "Temperature (K)"
        suffix = "temperature"
    else:
        xlabel = "Delithiation (%)"
        suffix = "delithiation"

    metrics = [
        (
            "delta_energy_per_atom_eV",
            "|ΔE| per atom (eV)",
            "energy_per_atom",
            "base_delta_energy_per_atom_eV",
            "ft_delta_energy_per_atom_eV",
        ),
        (
            "mean_force_difference_eV_per_A",
            "Mean |ΔF| (eV/Å)",
            "mean_force",
            "base_mean_force_difference_eV_per_A",
            "ft_mean_force_difference_eV_per_A",
        ),
        (
            "rms_force_difference_eV_per_A",
            "RMS |ΔF| (eV/Å)",
            "rms_force",
            "base_rms_force_difference_eV_per_A",
            "ft_rms_force_difference_eV_per_A",
        ),
    ]

    for _, ylabel, name, base_key, ft_key in metrics:
        base = np.array([abs(float(r[base_key])) for r in rows], dtype=float)
        ft = np.array([abs(float(r[ft_key])) for r in rows], dtype=float)
        mask = valid & np.isfinite(base) & np.isfinite(ft)
        if not np.any(mask):
            continue

        plt.figure(figsize=(8, 5))
        plt.scatter(x[mask], base[mask], alpha=0.25, s=16, label="Pre-finetune UMA")
        plt.scatter(x[mask], ft[mask], alpha=0.25, s=16, label="Post-finetune UMA")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.2)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"pre_post_{name}_vs_{suffix}.png", dpi=220)
        plt.close()

    def _write_parity_plot(xvals: np.ndarray, yvals: np.ndarray, title: str, xlabel: str, ylabel: str, filename: str) -> None:
        mask = np.isfinite(xvals) & np.isfinite(yvals)
        if not np.any(mask):
            return
        x_use = xvals[mask]
        y_use = yvals[mask]
        residual = y_use - x_use
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        ss_tot = float(np.sum(np.square(x_use - np.mean(x_use))))
        ss_res = float(np.sum(np.square(residual)))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        lo = float(min(np.min(x_use), np.min(y_use)))
        hi = float(max(np.max(x_use), np.max(y_use)))
        pad = max(1e-8, 0.03 * (hi - lo if hi > lo else 1.0))

        plt.figure(figsize=(6.5, 6.0))
        plt.scatter(x_use, y_use, s=16, alpha=0.3)
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0)
        plt.xlim(lo - pad, hi + pad)
        plt.ylim(lo - pad, hi + pad)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        metric_text = f"RMSE = {rmse:.4f}\\nR² = {r2:.4f}" if np.isfinite(r2) else f"RMSE = {rmse:.4f}\\nR² = n/a"
        plt.text(
            0.03,
            0.97,
            metric_text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()

    vasp_e = np.array([float(r["vasp_energy_eV"]) for r in rows], dtype=float)
    base_e = np.array([float(r["base_energy_eV"]) for r in rows], dtype=float)
    ft_e = np.array([float(r["ft_energy_eV"]) for r in rows], dtype=float)
    _write_parity_plot(
        vasp_e,
        base_e,
        title="Parity: Baseline energy vs VASP",
        xlabel="VASP energy (eV)",
        ylabel="Baseline UMA energy (eV)",
        filename="parity_energy_baseline_vs_vasp.png",
    )
    _write_parity_plot(
        vasp_e,
        ft_e,
        title="Parity: Fine-tuned energy vs VASP",
        xlabel="VASP energy (eV)",
        ylabel="Fine-tuned UMA energy (eV)",
        filename="parity_energy_finetuned_vs_vasp.png",
    )

    vasp_f = np.array([float(r["vasp_mean_force_abs_eV_per_A"]) for r in rows], dtype=float)
    base_f = np.array([float(r["base_mean_force_abs_eV_per_A"]) for r in rows], dtype=float)
    ft_f = np.array([float(r["ft_mean_force_abs_eV_per_A"]) for r in rows], dtype=float)
    _write_parity_plot(
        vasp_f,
        base_f,
        title="Parity: Baseline mean|F| vs VASP",
        xlabel="VASP mean |F| (eV/Å)",
        ylabel="Baseline UMA mean |F| (eV/Å)",
        filename="parity_force_mean_abs_baseline_vs_vasp.png",
    )
    _write_parity_plot(
        vasp_f,
        ft_f,
        title="Parity: Fine-tuned mean|F| vs VASP",
        xlabel="VASP mean |F| (eV/Å)",
        ylabel="Fine-tuned UMA mean |F| (eV/Å)",
        filename="parity_force_mean_abs_finetuned_vs_vasp.png",
    )

    if training_log is not None:
        hist = _parse_training_history(training_log)
        if hist:
            hist_csv = out_dir / "training_history_parsed.csv"
            fields = sorted({k for row in hist for k in row.keys()})
            with hist_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in hist:
                    writer.writerow(row)

            def _loss_filter_start(losses_in: np.ndarray) -> int:
                npts = int(losses_in.size)
                if npts <= 5:
                    return 0
                warmup_min = max(3, int(0.02 * npts))
                high_loss_threshold = float(np.quantile(losses_in, 0.9))
                below = np.where(losses_in[warmup_min:] <= high_loss_threshold)[0]
                if below.size == 0:
                    return warmup_min
                return int(warmup_min + below[0])

            def _convergence_assessment(steps_in: np.ndarray, losses_in: np.ndarray) -> tuple[float, float, float, str]:
                npts = int(losses_in.size)
                if npts < 4:
                    return float("nan"), float("nan"), float("nan"), "Insufficient points"

                tail_n = max(4, int(0.2 * npts))
                tail_x = steps_in[-tail_n:]
                tail_y = losses_in[-tail_n:]
                if float(np.ptp(tail_x)) <= 0.0:
                    slope = 0.0
                else:
                    slope = float(np.polyfit(tail_x, tail_y, 1)[0])
                slope_per_1k = 1000.0 * slope
                rel_tail_change = float((tail_y[0] - tail_y[-1]) / max(abs(float(tail_y[0])), 1e-12))
                tail_cv = float(np.std(tail_y) / max(abs(float(np.mean(tail_y))), 1e-12))

                if slope_per_1k < -0.01 and rel_tail_change > 0.03:
                    assessment = "Still improving; longer run may help"
                elif abs(slope_per_1k) <= 0.01 and abs(rel_tail_change) <= 0.03:
                    assessment = "Near plateau; likely converged"
                else:
                    assessment = "Noisy/slow improvement"
                return slope_per_1k, rel_tail_change, tail_cv, assessment

            def _write_loss_dual_plot(
                steps_in: np.ndarray,
                losses_in: np.ndarray,
                title: str,
                xlabel: str,
                ylabel: str,
                filename: str,
                marker: str | None = None,
            ) -> None:
                mask = np.isfinite(steps_in) & np.isfinite(losses_in)
                if not np.any(mask):
                    return

                steps_use = steps_in[mask]
                losses_use = losses_in[mask]
                cutoff_idx = _loss_filter_start(losses_use)
                steps_f = steps_use[cutoff_idx:]
                losses_f = losses_use[cutoff_idx:]

                slope_per_1k, rel_tail_change, tail_cv, assessment = _convergence_assessment(steps_f, losses_f)

                fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.5))

                if marker is None:
                    axes[0].plot(steps_use, losses_use, linewidth=1.4)
                    axes[1].plot(steps_f, losses_f, linewidth=1.4)
                else:
                    axes[0].plot(steps_use, losses_use, marker=marker, linewidth=1.1)
                    axes[1].plot(steps_f, losses_f, marker=marker, linewidth=1.1)

                axes[0].axvline(float(steps_use[cutoff_idx]), color="k", linestyle="--", linewidth=1.0, alpha=0.6)
                axes[0].set_title(f"{title} (full)")
                axes[0].set_xlabel(xlabel)
                axes[0].set_ylabel(ylabel)
                axes[0].grid(alpha=0.2)

                axes[1].set_title(f"{title} (filtered after warmup)")
                axes[1].set_xlabel(xlabel)
                axes[1].set_ylabel(ylabel)
                axes[1].grid(alpha=0.2)

                metrics_text = (
                    f"Tail slope: {slope_per_1k:.4f} loss/1k steps\\n"
                    f"Tail Δloss: {100.0 * rel_tail_change:.2f}%\\n"
                    f"Tail CV: {tail_cv:.3f}\\n"
                    f"Assessment: {assessment}"
                )
                axes[1].text(
                    0.02,
                    0.98,
                    metrics_text,
                    transform=axes[1].transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
                )

                fig.tight_layout()
                fig.savefig(out_dir / filename, dpi=220)
                plt.close(fig)

            train_rows = [r for r in hist if "train_loss" in r and "step" in r]
            if train_rows:
                steps = np.array([float(r["step"]) for r in train_rows], dtype=float)
                losses = np.array([float(r["train_loss"]) for r in train_rows], dtype=float)
                _write_loss_dual_plot(
                    steps,
                    losses,
                    title="Train loss vs step",
                    xlabel="Training step",
                    ylabel="Train loss",
                    filename="train_loss_vs_step.png",
                )

            val_rows = [r for r in hist if "val_loss" in r and "step" in r]
            if val_rows:
                steps = np.array([float(r["step"]) for r in val_rows], dtype=float)
                losses = np.array([float(r["val_loss"]) for r in val_rows], dtype=float)
                _write_loss_dual_plot(
                    steps,
                    losses,
                    title="Validation loss vs step",
                    xlabel="Training step (nearest eval)",
                    ylabel="Validation loss",
                    filename="val_loss_vs_step.png",
                    marker="o",
                )


def cmd_analyze_pre_post(args: argparse.Namespace) -> None:
    split_dir = args.dataset_dir / "split"
    if args.split == "all":
        rows = _load_rows(split_dir / "all.csv")
    else:
        rows = _load_rows(split_dir / f"{args.split}.csv")
    if not rows:
        raise RuntimeError(f"No rows found for split '{args.split}' in {split_dir}")

    resolved_device = _resolve_device(args.device)
    base_calc = _build_calculator_pretrained(args.base_model, args.task_name, resolved_device)
    ft_calc = _build_calculator_checkpoint(Path(args.fine_tuned_checkpoint), args.task_name, resolved_device)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case, summary = _compare_records(rows, base_calc, ft_calc, args.max_cases)

    csv_path = out_dir / "pre_post_metrics.csv"
    if per_case:
        fields = list(per_case[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in per_case:
                writer.writerow(row)

    by_temp: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_delith: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_case:
        if row.get("temperature_k") is not None:
            by_temp[str(int(row["temperature_k"]))].append(row)
        if row.get("delithiation_pct") is not None:
            by_delith[f"{float(row['delithiation_pct']):.2f}"].append(row)

    summary_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "analysis_dir": str(out_dir),
        "split": args.split,
        "n_cases": len(per_case),
        "base_model": args.base_model,
        "fine_tuned_checkpoint": str(args.fine_tuned_checkpoint),
        "task_name": args.task_name,
        "resolved_device": resolved_device,
        "overall": summary,
        "by_temperature": {
            key: _metric_summary(
                [
                    {
                        "delta_energy_eV": r["ft_delta_energy_eV"],
                        "delta_energy_per_atom_eV": r["ft_delta_energy_per_atom_eV"],
                        "mean_force_difference_eV_per_A": r["ft_mean_force_difference_eV_per_A"],
                        "rms_force_difference_eV_per_A": r["ft_rms_force_difference_eV_per_A"],
                        "max_force_difference_eV_per_A": r["ft_max_force_difference_eV_per_A"],
                    }
                    for r in rows_group
                ]
            )
            for key, rows_group in sorted(by_temp.items())
        },
        "by_delithiation_pct": {
            key: _metric_summary(
                [
                    {
                        "delta_energy_eV": r["ft_delta_energy_eV"],
                        "delta_energy_per_atom_eV": r["ft_delta_energy_per_atom_eV"],
                        "mean_force_difference_eV_per_A": r["ft_mean_force_difference_eV_per_A"],
                        "rms_force_difference_eV_per_A": r["ft_rms_force_difference_eV_per_A"],
                        "max_force_difference_eV_per_A": r["ft_max_force_difference_eV_per_A"],
                    }
                    for r in rows_group
                ]
            )
            for key, rows_group in sorted(by_delith.items())
        },
        "per_case_csv": str(csv_path),
    }
    (out_dir / "pre_post_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_analysis_plots(per_case, out_dir, args.training_log)
    print(json.dumps(summary_payload["overall"], indent=2))
    print(f"Wrote analysis outputs in {out_dir}")


def cmd_analyze_checkpoints(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    checkpoints = sorted(run_dir.glob("**/checkpoints/**/inference_ckpt.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under {run_dir}")

    if args.latest_only:
        checkpoints = [checkpoints[-1]]
    elif args.max_checkpoints is not None and args.max_checkpoints > 0:
        checkpoints = checkpoints[-int(args.max_checkpoints) :]

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    index_payload: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_dir": str(args.dataset_dir),
        "n_checkpoints_analyzed": len(checkpoints),
        "entries": [],
    }

    for ckpt in checkpoints:
        run_id = ckpt.parents[2].name
        ckpt_step = ckpt.parent.name
        analysis_dir = output_root / f"analysis_{_safe_id(run_id)}_{_safe_id(ckpt_step)}"

        analyze_args = argparse.Namespace(
            dataset_dir=args.dataset_dir,
            output_dir=analysis_dir,
            fine_tuned_checkpoint=ckpt,
            base_model=args.base_model,
            task_name=args.task_name,
            device=args.device,
            split=args.split,
            max_cases=args.max_cases,
            training_log=args.training_log,
        )
        cmd_analyze_pre_post(analyze_args)

        index_payload["entries"].append(
            {
                "run_id": run_id,
                "checkpoint_step_dir": ckpt_step,
                "checkpoint": str(ckpt),
                "analysis_dir": str(analysis_dir),
                "summary_json": str(analysis_dir / "pre_post_summary.json"),
            }
        )

    (output_root / "checkpoint_analysis_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(json.dumps(index_payload, indent=2))


def cmd_run_all(args: argparse.Namespace) -> None:
    work_dir = args.work_dir
    dataset_dir = work_dir / "dataset"
    finetune_dir = work_dir / "finetune"
    analysis_dir = work_dir / "analysis"

    prep_args = argparse.Namespace(
        cases_root=args.cases_root,
        output_dir=dataset_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        max_cases=None,
        require_completed=True,
        allow_incomplete=False,
    )
    cmd_prepare_dataset(prep_args)

    finetune_args = argparse.Namespace(
        dataset_dir=dataset_dir,
        output_dir=finetune_dir,
        uma_task=args.uma_task,
        regression_tasks=args.regression_tasks,
        base_model=args.base_model,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        run_dir=None,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    cmd_run_finetune(finetune_args)

    if args.dry_run:
        print("Dry run: skipping post-finetune analysis because no checkpoint was created.")
        return

    summary_path = finetune_dir / "finetune_run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing {summary_path}; cannot locate fine-tuned checkpoint.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ckpt = summary.get("inference_checkpoint")
    if not ckpt:
        raise RuntimeError("Fine-tune run did not report inference checkpoint path.")

    analyze_args = argparse.Namespace(
        dataset_dir=dataset_dir,
        output_dir=analysis_dir,
        fine_tuned_checkpoint=Path(str(ckpt)),
        base_model=args.base_model,
        task_name=args.uma_task,
        device=args.device,
        split="test",
        max_cases=None,
        training_log=None,
    )
    cmd_analyze_pre_post(analyze_args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-dataset":
        cmd_prepare_dataset(args)
        return
    if args.command == "run-finetune":
        cmd_run_finetune(args)
        return
    if args.command == "analyze-pre-post":
        cmd_analyze_pre_post(args)
        return
    if args.command == "analyze-checkpoints":
        cmd_analyze_checkpoints(args)
        return
    if args.command == "run-all":
        cmd_run_all(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()