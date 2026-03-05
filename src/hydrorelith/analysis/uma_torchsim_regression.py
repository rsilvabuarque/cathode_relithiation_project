from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


def load_experimental_rates(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rec = dict(row)
            if "log_rate" in rec and rec["log_rate"] not in ("", None):
                rec["log_rate"] = float(rec["log_rate"])
            rows.append(rec)
    return rows


def merge_feature_rows(electrode_rows: list[dict], electrolyte_rows: list[dict], rates: list[dict] | None) -> list[dict]:
    by_key: dict[tuple, dict] = {}

    for row in electrode_rows:
        key = (row.get("condition_id"), row.get("temperature_C"), row.get("pressure_MPa"))
        by_key.setdefault(key, {}).update(row)
    for row in electrolyte_rows:
        key = (row.get("condition_id"), row.get("temperature_C"), row.get("pressure_MPa"))
        by_key.setdefault(key, {}).update(row)

    merged = list(by_key.values())
    if rates:
        rates_by_cond = {r.get("condition_id"): r for r in rates}
        for row in merged:
            r = rates_by_cond.get(row.get("condition_id"))
            if r and "log_rate" in r:
                row["log_rate"] = float(r["log_rate"])
    return merged


def compute_rate_score(row: dict) -> float:
    eps = 1e-12
    d_li = float(row.get("D_li_A2_per_ps", 0.0))
    residence_oh = float(row.get("residence_proxy_oh", 0.0))
    cn_oh = float(row.get("cn_hydroxide_mean", 0.0))
    lioh_m = float(row.get("liOH_M", 0.0))
    vac_acc = float(row.get("vacancy_accessibility_mean", 0.0))
    li_msd_1ps = float(row.get("electrode_li_msd_1ps", 0.0))

    terms = [
        math.log(max(d_li, eps)),
        -math.log(max(residence_oh, eps)),
        -cn_oh,
        math.log(max(lioh_m + eps, eps)),
        vac_acc,
        math.log(max(li_msd_1ps, eps)),
    ]
    return float(sum(terms))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    return 0.0 if den <= 0 else float(np.sum(xm * ym) / den)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def fit_simple_models(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        row["predicted_score"] = compute_rate_score(row)

    pred_vs_true_path = output_dir / "pred_vs_true.csv"
    with pred_vs_true_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["condition_id", "predicted_score", "log_rate", "predicted_log_rate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        score = np.array([float(r["predicted_score"]) for r in rows], dtype=float)
        have_rate = [r for r in rows if "log_rate" in r]
        if have_rate:
            y = np.array([float(r["log_rate"]) for r in have_rate], dtype=float)
            x = np.array([float(r["predicted_score"]) for r in have_rate], dtype=float)
            m, b = np.polyfit(x, y, deg=1)
            pred = m * x + b
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            pearson = _pearson(x, y)
            spearman = _spearman(x, y)
            for r in rows:
                pred_log = float(m * float(r["predicted_score"]) + b)
                writer.writerow(
                    {
                        "condition_id": r.get("condition_id", ""),
                        "predicted_score": float(r["predicted_score"]),
                        "log_rate": r.get("log_rate", ""),
                        "predicted_log_rate": pred_log,
                    }
                )
            summary = {
                "n_rows": len(rows),
                "n_with_rates": len(have_rate),
                "linear_model": {"slope": float(m), "intercept": float(b)},
                "pearson": pearson,
                "spearman": spearman,
                "rmse": rmse,
            }
        else:
            for r in rows:
                writer.writerow(
                    {
                        "condition_id": r.get("condition_id", ""),
                        "predicted_score": float(r["predicted_score"]),
                        "log_rate": "",
                        "predicted_log_rate": "",
                    }
                )
            summary = {
                "n_rows": len(rows),
                "n_with_rates": 0,
                "linear_model": None,
                "pearson": None,
                "spearman": None,
                "rmse": None,
            }

    (output_dir / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
