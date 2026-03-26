from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(slots=True)
class Series:
    label: str
    times_ps: list[float]
    values: list[float]


_COND_RE = re.compile(
    r"ed_t(?P<Tk>\d+)k_p(?P<Ptok>[\dp]+)mpa_lith(?P<lithtok>[\dp]+)pct",
    flags=re.IGNORECASE,
)


def _parse_cond(condition_id: str) -> tuple[str, str]:
    match = _COND_RE.fullmatch(condition_id)
    if match is None:
        return ("unknown", condition_id)
    lith = match.group("lithtok").replace("p", ".")
    temp_k = match.group("Tk")
    pressure = match.group("Ptok").replace("p", ".")
    lith_key = f"lith_{lith}pct"
    tp_label = f"T={temp_k} K, P={pressure} MPa"
    return lith_key, tp_label


def _parse_cond_full(condition_id: str) -> tuple[str, str, str]:
    match = _COND_RE.fullmatch(condition_id)
    if match is None:
        return ("lith_unknown", "unknown", condition_id)
    lith = match.group("lithtok").replace("p", ".")
    temp_k = match.group("Tk")
    pressure = match.group("Ptok").replace("p", ".")
    lith_key = f"lith_{lith}pct"
    tp_key = f"T{temp_k}K_P{pressure}MPa"
    tp_label = f"T={temp_k} K, P={pressure} MPa"
    return lith_key, tp_key, tp_label


def _read_detailed(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_fs = float(row.get("time_fs", "nan"))
            if time_fs != time_fs:
                continue
            out["time_ps"].append(time_fs / 1000.0)
            out["potential_energy_eV"].append(float(row.get("potential_energy_eV", "nan")))
            out["kinetic_energy_eV"].append(float(row.get("kinetic_energy_eV", "nan")))
            out["total_energy_eV"].append(float(row.get("total_energy_eV", "nan")))
            out["temperature_K"].append(float(row.get("temperature_K", "nan")))
            out["volume_A3"].append(float(row.get("volume_A3", "nan")))
            out["density_g_cm3"].append(float(row.get("density_g_cm3", "nan")))
            pressure_atm = float(row.get("pressure_atm", "nan"))
            out["pressure_MPa"].append(pressure_atm * 0.101325 if pressure_atm == pressure_atm else float("nan"))
    return out


def _read_basic(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_ps = float(row.get("time_ps", "nan"))
            if time_ps != time_ps:
                continue
            pe = float(row.get("potential_energy_eV", "nan"))
            ke = float(row.get("kinetic_energy_eV", "nan"))
            out["time_ps"].append(time_ps)
            out["potential_energy_eV"].append(pe)
            out["kinetic_energy_eV"].append(ke)
            out["total_energy_eV"].append(pe + ke)
            out["temperature_K"].append(float(row.get("temperature_K", "nan")))
            out["volume_A3"].append(float(row.get("volume_A3", "nan")))
            out["density_g_cm3"].append(float(row.get("density_g_cm3", "nan")))
            out["pressure_MPa"].append(float(row.get("pressure_MPa", "nan")))
    return out


def _plot_dashboard(
    title: str,
    out_path: Path,
    metric_order: list[str],
    ylabel_by_metric: dict[str, str],
    metric_series: dict[str, list[Series]],
) -> bool:
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), constrained_layout=True)
    flat_axes = axes.flatten()

    anything_plotted = False
    for idx, metric in enumerate(metric_order):
        ax = flat_axes[idx]
        series_list = sorted(metric_series.get(metric, []), key=lambda s: s.label)
        plotted = 0
        for series in series_list:
            if not series.values:
                continue
            ax.plot(series.times_ps, series.values, label=series.label, linewidth=1.25)
            plotted += 1
        if plotted > 0:
            anything_plotted = True
            ax.legend(fontsize=7, loc="best")
        ax.set_title(metric)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(ylabel_by_metric[metric])
        ax.grid(alpha=0.25)

    for idx in range(len(metric_order), len(flat_axes)):
        flat_axes[idx].axis("off")

    if not anything_plotted:
        plt.close(fig)
        return False

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot in-progress MD thermo traces by lithiation percentage.")
    parser.add_argument("--run-root", type=Path, required=True, help="Path to .../uma_torchsim_screen")
    parser.add_argument("--phase", choices=["electrode"], default="electrode")
    parser.add_argument("--replica", type=str, default="replica_000")
    parser.add_argument("--run-name", choices=["retherm", "prod"], default="retherm")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-clean", action="store_true", help="Do not delete previously generated plots in out-dir")
    args = parser.parse_args()

    phase_root = args.run_root / args.phase
    out_dir = args.out_dir or (args.run_root / "live_plots" / args.phase / args.run_name)
    if out_dir.exists() and not args.no_clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_order = [
        "total_energy_eV",
        "kinetic_energy_eV",
        "potential_energy_eV",
        "temperature_K",
        "volume_A3",
        "density_g_cm3",
        "pressure_MPa",
    ]

    by_lith: dict[str, dict[str, list[Series]]] = {
        metric: defaultdict(list) for metric in metric_order
    }
    by_tp: dict[str, dict[str, list[Series]]] = {
        metric: defaultdict(list) for metric in metric_order
    }

    cond_dirs = sorted([path for path in phase_root.iterdir() if path.is_dir()]) if phase_root.exists() else []
    for cond_dir in cond_dirs:
        rep_dir = cond_dir / args.replica
        detailed_csv = rep_dir / f"{args.run_name}_thermo_detailed.csv"
        basic_csv = rep_dir / f"{args.run_name}_thermo.csv"
        if detailed_csv.exists():
            data = _read_detailed(detailed_csv)
        elif basic_csv.exists():
            data = _read_basic(basic_csv)
        else:
            continue

        times = data.get("time_ps", [])
        if not times:
            continue
        lith_key, tp_key, tp_label = _parse_cond_full(cond_dir.name)

        for metric in metric_order:
            vals = data.get(metric, [])
            if len(vals) != len(times):
                continue
            by_lith[metric][lith_key].append(Series(label=tp_label, times_ps=times, values=vals))
            by_tp[metric][tp_key].append(Series(label=lith_key, times_ps=times, values=vals))

    labels = {
        "total_energy_eV": "Total energy (eV)",
        "kinetic_energy_eV": "Kinetic energy (eV)",
        "potential_energy_eV": "Potential energy (eV)",
        "temperature_K": "Temperature (K)",
        "volume_A3": "Volume (A^3)",
        "density_g_cm3": "Density (g/cm^3)",
        "pressure_MPa": "Pressure (MPa)",
    }

    by_lith_dir = out_dir / "by_lithiation"
    by_tp_dir = out_dir / "by_tp"

    lith_fig_count = 0
    all_lith_keys = sorted({k for metric in metric_order for k in by_lith[metric].keys()})
    for lith_key in all_lith_keys:
        metric_data = {metric: by_lith[metric].get(lith_key, []) for metric in metric_order}
        out_path = by_lith_dir / f"{lith_key}_dashboard.png"
        ok = _plot_dashboard(
            title=f"{lith_key}: thermo vs time (series by T/P)",
            out_path=out_path,
            metric_order=metric_order,
            ylabel_by_metric=labels,
            metric_series=metric_data,
        )
        if ok:
            lith_fig_count += 1

    tp_fig_count = 0
    all_tp_keys = sorted({k for metric in metric_order for k in by_tp[metric].keys()})
    for tp_key in all_tp_keys:
        metric_data = {metric: by_tp[metric].get(tp_key, []) for metric in metric_order}
        out_path = by_tp_dir / f"{tp_key}_dashboard.png"
        ok = _plot_dashboard(
            title=f"{tp_key}: thermo vs time (series by lithiation %)",
            out_path=out_path,
            metric_order=metric_order,
            ylabel_by_metric=labels,
            metric_series=metric_data,
        )
        if ok:
            tp_fig_count += 1

    summary_path = out_dir / "plot_summary.txt"
    counts = {
        metric: {
            "by_lith_series": sum(len(v) for v in by_lith[metric].values()),
            "by_tp_series": sum(len(v) for v in by_tp[metric].values()),
        }
        for metric in metric_order
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"lithiation_dashboards: {lith_fig_count}",
        f"tp_dashboards: {tp_fig_count}",
    ]
    for metric, metric_counts in counts.items():
        lines.append(
            f"{metric}: by_lith_series={metric_counts['by_lith_series']}, "
            f"by_tp_series={metric_counts['by_tp_series']}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
