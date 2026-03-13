from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from tqdm.auto import tqdm

from hydrorelith.analysis.uma_torchsim_descriptors import (
    O_HYDROXIDE,
    O_HYDRONIUM,
    O_OTHER,
    O_WATER,
    classify_oxygen_species_frame,
    compute_coordination_from_cutoff,
    compute_msd,
    compute_residence_proxy,
    compute_rdf,
    compute_vacancy_metrics_electrode,
    fit_diffusion_from_msd,
    unwrap_positions,
)
from hydrorelith.analysis.uma_torchsim_plots import (
    plot_coordination,
    plot_coordination_with_band,
    plot_density_equilibration_with_band,
    plot_heatmap_tp_grid,
    plot_msd_and_fit,
    plot_mean_std_band,
    plot_pred_vs_exp,
    plot_rdf,
    plot_rdf_with_band,
    plot_residence_proxy,
    plot_vacancy_metrics,
)
from hydrorelith.analysis.uma_torchsim_regression import (
    compute_rate_score,
    fit_simple_models,
    load_experimental_rates,
)
from hydrorelith.io.torchsim_export2pt import export_h5md_to_lammps_dump, write_2pt_metadata
from hydrorelith.pipelines.uma_torchsim_screen_config import ScreenConfig, parse_args_to_config
from hydrorelith.pipelines.uma_torchsim_screen_run import run_one_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw-uma-torchsim-screen",
        description="GPU-first UMA TorchSim screening workflow.",
    )
    parser.add_argument("--electrode-manifest", type=Path, default=None)
    parser.add_argument("--electrolyte-manifest", type=Path, default=None)
    parser.add_argument("--electrode-root", type=Path, default=None)
    parser.add_argument("--electrolyte-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["electrode", "electrolyte", "both"], default="both")
    parser.add_argument("--stage", default="all", help="Comma-separated: md,analyze,export-2pt,regress,plots,all")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--md-only", action="store_true")
    parser.add_argument("--model-name", type=str, default="uma-s-1p1")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--ensemble", choices=["nvt", "npt"], default="nvt")
    parser.add_argument("--timestep-ps", type=float, default=0.001)
    parser.add_argument("--dump-every-steps", type=int, default=2)
    parser.add_argument("--retherm-steps-electrode", type=int, default=2000)
    parser.add_argument("--retherm-steps-electrolyte", type=int, default=8000)
    parser.add_argument("--prod-steps", type=int, default=25000)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--compute-stress", action="store_true")
    parser.add_argument("--export-2pt", action="store_true")
    parser.add_argument("--experimental-rates-csv", type=Path, default=None)
    parser.add_argument("--electrode-reference-pristine", type=Path, default=None)
    parser.add_argument("--use-default-tp-grid", action="store_true")
    parser.add_argument("--tp-grid-csv", type=Path, default=None)
    parser.add_argument("--o-h-covalent-cutoff-A", type=float, default=1.25)
    parser.add_argument("--li-o-cutoff-A", type=float, default=2.6)
    parser.add_argument("--vacancy-site-match-cutoff-A", type=float, default=1.0)
    parser.add_argument("--analysis-frame-stride", type=int, default=None)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--max-memory-scaler", type=float, default=None)
    parser.add_argument("--skip-batch-benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=40)
    parser.add_argument("--benchmark-warmup-steps", type=int, default=5)
    parser.add_argument("--benchmark-max-systems", type=int, default=None)
    parser.add_argument("--benchmark-step-size", type=int, default=1)
    parser.add_argument("--benchmark-temperature-k", type=float, default=298.0)
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--debug", action="store_true")
    return parser


def _stage_enabled(config: ScreenConfig, stage: str) -> bool:
    return "all" in config.stages or stage in config.stages


def _phase_list(config: ScreenConfig) -> list[str]:
    return ["electrode", "electrolyte"] if config.phase == "both" else [config.phase]


def _read_h5md(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as h5:
        return {k: np.array(v) for k, v in h5.items()}


def _simple_hist_plot(x, y, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_manifest_row_from_metadata(rep_dir: Path) -> dict:
    meta_path = rep_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr)))


def _compute_species_counts(o_type_series: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "water": np.sum(o_type_series == O_WATER, axis=1),
        "hydroxide": np.sum(o_type_series == O_HYDROXIDE, axis=1),
        "hydronium": np.sum(o_type_series == O_HYDRONIUM, axis=1),
        "other": np.sum(o_type_series == O_OTHER, axis=1),
    }


def _plot_species_counts(times_ps: np.ndarray, counts: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, vals in counts.items():
        ax.plot(times_ps, vals, label=key)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("O counts")
    ax.set_title("O species counts vs time")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _analyze_phase(config: ScreenConfig, phase: str) -> list[dict]:
    phase_root = config.output_dir / "uma_torchsim_screen" / phase
    plot_root = config.output_dir / "uma_torchsim_screen" / "plots"
    rows: list[dict] = []
    cond_dirs = sorted([p for p in phase_root.iterdir() if p.is_dir()] if phase_root.exists() else [])
    tqdm.write(f"[analyze:{phase}] discovered {len(cond_dirs)} condition directories under {phase_root}")

    pristine = None
    if phase == "electrode" and config.electrode_reference_pristine is not None:
        pristine = ase_read(config.electrode_reference_pristine)

    cond_bar = tqdm(cond_dirs, desc=f"analyze {phase}", unit="condition")
    for cond_dir in cond_bar:
        t_cond = time.perf_counter()
        rep_dirs = sorted([p for p in cond_dir.iterdir() if p.is_dir() and p.name.startswith("replica_")])
        cond_bar.set_postfix_str(f"replicas={len(rep_dirs)}")
        d_vals, cn_oh_vals, res_oh_vals, vac_vals = [], [], [], []
        li_msd_1ps_vals = []
        cond_meta: dict | None = None
        msd_curves: list[np.ndarray] = []
        rdf_curves: list[np.ndarray] = []
        cn_water_curves: list[np.ndarray] = []
        cn_oh_curves: list[np.ndarray] = []
        density_curves: list[np.ndarray] = []
        density_eq_start_times: list[float] = []
        density_eq_means: list[float] = []
        common_times: np.ndarray | None = None
        rdf_r_axis: np.ndarray | None = None

        rep_bar = tqdm(rep_dirs, desc=f"{cond_dir.name}", unit="replica", leave=False)
        for rep_dir in rep_bar:
            prod_h5 = rep_dir / "prod.h5md"
            if not prod_h5.exists():
                rep_bar.set_postfix_str("missing prod.h5md")
                continue
            t_rep = time.perf_counter()
            data = _read_h5md(prod_h5)
            if cond_meta is None:
                cond_meta = _parse_manifest_row_from_metadata(rep_dir)

            positions = data["positions"]
            cells = data["cell"]
            z = data["atomic_numbers"]
            pbc = data.get("pbc", np.array([1, 1, 1], dtype=bool))
            times = data["time_ps"]
            rep_bar.set_postfix_str(f"frames={len(times)}")

            stride = config.analysis_frame_stride
            if stride is None:
                stride = 5 if phase == "electrolyte" else 10
            positions = positions[::stride]
            cells = cells[::stride]
            times = times[::stride]

            uw = unwrap_positions(positions, cells, pbc)
            li_idx = np.where(z == 3)[0]
            li_msd = compute_msd(uw, li_idx)
            msd_curves.append(li_msd)
            fit = fit_diffusion_from_msd(
                li_msd,
                times,
                fit_start_ps=float(times[min(1, len(times) - 1)]),
                fit_end_ps=float(times[-1]) if len(times) else 0.0,
            )

            desc = {
                "condition_id": cond_dir.name,
                "phase": phase,
                "D_li_A2_per_ps": float(fit["D_A2_per_ps"]),
                "temperature_C": cond_meta.get("temperature_C") if cond_meta else None,
                "pressure_MPa": cond_meta.get("pressure_MPa") if cond_meta else None,
            }

            if phase == "electrolyte":
                o_type_series = np.array(
                    [
                        classify_oxygen_species_frame(
                            positions[i],
                            z,
                            cells[i],
                            pbc,
                            config.o_h_covalent_cutoff_A,
                        )
                        for i in range(positions.shape[0])
                    ]
                )
                species_counts = _compute_species_counts(o_type_series)
                _plot_species_counts(times, species_counts, plot_root / f"{phase}_{cond_dir.name}_o_species_counts_vs_time.png")

                rdf_total = compute_rdf(
                    positions,
                    z,
                    cells,
                    pbc,
                    pairs={"li_o_total": ("Li", "O")},
                    r_max_A=6.0,
                    dr_A=0.05,
                )["li_o_total"]
                plot_rdf(
                    rdf_total["r_A"],
                    rdf_total["g_r"],
                    plot_root / f"{phase}_{cond_dir.name}_rdf_li_o_total.png",
                    "RDF Li-O total",
                    cutoff_A=config.li_o_cutoff_A,
                )
                plot_rdf(
                    rdf_total["r_A"],
                    rdf_total["g_r"],
                    plot_root / f"{phase}_{cond_dir.name}_rdf_li_o_water.png",
                    "RDF Li-O water proxy",
                    cutoff_A=config.li_o_cutoff_A,
                )
                plot_rdf(
                    rdf_total["r_A"],
                    rdf_total["g_r"],
                    plot_root / f"{phase}_{cond_dir.name}_rdf_li_o_hydroxide.png",
                    "RDF Li-O hydroxide proxy",
                    cutoff_A=config.li_o_cutoff_A,
                )

                cn = compute_coordination_from_cutoff(
                    positions,
                    z,
                    cells,
                    pbc,
                    config.li_o_cutoff_A,
                    o_type_series,
                )
                plot_coordination(
                    times,
                    cn["cn_water_series"],
                    cn["cn_hydroxide_series"],
                    plot_root / f"{phase}_{cond_dir.name}_cn_time_series.png",
                    plot_root / f"{phase}_{cond_dir.name}_cn_hist.png",
                )
                cn_water_curves.append(np.asarray(cn["cn_water_series"], dtype=float))
                cn_oh_curves.append(np.asarray(cn["cn_hydroxide_series"], dtype=float))

                residence = compute_residence_proxy(
                    positions,
                    z,
                    cells,
                    pbc,
                    config.li_o_cutoff_A,
                    o_type_series,
                    lag_ps=2.0,
                    times_ps=times,
                )
                plot_residence_proxy(
                    residence["lag_ps"],
                    residence["residence_proxy"],
                    plot_root / f"{phase}_{cond_dir.name}_residence_proxy_curve.png",
                )

                desc["cn_hydroxide_mean"] = float(np.mean(cn["cn_hydroxide_series"]))
                desc["residence_proxy_oh"] = float(residence["residence_proxy_oh"])
                desc["liOH_M"] = cond_meta.get("liOH_M") if cond_meta else None
                desc["kOH_M"] = cond_meta.get("kOH_M") if cond_meta else None
                if rdf_total["r_A"]:
                    r_arr = np.asarray(rdf_total["r_A"], dtype=float)
                    g_arr = np.asarray(rdf_total["g_r"], dtype=float)
                    rdf_r_axis = r_arr
                    rdf_curves.append(g_arr)
                    shell_mask = (r_arr >= 1.5) & (r_arr <= 4.0)
                    if np.any(shell_mask):
                        local = np.where(shell_mask)[0]
                        i_peak = int(local[np.argmax(g_arr[shell_mask])])
                    else:
                        i_peak = int(np.argmax(g_arr))
                    desc["rdf_li_o_peak_r_A"] = float(r_arr[i_peak])
                    desc["rdf_li_o_peak_g"] = float(g_arr[i_peak])
                cn_oh_vals.append(desc["cn_hydroxide_mean"])
                res_oh_vals.append(desc["residence_proxy_oh"])
            else:
                if pristine is not None and positions.shape[0] > 0:
                    vac_series = []
                    for i in range(positions.shape[0]):
                        metrics = compute_vacancy_metrics_electrode(
                            {
                                "positions": positions[i],
                                "Z": z,
                                "cell": cells[i],
                            },
                            pristine,
                            config.vacancy_site_match_cutoff_A,
                        )
                        vac_series.append(metrics["vacancy_accessibility"])
                    vac_series_arr = np.asarray(vac_series, dtype=float)
                    plot_vacancy_metrics(
                        times,
                        vac_series_arr,
                        plot_root / f"{phase}_{cond_dir.name}_vacancy_accessibility_vs_time.png",
                        plot_root / f"{phase}_{cond_dir.name}_vacancy_summary_hist.png",
                    )
                    desc["vacancy_accessibility_mean"] = float(np.mean(vac_series_arr))
                    vac_vals.append(desc["vacancy_accessibility_mean"])
                else:
                    desc["vacancy_accessibility_mean"] = 0.0

                if len(times) > 0:
                    i_1ps = int(np.argmin(np.abs(times - 1.0)))
                    desc["electrode_li_msd_1ps"] = float(li_msd[i_1ps])
                    li_msd_1ps_vals.append(desc["electrode_li_msd_1ps"])
                else:
                    desc["electrode_li_msd_1ps"] = 0.0

            d_vals.append(desc["D_li_A2_per_ps"])
            plot_msd_and_fit(
                times,
                li_msd,
                fit,
                plot_root / f"{phase}_{cond_dir.name}_{rep_dir.name}_msd_li.png",
                title=f"{phase} {cond_dir.name} {rep_dir.name}",
            )
            (rep_dir / "descriptors.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")

            if common_times is None:
                common_times = np.asarray(times, dtype=float)

            retherm_thermo = rep_dir / "retherm_thermo.csv"
            retherm_eq = rep_dir / "retherm_equilibration.json"
            if retherm_thermo.exists():
                with retherm_thermo.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    d_series = [float(r.get("density_g_cm3", 0.0)) for r in reader]
                if d_series:
                    density_curves.append(np.asarray(d_series, dtype=float))
            if retherm_eq.exists():
                eq_obj = json.loads(retherm_eq.read_text(encoding="utf-8"))
                density_eq_start_times.append(float(eq_obj.get("equilibration_start_time_ps", 0.0)))
                density_eq_means.append(float(eq_obj.get("equilibrated_density_mean_g_cm3", 0.0)))

            rep_bar.set_postfix_str(f"done {time.perf_counter() - t_rep:.1f}s")
        rep_bar.close()

        if cond_meta is None:
            tqdm.write(f"[analyze:{phase}] skipping {cond_dir.name}: no readable prod.h5md files")
            continue
        d_mean, d_std = _mean_std(d_vals)
        cond_row = {
            "condition_id": cond_dir.name,
            "phase": phase,
            "temperature_C": float(cond_meta.get("temperature_C", 0.0)),
            "pressure_MPa": float(cond_meta.get("pressure_MPa", 0.0)),
            "D_li_A2_per_ps": d_mean,
            "D_li_std": d_std,
        }
        if phase == "electrolyte":
            cond_row["cn_hydroxide_mean"] = _mean_std(cn_oh_vals)[0]
            cond_row["cn_hydroxide_std"] = _mean_std(cn_oh_vals)[1]
            cond_row["residence_proxy_oh"] = _mean_std(res_oh_vals)[0]
            cond_row["residence_proxy_oh_std"] = _mean_std(res_oh_vals)[1]
            cond_row["liOH_M"] = float(cond_meta.get("liOH_M", 0.0))
            cond_row["kOH_M"] = float(cond_meta.get("kOH_M", 0.0))

            if rdf_curves and rdf_r_axis is not None:
                rdf_mat = np.vstack(rdf_curves)
                rdf_mean = np.mean(rdf_mat, axis=0)
                rdf_std = np.std(rdf_mat, axis=0)
                plot_rdf_with_band(
                    rdf_r_axis,
                    rdf_mean,
                    rdf_std,
                    plot_root / f"{phase}_{cond_dir.name}_rdf_li_o_total_replicas_mean_std.png",
                    "RDF Li-O total (replica mean +/- std)",
                    cutoff_A=config.li_o_cutoff_A,
                )
                shell_mask = (rdf_r_axis >= 1.5) & (rdf_r_axis <= 4.0)
                if np.any(shell_mask):
                    local = np.where(shell_mask)[0]
                    i_peak = int(local[np.argmax(rdf_mean[shell_mask])])
                else:
                    i_peak = int(np.argmax(rdf_mean))
                cond_row["rdf_li_o_peak_r_A"] = float(rdf_r_axis[i_peak])
                cond_row["rdf_li_o_peak_g"] = float(rdf_mean[i_peak])
                cond_row["rdf_li_o_peak_g_std"] = float(rdf_std[i_peak])

            if cn_water_curves and cn_oh_curves and common_times is not None:
                cwm = np.mean(np.vstack(cn_water_curves), axis=0)
                cws = np.std(np.vstack(cn_water_curves), axis=0)
                cohm = np.mean(np.vstack(cn_oh_curves), axis=0)
                cohs = np.std(np.vstack(cn_oh_curves), axis=0)
                plot_coordination_with_band(
                    common_times,
                    cwm,
                    cws,
                    cohm,
                    cohs,
                    plot_root / f"{phase}_{cond_dir.name}_cn_time_series_replicas_mean_std.png",
                )

            if density_curves:
                dmat = np.vstack(density_curves)
                dmean = np.mean(dmat, axis=0)
                dstd = np.std(dmat, axis=0)
                eq_time = float(np.mean(density_eq_start_times)) if density_eq_start_times else 0.0
                eq_den = float(np.mean(density_eq_means)) if density_eq_means else float(np.mean(dmean))
                dtime = np.arange(dmean.shape[0], dtype=float) * (config.dump_every_steps * config.timestep_ps)
                plot_density_equilibration_with_band(
                    dtime,
                    dmean,
                    dstd,
                    equilibration_time_ps=eq_time,
                    avg_density_g_cm3=eq_den,
                    out_path=plot_root / f"{phase}_{cond_dir.name}_density_retherm_equilibration.png",
                )
                cond_row["equilibrated_density_mean_g_cm3"] = eq_den
                cond_row["equilibrated_density_std_g_cm3"] = (
                    0.0 if not density_eq_means else float(np.std(np.asarray(density_eq_means, dtype=float)))
                )
        else:
            cond_row["vacancy_accessibility_mean"] = _mean_std(vac_vals)[0]
            cond_row["vacancy_accessibility_std"] = _mean_std(vac_vals)[1]
            cond_row["electrode_li_msd_1ps"] = _mean_std(li_msd_1ps_vals)[0]
            cond_row["electrode_li_msd_1ps_std"] = _mean_std(li_msd_1ps_vals)[1]
            if "lithiation_fraction" in cond_meta:
                cond_row["lithiation_fraction"] = float(cond_meta["lithiation_fraction"])

        if msd_curves and common_times is not None:
            mm = np.mean(np.vstack(msd_curves), axis=0)
            ms = np.std(np.vstack(msd_curves), axis=0)
            plot_mean_std_band(
                common_times,
                mm,
                ms,
                plot_root / f"{phase}_{cond_dir.name}_msd_li_replicas_mean_std.png",
                title=f"{phase} {cond_dir.name} MSD Li (replica mean +/- std)",
                xlabel="Time (ps)",
                ylabel="MSD (A^2)",
                label="MSD",
            )
        rows.append(cond_row)
        tqdm.write(
            f"[analyze:{phase}] condition {cond_dir.name} complete in "
            f"{time.perf_counter() - t_cond:.1f}s"
        )

    cond_bar.close()
    tqdm.write(f"[analyze:{phase}] completed {len(rows)} summarized condition rows")

    return rows


def _write_rows_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _cartesian_merge_rows(electrode_rows: list[dict], electrolyte_rows: list[dict]) -> list[dict]:
    merged = []
    for er in electrode_rows:
        for lr in electrolyte_rows:
            if abs(float(er.get("temperature_C", 0.0)) - float(lr.get("temperature_C", 0.0))) > 1e-8:
                continue
            if abs(float(er.get("pressure_MPa", 0.0)) - float(lr.get("pressure_MPa", 0.0))) > 1e-8:
                continue
            row = {
                **{f"electrode_{k}": v for k, v in er.items()},
                **{f"electrolyte_{k}": v for k, v in lr.items()},
                "condition_id": f"{er.get('condition_id')}__{lr.get('condition_id')}",
                "temperature_C": er.get("temperature_C"),
                "pressure_MPa": er.get("pressure_MPa"),
                "D_li_A2_per_ps": lr.get("D_li_A2_per_ps", 0.0),
                "residence_proxy_oh": lr.get("residence_proxy_oh", 0.0),
                "cn_hydroxide_mean": lr.get("cn_hydroxide_mean", 0.0),
                "liOH_M": lr.get("liOH_M", 0.0),
                "vacancy_accessibility_mean": er.get("vacancy_accessibility_mean", 0.0),
                "electrode_li_msd_1ps": er.get("electrode_li_msd_1ps", 0.0),
            }
            row["lithiation_bin"] = str(er.get("lithiation_fraction", "all"))
            row["liOH_bin"] = str(lr.get("liOH_M", "all"))
            merged.append(row)
    return merged


def _run_export_2pt(config: ScreenConfig, phase: str) -> None:
    phase_root = config.output_dir / "uma_torchsim_screen" / phase
    out_root = config.output_dir / "uma_torchsim_screen" / "export2pt" / phase
    for cond_dir in sorted([p for p in phase_root.iterdir() if p.is_dir()] if phase_root.exists() else []):
        for rep_dir in sorted([p for p in cond_dir.iterdir() if p.is_dir() and p.name.startswith("replica_")]):
            prod_h5 = rep_dir / "prod.h5md"
            thermo = rep_dir / "prod_thermo.csv"
            if not (prod_h5.exists() and thermo.exists()):
                continue
            out_dir = out_root / cond_dir.name / rep_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            export_h5md_to_lammps_dump(prod_h5, out_dir / "prod.lammpstrj", unwrap=True, include_ke_atom=True)
            write_2pt_metadata(
                thermo,
                out_dir / "2pt_metadata.json",
                timestep_ps=config.timestep_ps,
                dump_every_steps=config.dump_every_steps,
            )


def _run_regression_and_plots(config: ScreenConfig, electrode_rows: list[dict], electrolyte_rows: list[dict]) -> None:
    merged_dir = config.output_dir / "uma_torchsim_screen" / "merged"
    plot_root = config.output_dir / "uma_torchsim_screen" / "plots"

    merged_rows = _cartesian_merge_rows(electrode_rows, electrolyte_rows)
    tqdm.write(
        "[plots] regression input rows: "
        f"electrode={len(electrode_rows)}, electrolyte={len(electrolyte_rows)}, merged={len(merged_rows)}"
    )
    rates = load_experimental_rates(config.experimental_rates_csv) if config.experimental_rates_csv else None

    rates_by_key = {}
    if rates:
        rates_by_key = {(r.get("temperature_C"), r.get("pressure_MPa"), r.get("lithiation_fraction"), r.get("liOH_M")): r for r in rates}
    for row in merged_rows:
        row["predicted_score"] = compute_rate_score(row)
        key = (row.get("temperature_C"), row.get("pressure_MPa"), row.get("electrode_lithiation_fraction"), row.get("liOH_M"))
        if key in rates_by_key and "log_rate" in rates_by_key[key]:
            row["log_rate"] = float(rates_by_key[key]["log_rate"])

    _write_rows_csv(merged_rows, merged_dir / "features.csv")
    fit_simple_models(merged_rows, merged_dir)
    tqdm.write(f"[plots] wrote merged tables to {merged_dir}")

    fig, ax = plt.subplots(figsize=(6, 4))
    scores = [float(r["predicted_score"]) for r in merged_rows]
    if scores:
        ax.hist(scores, bins=20)
    ax.set_xlabel("Predicted rate score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution")
    (plot_root).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_root / "score_distribution.png", dpi=180)
    plt.close(fig)

    with (merged_dir / "pred_vs_true.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    y_true = [float(r["log_rate"]) for r in rows if r.get("log_rate") not in ("", None)]
    y_pred = [float(r["predicted_log_rate"]) for r in rows if r.get("log_rate") not in ("", None)]
    if y_true and y_pred:
        plot_pred_vs_exp(y_pred, y_true, plot_root / "pred_vs_exp.png")

    plot_heatmap_tp_grid(
        merged_rows,
        plot_root / "heatmap_tp_by_x_and_conc.png",
        plot_root / "best_conditions_table.csv",
    )
    tqdm.write(f"[plots] wrote plot outputs to {plot_root}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = parse_args_to_config(args)

    run_root = config.output_dir / "uma_torchsim_screen"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_config.json").write_text(
        json.dumps(
            {
                "phase": config.phase,
                "stages": list(config.stages),
                "model_name": config.model_name,
                "device": config.device,
                "ensemble": config.ensemble,
                "timestep_ps": config.timestep_ps,
                "dump_every_steps": config.dump_every_steps,
                "replicas": config.replicas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    phases = _phase_list(config)

    if _stage_enabled(config, "md"):
        tqdm.write("[stage] starting md")
        for phase in phases:
            tqdm.write(f"[stage] md phase={phase}")
            run_one_phase(phase, config)

    electrode_rows: list[dict] = []
    electrolyte_rows: list[dict] = []
    if _stage_enabled(config, "analyze"):
        tqdm.write("[stage] starting analyze")
        if "electrode" in phases:
            electrode_rows = _analyze_phase(config, "electrode")
            _write_rows_csv(electrode_rows, run_root / "electrode_descriptors.csv")
            tqdm.write(f"[stage] wrote electrode descriptors: {run_root / 'electrode_descriptors.csv'}")
        if "electrolyte" in phases:
            electrolyte_rows = _analyze_phase(config, "electrolyte")
            _write_rows_csv(electrolyte_rows, run_root / "electrolyte_descriptors.csv")
            tqdm.write(f"[stage] wrote electrolyte descriptors: {run_root / 'electrolyte_descriptors.csv'}")

    if _stage_enabled(config, "export-2pt"):
        tqdm.write("[stage] starting export-2pt")
        for phase in phases:
            _run_export_2pt(config, phase)

    if _stage_enabled(config, "regress"):
        tqdm.write("[stage] starting regress")
        if not electrode_rows and (run_root / "electrode_descriptors.csv").exists():
            with (run_root / "electrode_descriptors.csv").open("r", newline="", encoding="utf-8") as handle:
                electrode_rows = list(csv.DictReader(handle))
        if not electrolyte_rows and (run_root / "electrolyte_descriptors.csv").exists():
            with (run_root / "electrolyte_descriptors.csv").open("r", newline="", encoding="utf-8") as handle:
                electrolyte_rows = list(csv.DictReader(handle))
        _run_regression_and_plots(config, electrode_rows, electrolyte_rows)

    if _stage_enabled(config, "plots") and not _stage_enabled(config, "regress"):
        tqdm.write("[stage] starting plots")
        if not electrode_rows and (run_root / "electrode_descriptors.csv").exists():
            with (run_root / "electrode_descriptors.csv").open("r", newline="", encoding="utf-8") as handle:
                electrode_rows = list(csv.DictReader(handle))
        if not electrolyte_rows and (run_root / "electrolyte_descriptors.csv").exists():
            with (run_root / "electrolyte_descriptors.csv").open("r", newline="", encoding="utf-8") as handle:
                electrolyte_rows = list(csv.DictReader(handle))
        _run_regression_and_plots(config, electrode_rows, electrolyte_rows)


def main_analyze() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.analysis_only = True
    config = parse_args_to_config(args)
    phases = _phase_list(config)
    if "electrode" in phases:
        _analyze_phase(config, "electrode")
    if "electrolyte" in phases:
        _analyze_phase(config, "electrolyte")


def main_export2pt() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = parse_args_to_config(args)
    for phase in _phase_list(config):
        _run_export_2pt(config, phase)


if __name__ == "__main__":
    main()
