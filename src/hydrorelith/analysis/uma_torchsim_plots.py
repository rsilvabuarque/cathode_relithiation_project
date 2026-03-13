from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_msd_and_fit(times_ps, msd, fit: dict, out_path: Path, title: str = "MSD") -> None:
    t = np.asarray(times_ps, dtype=float)
    y = np.asarray(msd, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, y, label="MSD")
    y_fit = fit.get("slope_A2_per_ps", 0.0) * t + fit.get("intercept_A2", 0.0)
    ax.plot(t, y_fit, "--", label=f"fit D={fit.get('D_A2_per_ps', 0.0):.3g} A^2/ps")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD (A^2)")
    ax.set_title(title)
    ax.legend()
    _save(fig, out_path)


def plot_mean_std_band(x, mean_y, std_y, out_path: Path, title: str, xlabel: str, ylabel: str, label: str) -> None:
    xv = np.asarray(x, dtype=float)
    ym = np.asarray(mean_y, dtype=float)
    ys = np.asarray(std_y, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xv, ym, label=label)
    ax.fill_between(xv, ym - ys, ym + ys, alpha=0.25, linewidth=0.0, label=f"{label} +/- 1 std")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, out_path)


def plot_rdf(r_A, g_r, out_path: Path, title: str, cutoff_A: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_A, g_r)
    if cutoff_A is not None:
        ax.axvline(cutoff_A, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_title(title)
    _save(fig, out_path)


def plot_rdf_with_band(r_A, g_r_mean, g_r_std, out_path: Path, title: str, cutoff_A: float | None = None) -> None:
    r = np.asarray(r_A, dtype=float)
    gmean = np.asarray(g_r_mean, dtype=float)
    gstd = np.asarray(g_r_std, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, gmean, label="mean g(r)")
    ax.fill_between(r, gmean - gstd, gmean + gstd, alpha=0.25, linewidth=0.0, label="+/- 1 std")
    if cutoff_A is not None:
        ax.axvline(cutoff_A, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_title(title)
    ax.legend()
    _save(fig, out_path)


def plot_coordination(times_ps, cn_water, cn_hydroxide, out_ts: Path, out_hist: Path) -> None:
    t = np.asarray(times_ps)
    c_w = np.asarray(cn_water)
    c_oh = np.asarray(cn_hydroxide)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, c_w, label="water")
    ax.plot(t, c_oh, label="hydroxide")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("CN")
    ax.set_title("Li-O coordination")
    ax.legend()
    _save(fig, out_ts)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(c_w, bins=20, alpha=0.6, label="water")
    ax.hist(c_oh, bins=20, alpha=0.6, label="hydroxide")
    ax.set_xlabel("CN")
    ax.set_ylabel("Count")
    ax.set_title("CN histogram")
    ax.legend()
    _save(fig, out_hist)


def plot_coordination_with_band(
    times_ps,
    cn_water_mean,
    cn_water_std,
    cn_hydroxide_mean,
    cn_hydroxide_std,
    out_ts: Path,
) -> None:
    t = np.asarray(times_ps, dtype=float)
    cwm = np.asarray(cn_water_mean, dtype=float)
    cws = np.asarray(cn_water_std, dtype=float)
    cohm = np.asarray(cn_hydroxide_mean, dtype=float)
    cohs = np.asarray(cn_hydroxide_std, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, cwm, label="water mean")
    ax.fill_between(t, cwm - cws, cwm + cws, alpha=0.2, linewidth=0.0)
    ax.plot(t, cohm, label="hydroxide mean")
    ax.fill_between(t, cohm - cohs, cohm + cohs, alpha=0.2, linewidth=0.0)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("CN")
    ax.set_title("Li-O coordination (replica mean +/- std)")
    ax.legend()
    _save(fig, out_ts)


def plot_residence_proxy(lag_ps, residence_proxy, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lag_ps, residence_proxy)
    ax.set_xlabel("Lag (ps)")
    ax.set_ylabel("Residence proxy (fraction, unitless)")
    ax.set_title("Residence-proxy curve")
    _save(fig, out_path)


def plot_vacancy_metrics(times_ps, vacancy_series, out_ts: Path, out_hist: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times_ps, vacancy_series)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Vacancy accessibility")
    ax.set_title("Vacancy accessibility vs time")
    _save(fig, out_ts)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vacancy_series, bins=20)
    ax.set_xlabel("Vacancy accessibility")
    ax.set_ylabel("Count")
    ax.set_title("Vacancy summary histogram")
    _save(fig, out_hist)


def plot_density_equilibration_with_band(
    times_ps,
    density_mean,
    density_std,
    equilibration_time_ps,
    avg_density_g_cm3,
    out_path: Path,
) -> None:
    t = np.asarray(times_ps, dtype=float)
    dmean = np.asarray(density_mean, dtype=float)
    dstd = np.asarray(density_std, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, dmean, label="density mean")
    ax.fill_between(t, dmean - dstd, dmean + dstd, alpha=0.25, linewidth=0.0, label="+/- 1 std")
    ax.axvline(float(equilibration_time_ps), color="tab:orange", linestyle="--", linewidth=1.2, label="equilibration cutoff")
    ax.axhline(float(avg_density_g_cm3), color="tab:green", linestyle=":", linewidth=1.4, label="production density average")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Density (g/cm^3)")
    ax.set_title("NPT rethermalization density evolution")
    ax.legend()
    _save(fig, out_path)


def plot_pred_vs_exp(y_pred, y_true, out_path: Path) -> None:
    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp)
    lo = min(float(np.min(yt)), float(np.min(yp))) if yt.size and yp.size else -1
    hi = max(float(np.max(yt)), float(np.max(yp))) if yt.size and yp.size else 1
    ax.plot([lo, hi], [lo, hi], "--", color="gray")
    ax.set_xlabel("Experimental log(rate)")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs experimental")
    _save(fig, out_path)


def plot_heatmap_tp_grid(rows: list[dict], out_path: Path, best_csv_path: Path) -> None:
    if not rows:
        return

    temps = sorted({float(r.get("temperature_C", 0.0)) for r in rows})
    press = sorted({float(r.get("pressure_MPa", 0.0)) for r in rows})
    lith_vals = sorted({str(r.get("lithiation_bin", "all")) for r in rows})
    conc_vals = sorted({str(r.get("liOH_bin", "all")) for r in rows})

    nrows = max(1, len(lith_vals))
    ncols = max(1, len(conc_vals))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    best = []
    for i, lith in enumerate(lith_vals):
        for j, conc in enumerate(conc_vals):
            ax = axes[i, j]
            grid = np.full((len(temps), len(press)), np.nan)
            for row in rows:
                if str(row.get("lithiation_bin", "all")) != lith:
                    continue
                if str(row.get("liOH_bin", "all")) != conc:
                    continue
                ti = temps.index(float(row.get("temperature_C", 0.0)))
                pj = press.index(float(row.get("pressure_MPa", 0.0)))
                grid[ti, pj] = float(row.get("predicted_score", np.nan))
            im = ax.imshow(grid, aspect="auto", origin="lower")
            ax.set_xticks(range(len(press)), [f"{p:g}" for p in press])
            ax.set_yticks(range(len(temps)), [f"{t:g}" for t in temps])
            ax.set_xlabel("Pressure (MPa)")
            ax.set_ylabel("Temperature (C)")
            ax.set_title(f"lith={lith}, LiOH={conc}")
            fig.colorbar(im, ax=ax, fraction=0.046)

            if np.isfinite(grid).any():
                idx = np.nanargmax(grid)
                ti, pj = np.unravel_index(idx, grid.shape)
                best.append({
                    "lithiation_bin": lith,
                    "liOH_bin": conc,
                    "temperature_C": temps[ti],
                    "pressure_MPa": press[pj],
                    "predicted_score": float(grid[ti, pj]),
                })

    _save(fig, out_path)

    import csv

    best_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with best_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["lithiation_bin", "liOH_bin", "temperature_C", "pressure_MPa", "predicted_score"],
        )
        writer.writeheader()
        writer.writerows(best)
