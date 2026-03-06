from __future__ import annotations

import numpy as np
from ase import Atoms


O_WATER = 0
O_HYDROXIDE = 1
O_HYDRONIUM = 2
O_OTHER = 3


def _to_fractional(vecs: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return vecs @ np.linalg.inv(cell.T)


def _min_image_delta(delta: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    frac = _to_fractional(delta, cell)
    for axis in range(3):
        if pbc[axis]:
            frac[:, axis] -= np.rint(frac[:, axis])
    return frac @ cell


def unwrap_positions(positions, cell, pbc) -> np.ndarray:
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")
    cell_arr = np.asarray(cell, dtype=float)
    if cell_arr.ndim == 2:
        cell_arr = np.repeat(cell_arr[None, :, :], pos.shape[0], axis=0)
    pbc_arr = np.asarray(pbc, dtype=bool)

    unwrapped = np.array(pos, copy=True)
    for frame in range(1, pos.shape[0]):
        delta = pos[frame] - pos[frame - 1]
        delta = _min_image_delta(delta, cell_arr[frame], pbc_arr)
        unwrapped[frame] = unwrapped[frame - 1] + delta
    return unwrapped


def compute_msd(unwrapped_positions, atom_indices) -> np.ndarray:
    arr = np.asarray(unwrapped_positions, dtype=float)
    idx = np.asarray(atom_indices, dtype=int)
    if idx.size == 0:
        return np.zeros(arr.shape[0], dtype=float)
    r0 = arr[0, idx, :]
    dr = arr[:, idx, :] - r0[None, :, :]
    return np.mean(np.sum(dr**2, axis=2), axis=1)


def fit_diffusion_from_msd(msd, times_ps, fit_start_ps, fit_end_ps) -> dict:
    msd_arr = np.asarray(msd, dtype=float)
    t_arr = np.asarray(times_ps, dtype=float)
    mask = (t_arr >= fit_start_ps) & (t_arr <= fit_end_ps)
    if np.count_nonzero(mask) < 2:
        return {"slope_A2_per_ps": 0.0, "intercept_A2": 0.0, "D_A2_per_ps": 0.0, "r2": 0.0}
    coef = np.polyfit(t_arr[mask], msd_arr[mask], deg=1)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = slope * t_arr[mask] + intercept
    ss_res = float(np.sum((msd_arr[mask] - pred) ** 2))
    ss_tot = float(np.sum((msd_arr[mask] - np.mean(msd_arr[mask])) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "slope_A2_per_ps": slope,
        "intercept_A2": intercept,
        "D_A2_per_ps": slope / 6.0,
        "r2": r2,
    }


def classify_oxygen_species_frame(pos_A, Z, cell_A, pbc, o_h_cutoff_A) -> np.ndarray:
    pos = np.asarray(pos_A, dtype=float)
    z = np.asarray(Z, dtype=int)
    cell = np.asarray(cell_A, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    labels = np.full(z.shape[0], -1, dtype=int)

    o_idx = np.where(z == 8)[0]
    h_idx = np.where(z == 1)[0]
    if o_idx.size == 0:
        return labels

    for oi in o_idx:
        if h_idx.size == 0:
            labels[oi] = O_OTHER
            continue
        delta = pos[h_idx] - pos[oi]
        delta = _min_image_delta(delta, cell, pbc_arr)
        d = np.linalg.norm(delta, axis=1)
        count = int(np.sum(d <= o_h_cutoff_A))
        if count == 2:
            labels[oi] = O_WATER
        elif count == 1:
            labels[oi] = O_HYDROXIDE
        elif count == 3:
            labels[oi] = O_HYDRONIUM
        else:
            labels[oi] = O_OTHER
    return labels


def compute_rdf(pos_A_series, Z, cell_A_series, pbc, pairs, r_max_A, dr_A) -> dict:
    positions = np.asarray(pos_A_series, dtype=float)
    cells = np.asarray(cell_A_series, dtype=float)
    z = np.asarray(Z, dtype=int)
    pbc_arr = np.asarray(pbc, dtype=bool)

    bins = np.arange(0.0, r_max_A + dr_A, dr_A)
    r = 0.5 * (bins[1:] + bins[:-1])
    out: dict[str, dict[str, list[float]]] = {}

    selector = {
        "Li": np.where(z == 3)[0],
        "O": np.where(z == 8)[0],
        "H": np.where(z == 1)[0],
    }

    for pair_name, (a_name, b_name) in pairs.items():
        a_idx = selector.get(a_name, np.array([], dtype=int))
        b_idx = selector.get(b_name, np.array([], dtype=int))
        hist = np.zeros(len(r), dtype=float)
        n_count = 0
        shell_vol = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        density_terms: list[float] = []
        for frame in range(positions.shape[0]):
            if a_idx.size == 0 or b_idx.size == 0:
                continue
            cell = cells[frame]
            vol = abs(np.linalg.det(cell))
            if vol <= 0:
                continue
            density_terms.append(float(b_idx.size) / float(vol))
            for ai in a_idx:
                delta = positions[frame, b_idx] - positions[frame, ai]
                delta = _min_image_delta(delta, cell, pbc_arr)
                dist = np.linalg.norm(delta, axis=1)
                h, _ = np.histogram(dist, bins=bins)
                hist += h
                n_count += 1
        if n_count > 0 and density_terms:
            rho_b = float(np.mean(density_terms))
            denom = float(n_count) * rho_b * shell_vol
            # Avoid division-by-zero near r=0 for degenerate bins.
            denom = np.where(denom > 1e-16, denom, 1.0)
            hist = hist / denom
        out[pair_name] = {"r_A": r.tolist(), "g_r": hist.tolist()}
    return out


def compute_coordination_from_cutoff(pos_A_series, Z, cell_A_series, pbc, li_o_cutoff_A, o_type_series) -> dict:
    positions = np.asarray(pos_A_series, dtype=float)
    cells = np.asarray(cell_A_series, dtype=float)
    z = np.asarray(Z, dtype=int)
    pbc_arr = np.asarray(pbc, dtype=bool)
    o_types = np.asarray(o_type_series, dtype=int)

    li_idx = np.where(z == 3)[0]
    o_idx = np.where(z == 8)[0]
    cn_w, cn_oh = [], []
    for frame in range(positions.shape[0]):
        cw = 0
        coh = 0
        for li in li_idx:
            delta = positions[frame, o_idx] - positions[frame, li]
            delta = _min_image_delta(delta, cells[frame], pbc_arr)
            dist = np.linalg.norm(delta, axis=1)
            near_mask = dist <= li_o_cutoff_A
            near_o = o_idx[near_mask]
            for oi in near_o:
                if o_types[frame, oi] == O_WATER:
                    cw += 1
                elif o_types[frame, oi] == O_HYDROXIDE:
                    coh += 1
        denom = max(1, li_idx.size)
        cn_w.append(cw / denom)
        cn_oh.append(coh / denom)
    return {
        "cn_water_series": np.asarray(cn_w, dtype=float),
        "cn_hydroxide_series": np.asarray(cn_oh, dtype=float),
    }


def compute_residence_proxy(pos_A_series, Z, cell_A_series, pbc, li_o_cutoff_A, o_type_series, lag_ps, times_ps) -> dict:
    positions = np.asarray(pos_A_series, dtype=float)
    cells = np.asarray(cell_A_series, dtype=float)
    z = np.asarray(Z, dtype=int)
    pbc_arr = np.asarray(pbc, dtype=bool)
    o_types = np.asarray(o_type_series, dtype=int)
    times = np.asarray(times_ps, dtype=float)

    li_idx = np.where(z == 3)[0]
    o_idx = np.where(z == 8)[0]

    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.001
    lags = np.arange(0.0, lag_ps + dt, dt)
    residence = np.zeros(lags.shape[0], dtype=float)

    if li_idx.size == 0 or o_idx.size == 0:
        return {"lag_ps": lags, "residence_proxy": residence, "residence_proxy_oh": 0.0}

    contacts = np.zeros((positions.shape[0], li_idx.size, o_idx.size), dtype=bool)
    for frame in range(positions.shape[0]):
        for i_li, li in enumerate(li_idx):
            delta = positions[frame, o_idx] - positions[frame, li]
            delta = _min_image_delta(delta, cells[frame], pbc_arr)
            dist = np.linalg.norm(delta, axis=1)
            contacts[frame, i_li, :] = dist <= li_o_cutoff_A

    for i_lag, lag in enumerate(lags):
        shift = int(round(lag / dt))
        if shift >= positions.shape[0]:
            break
        c0 = contacts[: positions.shape[0] - shift]
        c1 = contacts[shift:]
        same = c0 & c1
        residence[i_lag] = float(np.mean(same))

    # OH-specific near-zero lag proxy.
    oh_mask = (o_types == O_HYDROXIDE)[:, o_idx]
    oh_contact = contacts & oh_mask[:, None, :]
    residence_oh = float(np.mean(oh_contact))
    return {"lag_ps": lags, "residence_proxy": residence, "residence_proxy_oh": residence_oh}


def compute_vacancy_metrics_electrode(frame_atoms_or_arrays, pristine_reference_atoms, site_match_cutoff_A) -> dict:
    if isinstance(frame_atoms_or_arrays, Atoms):
        frame_pos = frame_atoms_or_arrays.get_positions()
        frame_z = frame_atoms_or_arrays.get_atomic_numbers()
        frame_cell = frame_atoms_or_arrays.cell.array
    else:
        frame_pos = np.asarray(frame_atoms_or_arrays["positions"], dtype=float)
        frame_z = np.asarray(frame_atoms_or_arrays["Z"], dtype=int)
        frame_cell = np.asarray(frame_atoms_or_arrays.get("cell", np.eye(3)), dtype=float)

    if isinstance(pristine_reference_atoms, Atoms):
        ref_pos = pristine_reference_atoms.get_positions()
        ref_z = pristine_reference_atoms.get_atomic_numbers()
    else:
        ref_pos = np.asarray(pristine_reference_atoms["positions"], dtype=float)
        ref_z = np.asarray(pristine_reference_atoms["Z"], dtype=int)

    ref_li = ref_pos[np.asarray(ref_z) == 3]
    cur_li = frame_pos[np.asarray(frame_z) == 3]
    if ref_li.size == 0:
        return {"vacancy_fraction": 0.0, "vacancy_accessibility": 0.0, "n_vacant": 0, "n_sites": 0}

    occupied = 0
    for site in ref_li:
        if cur_li.size == 0:
            continue
        d = np.linalg.norm(cur_li - site[None, :], axis=1)
        if np.min(d) <= site_match_cutoff_A:
            occupied += 1
    n_sites = int(ref_li.shape[0])
    n_vac = int(n_sites - occupied)
    vac_frac = n_vac / max(1, n_sites)
    # proxy: higher vacancies and broader Li spread mean higher accessibility.
    li_spread = float(np.std(cur_li[:, 0]) + np.std(cur_li[:, 1]) + np.std(cur_li[:, 2])) if cur_li.size else 0.0
    accessibility = float(vac_frac * (1.0 + 0.1 * li_spread))
    return {
        "vacancy_fraction": vac_frac,
        "vacancy_accessibility": accessibility,
        "n_vacant": n_vac,
        "n_sites": n_sites,
    }
