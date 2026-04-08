from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hydrorelith.analysis.uma_torchsim_descriptors import unwrap_positions


AMU_TO_EV_PS2_PER_A2 = 103.642691
ATM_PER_MPA = 9.869232667160128
PA_PER_EV_A3 = 1.602176634e11
ATM_PER_EV_A3 = PA_PER_EV_A3 / 101325.0


def _read_h5md(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as h5:
        out = {k: np.array(v) for k, v in h5.items()}
    return out


def export_h5md_to_lammps_dump(
    traj_h5md: Path,
    out_lammpstrj: Path,
    *,
    unwrap: bool = True,
    include_ke_atom: bool = True,
    lammps_template_style: bool = False,
) -> None:
    def _normalize_frame_atom_array(arr: np.ndarray, nframes: int, natoms: int) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.ndim == 1:
            if nframes != 1 or out.shape[0] != natoms:
                raise ValueError(
                    f"Invalid atomwise array shape {out.shape}; expected ({nframes},{natoms}) or ({natoms},) for a single frame"
                )
            return out.reshape(1, natoms)
        if out.ndim >= 2:
            out = out.reshape(out.shape[0], -1)
            if out.shape == (nframes, natoms):
                return out
        raise ValueError(f"Invalid atomwise array shape {out.shape}; expected ({nframes},{natoms})")

    def _normalize_frame_atom_vec3(arr: np.ndarray, nframes: int, natoms: int) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.ndim == 2:
            if nframes != 1 or out.shape != (natoms, 3):
                raise ValueError(
                    f"Invalid vector array shape {out.shape}; expected ({nframes},{natoms},3) or ({natoms},3) for a single frame"
                )
            return out.reshape(1, natoms, 3)
        if out.ndim == 3 and out.shape == (nframes, natoms, 3):
            return out
        raise ValueError(f"Invalid vector array shape {out.shape}; expected ({nframes},{natoms},3)")

    data = _read_h5md(traj_h5md)
    positions = np.asarray(data["positions"], dtype=float)
    velocities = np.asarray(data["velocities"], dtype=float)
    cells = np.asarray(data["cell"], dtype=float)
    z = np.asarray(data["atomic_numbers"], dtype=int)
    masses = np.asarray(data["masses"], dtype=float)
    pbc = np.asarray(data.get("pbc", [1, 1, 1]), dtype=bool)
    steps = np.asarray(data.get("step", np.arange(positions.shape[0])), dtype=int)

    if cells.ndim != 3:
        raise ValueError("Only trajectory cell data with shape (n_frames,3,3) is supported")

    # Triclinic support would require tilt factors in BOX BOUNDS; fail fast for now.
    offdiag = np.abs(cells[:, np.triu_indices(3, k=1)[0], np.triu_indices(3, k=1)[1]])
    if np.any(offdiag > 1e-10):
        raise ValueError(
            "Triclinic cell detected; LAMMPS dump export currently supports orthorhombic cells only."
        )

    if unwrap:
        positions = unwrap_positions(positions, cells, pbc)

    unique_z = sorted(set(int(v) for v in z.tolist()))
    z_to_type = {zz: i + 1 for i, zz in enumerate(unique_z)}
    type_ids = np.array([z_to_type[int(zz)] for zz in z], dtype=int)

    nframes = int(positions.shape[0])
    natoms = int(len(z))

    forces = None
    if "forces" in data:
        forces = _normalize_frame_atom_vec3(np.asarray(data["forces"], dtype=float), nframes, natoms)
    forces_available = forces is not None

    atom_pe = None
    for key in (
        "energies",
        "atom_potential_energy",
        "pe_atom",
        "potential_energy_per_atom",
        "potential_energy_atom",
        "energy_per_atom_eV",
        "potential_energy_eV",
    ):
        if key not in data:
            continue
        try:
            atom_pe = _normalize_frame_atom_array(np.asarray(data[key], dtype=float), nframes, natoms)
            break
        except ValueError:
            continue
    pe_atom_available = atom_pe is not None
    atom_eng_available = pe_atom_available

    mol_ids = np.ones((natoms,), dtype=int)
    for key in ("molecule_ids", "molecule_id", "mol_ids", "mol"):
        if key not in data:
            continue
        raw = np.asarray(data[key])
        if raw.ndim == 1 and raw.shape[0] == natoms:
            mol_ids = np.asarray(np.rint(raw), dtype=int)
            break

    charge_static = np.zeros((natoms,), dtype=float)
    charge_frame = None
    for key in ("charges", "charge", "q"):
        if key not in data:
            continue
        raw = np.asarray(data[key], dtype=float)
        if raw.ndim == 1 and raw.shape[0] == natoms:
            charge_static = raw
            break
        try:
            charge_frame = _normalize_frame_atom_array(raw, nframes, natoms)
            break
        except ValueError:
            continue

    out_lammpstrj.parent.mkdir(parents=True, exist_ok=True)
    written_cols: list[str] = []
    with out_lammpstrj.open("w", encoding="utf-8") as handle:
        for frame in range(positions.shape[0]):
            cell = cells[frame]
            xhi, yhi, zhi = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
            handle.write("ITEM: TIMESTEP\n")
            handle.write(f"{int(steps[frame])}\n")
            handle.write("ITEM: NUMBER OF ATOMS\n")
            handle.write(f"{len(z)}\n")
            handle.write("ITEM: BOX BOUNDS pp pp pp\n")
            handle.write(f"0.0 {xhi:.10f}\n")
            handle.write(f"0.0 {yhi:.10f}\n")
            handle.write(f"0.0 {zhi:.10f}\n")

            if lammps_template_style:
                cols = ["id", "mol", "type", "q", "xu", "yu", "zu", "vx", "vy", "vz", "v_atomEng"]
            else:
                cols = ["id", "type", "xu", "yu", "zu", "vx", "vy", "vz"]
                if include_ke_atom:
                    cols.append("ke_atom")
                if pe_atom_available:
                    cols.append("pe_atom")
                    cols.append("atomEng")
                if forces_available:
                    cols.extend(["fx", "fy", "fz"])
            written_cols = cols
            handle.write("ITEM: ATOMS " + " ".join(cols) + "\n")

            ke_atom = 0.5 * masses * np.sum(velocities[frame] ** 2, axis=1) / AMU_TO_EV_PS2_PER_A2
            atom_eng = (ke_atom + atom_pe[frame]) if atom_eng_available else None
            for i in range(len(z)):
                q_val = charge_frame[frame, i] if charge_frame is not None else charge_static[i]
                v_atomeng = float(atom_eng[i]) if atom_eng_available else float(ke_atom[i])
                if lammps_template_style:
                    vals = [
                        str(i + 1),
                        str(int(mol_ids[i])),
                        str(type_ids[i]),
                        f"{float(q_val):.10f}",
                        f"{positions[frame, i, 0]:.10f}",
                        f"{positions[frame, i, 1]:.10f}",
                        f"{positions[frame, i, 2]:.10f}",
                        f"{velocities[frame, i, 0]:.10f}",
                        f"{velocities[frame, i, 1]:.10f}",
                        f"{velocities[frame, i, 2]:.10f}",
                        f"{v_atomeng:.10f}",
                    ]
                else:
                    vals = [
                        str(i + 1),
                        str(type_ids[i]),
                        f"{positions[frame, i, 0]:.10f}",
                        f"{positions[frame, i, 1]:.10f}",
                        f"{positions[frame, i, 2]:.10f}",
                        f"{velocities[frame, i, 0]:.10f}",
                        f"{velocities[frame, i, 1]:.10f}",
                        f"{velocities[frame, i, 2]:.10f}",
                    ]
                    if include_ke_atom:
                        vals.append(f"{ke_atom[i]:.10f}")
                    if pe_atom_available:
                        vals.append(f"{float(atom_pe[frame, i]):.10f}")
                        vals.append(f"{float(atom_eng[i]):.10f}")
                    if forces_available:
                        vals.extend(
                            [
                                f"{float(forces[frame, i, 0]):.10f}",
                                f"{float(forces[frame, i, 1]):.10f}",
                                f"{float(forces[frame, i, 2]):.10f}",
                            ]
                        )
                handle.write(" ".join(vals) + "\n")

    type_map = {
        "atomic_number_to_type": {str(k): int(v) for k, v in z_to_type.items()},
        "forces_available": bool(forces_available),
        "ke_atom_available": bool(include_ke_atom),
        "pe_atom_available": bool(pe_atom_available),
        "atom_eng_available": bool(atom_eng_available),
        "atom_eng_definition": "atomEng = ke_atom + pe_atom (eV)",
        "v_atomeng_available": True,
        "v_atomeng_contains_pe_atom": bool(pe_atom_available),
        "v_atomeng_definition": (
            "v_atomEng = ke_atom + pe_atom (eV)"
            if pe_atom_available
            else "v_atomEng = ke_atom only (pe_atom unavailable)"
        ),
        "lammps_template_style": bool(lammps_template_style),
        "dump_columns": written_cols,
    }
    (out_lammpstrj.parent / "type_map.json").write_text(json.dumps(type_map, indent=2), encoding="utf-8")


def write_2pt_metadata(
    prod_thermo_csv: Path,
    out_json: Path,
    *,
    timestep_ps: float,
    dump_every_steps: int,
) -> None:
    energies, volumes, temps = [], [], []
    with prod_thermo_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pe = float(row.get("potential_energy_eV", 0.0))
            ke = float(row.get("kinetic_energy_eV", 0.0))
            energies.append(pe + ke)
            volumes.append(float(row.get("volume_A3", 0.0)))
            temps.append(float(row.get("temperature_K", 0.0)))

    payload = {
        "MD_AVGENERGY": float(np.mean(energies)) if energies else 0.0,
        "MD_AVGVOLUME": float(np.mean(volumes)) if volumes else 0.0,
        "MD_AVGTEMPERATURE": float(np.mean(temps)) if temps else 0.0,
        "TRAJ_DUMPFREQ": int(dump_every_steps),
        "timestep_ps": float(timestep_ps),
        "docs": {
            "user_guide": "for_chat_gpt/2pt_user_guide.pdf",
            "paper": "for_chat_gpt/2pt_paper.pdf",
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_lammps_eng_from_thermo_csv(
    thermo_csv: Path,
    out_eng: Path,
) -> None:
    def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
        raw = row.get(key, "")
        if raw is None:
            return float(default)
        text = str(raw).strip()
        if not text:
            return float(default)
        return float(text)

    def _time_fs(row: dict[str, str], step: int) -> float:
        if str(row.get("time_fs", "")).strip():
            return _float(row, "time_fs")
        if str(row.get("time_ps", "")).strip():
            return _float(row, "time_ps") * 1000.0
        return float(step)

    def _pressure_atm(row: dict[str, str]) -> float:
        if str(row.get("pressure_atm", "")).strip():
            return _float(row, "pressure_atm")
        if str(row.get("pressure_MPa", "")).strip():
            return _float(row, "pressure_MPa") * ATM_PER_MPA
        if str(row.get("pressure_eV_per_A3", "")).strip():
            return _float(row, "pressure_eV_per_A3") * ATM_PER_EV_A3
        return 0.0

    rows: list[tuple[int, float, float, float, float, float, float, float]] = []
    with thermo_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(round(_float(row, "step", default=0.0)))
            time_fs = _time_fs(row, step)
            temp_k = _float(row, "temperature_K", default=0.0)
            press_atm = _pressure_atm(row)
            volume_a3 = _float(row, "volume_A3", default=0.0)
            pe_ev = _float(row, "potential_energy_eV", default=0.0)
            ke_ev = _float(row, "kinetic_energy_eV", default=0.0)
            etotal_ev = pe_ev + ke_ev
            rows.append((step, time_fs, temp_k, press_atm, volume_a3, etotal_ev, pe_ev, ke_ev))

    out_eng.parent.mkdir(parents=True, exist_ok=True)
    with out_eng.open("w", encoding="utf-8") as handle:
        handle.write("# LAMMPS-like thermo log synthesized from TorchSim outputs\n")
        handle.write("# thermo 4\n")
        handle.write("# thermo_style custom etotal ke temp pe ebond eangle edihed eimp evdwl ecoul elong ebond press vol\n")
        handle.write("# thermo_modify line multi\n")
        for step, time_fs, temp_k, press_atm, volume_a3, etotal_ev, pe_ev, ke_ev in rows:
            handle.write(f"----------------------------- Step {step:d} -----------------------------\n")
            handle.write(
                f"TotEng = {etotal_ev:.10f} KinEng = {ke_ev:.10f} Temp = {temp_k:.10f} "
                f"PotEng = {pe_ev:.10f}\n"
            )
            handle.write("E_bond = 0.0000000000 E_angle = 0.0000000000 E_dihed = 0.0000000000 E_impro = 0.0000000000\n")
            handle.write("E_vdwl = 0.0000000000 E_coul = 0.0000000000 E_long = 0.0000000000\n")
            handle.write(f"Press = {press_atm:.10f} Volume = {volume_a3:.10f} Time_fs = {time_fs:.6f}\n")


def main_export2pt_cli() -> None:
    parser = argparse.ArgumentParser(description="Export UMA TorchSim trajectory to 2PT-friendly dump")
    parser.add_argument("--traj-h5md", type=Path, required=True)
    parser.add_argument("--out-lammpstrj", type=Path, required=True)
    parser.add_argument("--out-eng", type=Path, default=None)
    parser.add_argument("--prod-thermo-csv", type=Path, required=True)
    parser.add_argument("--out-metadata", type=Path, required=True)
    parser.add_argument("--timestep-ps", type=float, required=True)
    parser.add_argument("--dump-every-steps", type=int, required=True)
    args = parser.parse_args()

    export_h5md_to_lammps_dump(args.traj_h5md, args.out_lammpstrj)
    if args.out_eng is not None:
        write_lammps_eng_from_thermo_csv(args.prod_thermo_csv, args.out_eng)
    write_2pt_metadata(
        args.prod_thermo_csv,
        args.out_metadata,
        timestep_ps=args.timestep_ps,
        dump_every_steps=args.dump_every_steps,
    )


if __name__ == "__main__":
    main_export2pt_cli()
