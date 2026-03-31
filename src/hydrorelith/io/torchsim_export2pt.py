from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hydrorelith.analysis.uma_torchsim_descriptors import unwrap_positions


AMU_TO_EV_PS2_PER_A2 = 103.642691


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
    if "atom_potential_energy" in data:
        atom_pe = _normalize_frame_atom_array(np.asarray(data["atom_potential_energy"], dtype=float), nframes, natoms)
    pe_atom_available = atom_pe is not None
    atom_eng_available = pe_atom_available

    out_lammpstrj.parent.mkdir(parents=True, exist_ok=True)
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

            cols = ["id", "type", "xu", "yu", "zu", "vx", "vy", "vz"]
            if include_ke_atom:
                cols.append("ke_atom")
            if pe_atom_available:
                cols.append("pe_atom")
                cols.append("atomEng")
            if forces_available:
                cols.extend(["fx", "fy", "fz"])
            handle.write("ITEM: ATOMS " + " ".join(cols) + "\n")

            ke_atom = 0.5 * masses * np.sum(velocities[frame] ** 2, axis=1) / AMU_TO_EV_PS2_PER_A2
            atom_eng = (ke_atom + atom_pe[frame]) if atom_eng_available else None
            for i in range(len(z)):
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


def main_export2pt_cli() -> None:
    parser = argparse.ArgumentParser(description="Export UMA TorchSim trajectory to 2PT-friendly dump")
    parser.add_argument("--traj-h5md", type=Path, required=True)
    parser.add_argument("--out-lammpstrj", type=Path, required=True)
    parser.add_argument("--prod-thermo-csv", type=Path, required=True)
    parser.add_argument("--out-metadata", type=Path, required=True)
    parser.add_argument("--timestep-ps", type=float, required=True)
    parser.add_argument("--dump-every-steps", type=int, required=True)
    args = parser.parse_args()

    export_h5md_to_lammps_dump(args.traj_h5md, args.out_lammpstrj)
    write_2pt_metadata(
        args.prod_thermo_csv,
        args.out_metadata,
        timestep_ps=args.timestep_ps,
        dump_every_steps=args.dump_every_steps,
    )


if __name__ == "__main__":
    main_export2pt_cli()
