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

    atom_pe = data.get("atom_potential_energy")
    pe_atom_available = atom_pe is not None

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
            handle.write("ITEM: ATOMS " + " ".join(cols) + "\n")

            ke_atom = 0.5 * masses * np.sum(velocities[frame] ** 2, axis=1) / AMU_TO_EV_PS2_PER_A2
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
                handle.write(" ".join(vals) + "\n")

    type_map = {
        "atomic_number_to_type": {str(k): int(v) for k, v in z_to_type.items()},
        "pe_atom_available": bool(pe_atom_available),
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
