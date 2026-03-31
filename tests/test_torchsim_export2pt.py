from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hydrorelith.io.torchsim_export2pt import AMU_TO_EV_PS2_PER_A2, export_h5md_to_lammps_dump


@pytest.fixture
def h5py_mod():
    return pytest.importorskip("h5py")


def _write_minimal_h5md(path: Path, h5py_mod, *, with_atom_pe: bool) -> None:
    positions = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=float)
    velocities = np.array([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]], dtype=float)
    forces = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=float)
    cell = np.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=float)
    atomic_numbers = np.array([1, 8], dtype=int)
    masses = np.array([1.0, 16.0], dtype=float)

    with h5py_mod.File(path, "w") as h5:
        h5.create_dataset("positions", data=positions)
        h5.create_dataset("velocities", data=velocities)
        h5.create_dataset("forces", data=forces)
        h5.create_dataset("cell", data=cell)
        h5.create_dataset("atomic_numbers", data=atomic_numbers)
        h5.create_dataset("masses", data=masses)
        h5.create_dataset("pbc", data=np.array([1, 1, 1], dtype=np.int8))
        h5.create_dataset("step", data=np.array([0], dtype=int))
        if with_atom_pe:
            h5.create_dataset("atom_potential_energy", data=np.array([[0.1, 0.2]], dtype=float))


def _first_atoms_header_and_row(path: Path) -> tuple[list[str], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS "):
            cols = line.split()[2:]
            vals = [float(v) for v in lines[idx + 1].split()]
            return cols, vals
    raise AssertionError("No ITEM: ATOMS header found")


def test_export_includes_atom_eng_when_atom_pe_present(tmp_path: Path, h5py_mod) -> None:
    traj = tmp_path / "traj.h5md"
    out_dump = tmp_path / "prod.lammpstrj"
    _write_minimal_h5md(traj, h5py_mod, with_atom_pe=True)

    export_h5md_to_lammps_dump(traj, out_dump, unwrap=True, include_ke_atom=True)

    cols, vals = _first_atoms_header_and_row(out_dump)
    assert "ke_atom" in cols
    assert "pe_atom" in cols
    assert "atomEng" in cols
    assert "fx" in cols and "fy" in cols and "fz" in cols

    col_to_val = {name: vals[i] for i, name in enumerate(cols)}
    expected_ke = 0.5 * 1.0 * (2.0**2) / AMU_TO_EV_PS2_PER_A2
    assert col_to_val["ke_atom"] == pytest.approx(expected_ke)
    assert col_to_val["pe_atom"] == pytest.approx(0.1)
    assert col_to_val["atomEng"] == pytest.approx(expected_ke + 0.1)
    assert col_to_val["fx"] == pytest.approx(0.1)
    assert col_to_val["fy"] == pytest.approx(0.2)
    assert col_to_val["fz"] == pytest.approx(0.3)

    type_map = json.loads((tmp_path / "type_map.json").read_text(encoding="utf-8"))
    assert type_map["forces_available"] is True
    assert type_map["ke_atom_available"] is True
    assert type_map["pe_atom_available"] is True
    assert type_map["atom_eng_available"] is True


def test_export_omits_atom_eng_when_atom_pe_absent(tmp_path: Path, h5py_mod) -> None:
    traj = tmp_path / "traj_no_pe.h5md"
    out_dump = tmp_path / "prod_no_pe.lammpstrj"
    _write_minimal_h5md(traj, h5py_mod, with_atom_pe=False)

    export_h5md_to_lammps_dump(traj, out_dump, unwrap=True, include_ke_atom=True)

    cols, _ = _first_atoms_header_and_row(out_dump)
    assert "ke_atom" in cols
    assert "pe_atom" not in cols
    assert "atomEng" not in cols
    assert "fx" in cols and "fy" in cols and "fz" in cols

    type_map = json.loads((tmp_path / "type_map.json").read_text(encoding="utf-8"))
    assert type_map["forces_available"] is True
    assert type_map["ke_atom_available"] is True
    assert type_map["pe_atom_available"] is False
    assert type_map["atom_eng_available"] is False
