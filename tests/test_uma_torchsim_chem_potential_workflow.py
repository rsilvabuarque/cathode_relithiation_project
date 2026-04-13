from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hydrorelith.pipelines import uma_torchsim_chem_potential_workflow as wf


def test_parse_tp_combos_default_like_string() -> None:
    combos = wf._parse_tp_combos("220:2.02,200:1.32,160:0.46,120:0.08")
    assert combos == [(220.0, 2.02), (200.0, 1.32), (160.0, 0.46), (120.0, 0.08)]


def test_build_manifest_rows_electrolyte_fields(tmp_path: Path) -> None:
    prepared = [
        wf.PreparedStructure(
            source_path=tmp_path / "source.cif",
            prepared_path=tmp_path / "prepared.extxyz",
            condition_label="sample",
            lithiation_fraction=None,
            lioh_m=2.0,
            koh_m=2.0,
        )
    ]
    rows = wf._build_manifest_rows("electrolyte", prepared, [(120.0, 0.08)])
    assert len(rows) == 1
    row = rows[0]
    assert row["phase"] == "electrolyte"
    assert row["task_name"] == "omol"
    assert row["liOH_M"] == "2"
    assert row["kOH_M"] == "2"
    assert row["condition_id"].startswith("el_t393k_p0p08mpa")


def test_write_py2pt_group_file_from_h5md(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")

    h5_path = tmp_path / "prod.h5md"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("atomic_numbers", data=np.array([3, 8, 1, 1], dtype=int))

    group_path = tmp_path / "groups.grps"
    has_li = wf._write_py2pt_group_file(h5_path, group_path)

    assert has_li is True
    text = group_path.read_text(encoding="utf-8")
    assert "[group1]" in text
    assert "atoms = 1" in text
    assert "[group2]" in text


def test_extract_aq_group_total_direct_layout(tmp_path: Path) -> None:
    thermo_path = tmp_path / "sample.thermo"
    thermo_path.write_text(
        "A_q (kJ/mol/atom) -19987.222 -84057.795 -83305.821\n",
        encoding="utf-8",
    )
    assert wf._extract_aq_group_total(thermo_path, group_idx=1) == pytest.approx(-19987.222)
    assert wf._extract_aq_group_total(thermo_path, group_idx=2) == pytest.approx(-84057.795)


def test_extract_aq_group_total_legacy_layout(tmp_path: Path) -> None:
    thermo_path = tmp_path / "legacy.thermo"
    thermo_path.write_text(
        "A_q -1 -2 -3 -4 -5 -6 -7 -8\n",
        encoding="utf-8",
    )
    # Legacy parser expects the total term at index 4*(group_idx-1)+3.
    assert wf._extract_aq_group_total(thermo_path, group_idx=2) == pytest.approx(-8.0)


def test_write_py2pt_ini_molecular_with_topology(tmp_path: Path) -> None:
    ini_path = tmp_path / "sample.ini"
    wf._write_py2pt_ini(
        ini_path,
        trajectory_name="prod.h5md",
        group_name="groups.grps",
        timestep_ps=0.001,
        prefix="2pt_output/sample",
        mode=4,
        topology_path="/tmp/system.data",
        topology_format="LAMMPS",
    )

    text = ini_path.read_text(encoding="utf-8")
    assert "topology = /tmp/system.data" in text
    assert "topology_format = LAMMPS" in text
    assert "mode = 4" in text


def test_infer_topology_for_py2pt_lammps_data(tmp_path: Path) -> None:
    data_path = tmp_path / "system.data"
    data_path.write_text("header\n", encoding="utf-8")
    topo, fmt = wf._infer_topology_for_py2pt(str(data_path))
    assert topo == str(data_path)
    assert fmt == "LAMMPS"


def test_extract_li_k_conc_parses_publication_p_tokens() -> None:
    li, k = wf._extract_li_k_conc(Path("data.LiOH_3p2_KOH_0p8_seed01"))
    assert li == pytest.approx(3.2)
    assert k == pytest.approx(0.8)


def test_extract_lithiation_parses_poscar_milli_suffix() -> None:
    lith = wf._extract_lithiation(Path("POSCAR_000451"))
    assert lith == pytest.approx(0.451)


def test_discover_structure_files_supports_publication_lammps_names(tmp_path: Path) -> None:
    data_path = tmp_path / "data.LiOH_2p0_KOH_2p0_seed01"
    data_path.write_text("LAMMPS data\n", encoding="utf-8")

    files = wf._discover_structure_files(tmp_path)
    assert files == [data_path]


def test_read_structure_any_publication_lammps_name_uses_lammps_reader(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "data.LiOH_3p2_KOH_0p8_seed01"
    data_path.write_text("LAMMPS data\n", encoding="utf-8")

    observed: dict[str, object] = {}

    def _fake_ase_read(path, format=None):
        observed["path"] = path
        observed["format"] = format
        return "sentinel"

    monkeypatch.setattr(wf, "ase_read", _fake_ase_read)
    result = wf._read_structure_any(data_path)

    assert result == "sentinel"
    assert observed["path"] == data_path
    assert observed["format"] == "lammps-data"
