from __future__ import annotations

import json
from pathlib import Path

from hydrorelith.pipelines import uma_water_2pt_experiment as water_mod


def test_water_2pt_experiment_exports_expected_files(tmp_path: Path, monkeypatch) -> None:
    cif_path = tmp_path / "water.cif"
    cif_path.write_text("data_I\n", encoding="utf-8")

    def _fake_run_one_phase(phase: str, config) -> None:
        assert phase == "electrolyte"
        rep_dir = (
            config.output_dir
            / "uma_torchsim_screen"
            / "electrolyte"
            / "water_298k_1atm"
            / "replica_000"
        )
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "prod.h5md").write_text("fake-h5", encoding="utf-8")
        (rep_dir / "prod_thermo.csv").write_text(
            "step,time_ps,potential_energy_eV,kinetic_energy_eV,temperature_K,volume_A3,pressure_MPa\n"
            "0,0.000,-10.0,1.0,298.0,1000.0,0.101325\n",
            encoding="utf-8",
        )

    def _fake_export_h5md_to_lammps_dump(traj_h5md: Path, out_lammpstrj: Path, **_kwargs) -> None:
        assert traj_h5md.exists()
        out_lammpstrj.parent.mkdir(parents=True, exist_ok=True)
        out_lammpstrj.write_text("ITEM: TIMESTEP\n0\n", encoding="utf-8")
        (out_lammpstrj.parent / "type_map.json").write_text("{}\n", encoding="utf-8")

    def _fake_write_lammps_eng_from_thermo_csv(thermo_csv: Path, out_eng: Path) -> None:
        assert thermo_csv.exists()
        out_eng.write_text("# fake eng\n", encoding="utf-8")

    def _fake_write_2pt_metadata(prod_thermo_csv: Path, out_json: Path, *, timestep_ps: float, dump_every_steps: int) -> None:
        assert prod_thermo_csv.exists()
        out_json.write_text(
            json.dumps(
                {
                    "timestep_ps": timestep_ps,
                    "TRAJ_DUMPFREQ": dump_every_steps,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(water_mod, "run_one_phase", _fake_run_one_phase)
    monkeypatch.setattr(water_mod, "export_h5md_to_lammps_dump", _fake_export_h5md_to_lammps_dump)
    monkeypatch.setattr(water_mod, "write_lammps_eng_from_thermo_csv", _fake_write_lammps_eng_from_thermo_csv)
    monkeypatch.setattr(water_mod, "write_2pt_metadata", _fake_write_2pt_metadata)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hrw-uma-water-2pt-experiment",
            "--cif-path",
            str(cif_path),
            "--output-dir",
            str(out_dir),
            "--condition-id",
            "water_298k_1atm",
            "--sname",
            "electrolyte",
            "--npt-ps",
            "1.0",
            "--nvt-ps",
            "2.0",
        ],
    )

    water_mod.main()

    export_dir = out_dir / "uma_torchsim_screen" / "export2pt" / "electrolyte" / "water_298k_1atm" / "replica_000"
    assert (export_dir / "prod.lammpstrj").exists()
    assert (export_dir / "prod.lammps").exists()
    assert (export_dir / "prod.eng").exists()
    assert (export_dir / "electrolyte_prod.lammpstrj").exists()
    assert (export_dir / "electrolyte_prod.lammps").exists()
    assert (export_dir / "electrolyte_prod.eng").exists()
    assert (export_dir / "type_map.json").exists()
    assert (export_dir / "2pt_metadata.json").exists()

    summary = json.loads((out_dir / "uma_torchsim_screen" / "water_2pt_experiment_summary.json").read_text(encoding="utf-8"))
    assert summary["npt_steps"] == 1000
    assert summary["nvt_steps"] == 2000
    assert summary["model_name"] == "uma-s-1p2"
