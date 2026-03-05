from __future__ import annotations

import json
from pathlib import Path

from hydrorelith.io.structure_manifest import write_template_manifest
from hydrorelith.pipelines.uma_torchsim_screen import main


def test_uma_torchsim_screen_smoke(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]

    el_manifest = tmp_path / "electrolyte_manifest.csv"
    ed_manifest = tmp_path / "electrode_manifest.csv"
    write_template_manifest(el_manifest, "electrolyte")
    write_template_manifest(ed_manifest, "electrode")

    el_text = el_manifest.read_text(encoding="utf-8").replace(
        "path/to/electrolyte.xyz", str(root / "default_structures/electrolyte_templates/H2O.cif")
    )
    ed_text = ed_manifest.read_text(encoding="utf-8").replace(
        "path/to/electrode.cif", str(root / "default_structures/electrolyte_templates/Li.cif")
    )
    ed_text = ed_text.replace("0.85", "1.0")
    el_manifest.write_text(el_text, encoding="utf-8")
    ed_manifest.write_text(ed_text, encoding="utf-8")

    out_dir = tmp_path / "out"
    pristine_ref = root / "default_structures/electrolyte_templates/Li.cif"

    monkeypatch.setattr(
        "sys.argv",
        [
            "hrw-uma-torchsim-screen",
            "--electrode-manifest",
            str(ed_manifest),
            "--electrolyte-manifest",
            str(el_manifest),
            "--output-dir",
            str(out_dir),
            "--phase",
            "both",
            "--stage",
            "all",
            "--device",
            "cpu",
            "--replicas",
            "1",
            "--retherm-steps-electrode",
            "20",
            "--retherm-steps-electrolyte",
            "20",
            "--prod-steps",
            "20",
            "--dump-every-steps",
            "2",
            "--electrode-reference-pristine",
            str(pristine_ref),
        ],
    )

    main()

    base = out_dir / "uma_torchsim_screen"
    assert (base / "electrode").exists()
    assert (base / "electrolyte").exists()

    prod_files = list(base.glob("**/prod.h5md"))
    assert prod_files

    export_files = list((base / "export2pt").glob("**/prod.lammpstrj"))
    assert export_files

    desc_files = list(base.glob("**/descriptors.json"))
    assert desc_files
    sample = json.loads(desc_files[0].read_text(encoding="utf-8"))
    assert "D_li_A2_per_ps" in sample

    assert (base / "plots" / "score_distribution.png").exists()
    assert (base / "plots" / "heatmap_tp_by_x_and_conc.png").exists()
