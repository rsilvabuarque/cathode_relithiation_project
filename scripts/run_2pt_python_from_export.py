from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _build_control_text(out_prefix: str, traj_name: str, metadata: dict) -> str:
    timestep_ps = float(metadata.get("timestep_ps", 0.001))
    avg_temperature = float(metadata.get("MD_AVGTEMPERATURE", 298.15))
    avg_volume = float(metadata.get("MD_AVGVOLUME", 1.0))

    lines = [
        f"OUT_PREFIX {out_prefix}",
        f"IN_LMPTRJ {traj_name}",
        f"MD_TSTEP {timestep_ps:.12g}",
        "TRAJ_VEL_SFACTOR 1.0",
        f"MD_AVGTEMPERATURE {avg_temperature:.12g}",
        f"MD_AVGVOLUME {avg_volume:.12g}",
        "ANALYSIS_FRAME_INITIAL 1",
        "ANALYSIS_FRAME_STEP 1",
        "ANALYSIS_VAC_CORLENGTH 0.5",
        "ANALYSIS_VAC_2PT 1",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a twopoint control file from export2pt artifacts and run twopoint.",
    )
    parser.add_argument("--traj", type=Path, required=True, help="Path to prod.lammpstrj")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to 2pt_metadata.json")
    parser.add_argument("--out-prefix", type=str, default="twopoint_from_export")
    parser.add_argument("--control-name", type=str, default="twopoint.control")
    args = parser.parse_args()

    traj_path = args.traj.resolve()
    metadata_path = args.metadata.resolve()
    export_dir = traj_path.parent

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    control_path = export_dir / args.control_name
    control_text = _build_control_text(
        out_prefix=args.out_prefix,
        traj_name=traj_path.name,
        metadata=metadata,
    )
    control_path.write_text(control_text, encoding="utf-8")

    proc = subprocess.run(
        ["twopoint", "--control", str(control_path), "--root", str(export_dir)],
        cwd=export_dir,
        text=True,
        capture_output=True,
        check=False,
    )

    (export_dir / "2pt_python.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (export_dir / "2pt_python.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
