# UMA TorchSim Screen Workflow

`uma_torchsim_screen` is a GPU-first screening workflow for hydrothermal relithiation studies using TorchSim batched MD + FAIRChem UMA.

MD execution policy:
- The workflow is fail-fast and does not use a lightweight non-TorchSim fallback.
- MD runs are executed with batched TorchSim trajectories and then converted into analysis-ready `*.h5md` files.

Thermostat/barostat policy:
- `retherm` is run in NPT to relax density/volume at target T/P.
- `prod` is run in NVT and initialized from the NPT endpoint, with cell rescaled to the replica-averaged equilibrated volume from NPT.

- Electrode task: `omat`
- Electrolyte task: `omol`
- Output root: `<output_dir>/uma_torchsim_screen/`

## Input Structures

The workflow supports two input modes:

1. Recommended: explicit CSV manifests (source of truth)
2. Fallback: auto-discovery from folder trees

### Recommended manifest columns

Electrolyte (`electrolyte_manifest.csv`), one row per pre-equilibrated box:
- Required: `condition_id,structure_path,temperature_C,pressure_MPa,liOH_M,kOH_M,phase,task_name`
- Optional: `charge,spin,notes`
- Defaults: `phase=electrolyte`, `task_name=omol`, `charge=0`, `spin=1`

Electrode (`electrode_manifest.csv`), one row per structure/config:
- Required: `condition_id,structure_path,temperature_C,pressure_MPa,lithiation_fraction,vacancy_config_id,phase,task_name`
- Optional: `n_li,notes`
- Defaults: `phase=electrode`, `task_name=omat`

Generate template manifests:

```bash
hrw-uma-torchsim-manifest-template --phase electrolyte --out electrolyte_manifest.csv
hrw-uma-torchsim-manifest-template --phase electrode --out electrode_manifest.csv
```

### Discovery fallback

Use `--electrode-root` and/or `--electrolyte-root`.
The workflow recursively searches for ASE-readable structures (`.xyz`, `.extxyz`, `.vasp`/`POSCAR*`, `.cif`, `.pdb`).
T/P/composition tags are parsed from names only when present. If missing, provide `--tp-grid-csv` or `--use-default-tp-grid`.

## Default T/P Grid

When manifest rows miss T/P, or when `--use-default-tp-grid` is set, defaults are:

- 220 C @ 2.02 MPa
- 200 C @ 1.32 MPa
- 160 C @ 0.46 MPa
- 120 C @ 0.08 MPa

## Running Locally

Key runtime defaults:
- `--retherm-steps-electrode 2000`
- `--retherm-steps-electrolyte 8000`
- `--prod-steps 25000`
- `--replicas 3`

Batching/throughput controls:
- `--max-memory-scaler <float>` to reuse a known scaler and avoid repeated estimation.
- `--skip-batch-benchmark` to disable the optional batch-scaling benchmark.
- `--benchmark-steps`, `--benchmark-warmup-steps`, `--benchmark-max-systems`, `--benchmark-step-size`.
- `--precision {float32,float64}` and `--debug` for runtime diagnostics.

Resume behavior:
- Re-running MD in the same output directory resumes from existing TorchSim trajectory files when possible.
- If per-cohort trajectories disagree on last step, they are truncated to the common minimum step before resuming.

Full run:

```bash
hrw-uma-torchsim-screen \
  --electrode-manifest examples/uma_torchsim_screen/electrode_manifest.example.csv \
  --electrolyte-manifest examples/uma_torchsim_screen/electrolyte_manifest.example.csv \
  --output-dir runs/uma_screen \
  --phase both \
  --stage all \
  --device cuda \
  --max-memory-scaler 12000 \
  --benchmark-max-systems 32 \
  --benchmark-step-size 2 \
  --precision float32 \
  --electrode-reference-pristine default_structures/electrolyte_templates/Li.cif
```

Compartmentalized operation:

```bash
# Electrode-only MD on machine A
hrw-uma-torchsim-screen --electrode-manifest electrode_manifest.csv --output-dir runs/uma --phase electrode --stage md

# Electrolyte-only MD on machine B
hrw-uma-torchsim-screen --electrolyte-manifest electrolyte_manifest.csv --output-dir runs/uma --phase electrolyte --stage md

# Later, combined analysis/regression/plots
hrw-uma-torchsim-screen --output-dir runs/uma --phase both --stage analyze,regress,plots --electrode-reference-pristine <path>
```

Convenience aliases:

- `--analysis-only` => `--stage analyze,regress,plots`
- `--md-only` => `--stage md`

## Perlmutter (Slurm)

Templates are under `scripts/slurm/perlmutter/`:

- `run_electrode_md.sbatch`
- `run_electrolyte_md.sbatch`
- `run_analysis.sbatch`
- `run_export2pt.sbatch`

These request one GPU by default and include a commented 4-GPU variant plus array-job hints.

Recommended for production reruns on fixed manifests:
- Persist and reuse `--max-memory-scaler` from a prior run.
- Keep benchmark enabled for first calibration run, then use `--skip-batch-benchmark` for follow-up campaigns.

## 2PT Export

Use stage `export-2pt` (or `--export-2pt`) to generate:

- `prod.lammpstrj` with `id type xu yu zu vx vy vz` (+ extras)
- `prod.lammps` (LAMMPS trajectory alias)
- `prod.eng` (LAMMPS-like thermo log derived from TorchSim `prod_thermo.csv`)
- `type_map.json`
- `2pt_metadata.json`

Metadata includes required 2PT keys:

- `MD_AVGENERGY`
- `MD_AVGVOLUME`
- `MD_AVGTEMPERATURE`
- `TRAJ_DUMPFREQ`

Pointers for references:

- `for_chat_gpt/2pt_user_guide.pdf`
- `for_chat_gpt/2pt_paper.pdf`

## Dedicated DIRECT-tiny 2PT experiment

Use `hrw-uma-torchsim-2pt-experiment` for the fixed campaign setup:

- TorchSim-only batched UMA MD (no ASE fallback)
- one selected structure per lithiation bin from electrode `structure_generation_tiny/best_training_set`
- one selected structure per LiOH/KOH concentration from electrolyte `best_training_set`
- NPT equilibration then NVT production (`500000 fs` then `100000 fs` by default)
- electrode defaults to latest fine-tuned `omat` checkpoint in `uma_finetune_workflow`
- electrolyte defaults to base UMA model `uma-s-1p2` with task `omol`
- automatic 2PT export generation for each replica
- optional post-export execution of C++ 2PT and/or `2pt_python`

Default electrode tiny source root:

- `results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/best_training_set`

Default electrolyte source root:

- `results/publication/default_systems/electrolyte/LiOH_KOH_H2O/structure_generation/best_training_set`

Default fine-tuned electrode model source:

- `results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow`

Default T/P mapping (`K:MPa`):

- `393:0.08,433:0.46,473:1.32,493:2.02`

Example (run both electrode + electrolyte campaigns; export only):

```bash
hrw-uma-torchsim-2pt-experiment \
  --output-dir runs/uma_tiny_2pt \
  --replicas 1
```

Input-preparation only (no MD run):

```bash
hrw-uma-torchsim-2pt-experiment \
  --campaign both \
  --output-dir runs/uma_tiny_2pt_inputs \
  --prepare-only
```

Example (electrolyte-only, one structure per concentration):

```bash
hrw-uma-torchsim-2pt-experiment \
  --campaign electrolyte \
  --output-dir runs/uma_tiny_2pt_electrolyte \
  --electrolyte-model-name uma-s-1p2 \
  --electrolyte-task-name omol
```

Example (run both 2PT backends after export):

```bash
hrw-uma-torchsim-2pt-experiment \
  --output-dir runs/uma_tiny_2pt \
  --run-2pt-backends both \
  --two-pt-cpp-cmd-template "2pt_cpp_exec --traj {traj} --meta {metadata}" \
  --two-pt-python-cmd-template "python -m 2pt_python.cli --traj {traj} --meta {metadata}"
```

Install optional `2pt_python` dependency:

```bash
pip install ".[two-pt]"
```

Supported command placeholders for backend templates:

- `{export_dir}`
- `{traj}`
- `{metadata}`
- `{type_map}`

### Per-atom export semantics

- `ke_atom` is exported from masses + velocities.
- `pe_atom` is exported only when atomwise potential energies are available from model outputs.
- `atomEng` is exported only when `pe_atom` is available, with definition: `atomEng = ke_atom + pe_atom`.
- Force columns (`fx fy fz`) are exported when force arrays are present in `prod.h5md`.
- If atomwise PE is unavailable, it is not fabricated: `pe_atom` and `atomEng` are omitted and `type_map.json` reports `pe_atom_available=false` and `atom_eng_available=false`.

## Single water-box 2PT benchmark

Use `hrw-uma-water-2pt-experiment` to run one electrolyte job from a single CIF with:

- UMA `uma-s-1p2` on task `omol`
- NPT at 298 K / 1 atm for 10 ps
- NVT production for 50 ps
- 1 fs timestep
- one replica/job (no same-GPU parallel batch scheduling)

Example:

```bash
hrw-uma-water-2pt-experiment \
  --cif-path default_structures/electrolyte_templates/H2O.cif \
  --output-dir runs/uma_water_2pt \
  --sname electrolyte
```

This writes both generic and `${sname}`-prefixed 2PT inputs under export output directories:

- `prod.lammpstrj`, `prod.lammps`, `prod.eng`
- `${sname}_prod.lammpstrj`, `${sname}_prod.lammps`, `${sname}_prod.eng`
- `2pt_metadata.json`, `type_map.json`

## Output Layout

The workflow writes:

- `electrode/<condition_id>/replica_000/{retherm.h5md,retherm_thermo.csv,retherm_equilibration.json,prod.h5md,prod_thermo.csv,descriptors.json}`
- `electrolyte/<condition_id>/replica_000/{...}`
- `electrode/<condition_id>/replica_000/{retherm.trajectory.h5,prod.trajectory.h5,retherm_thermo_detailed.csv,prod_thermo_detailed.csv}`
- `electrolyte/<condition_id>/replica_000/{...}`
- `export2pt/<phase>/<condition_id>/replica_000/{prod.lammpstrj,prod.lammps,prod.eng,type_map.json,2pt_metadata.json}`
- `merged/{features.csv,regression_summary.json,pred_vs_true.csv}`
- `plots/*.png`

Phase-level run metadata and optional benchmark artifacts:
- `<output_dir>/uma_torchsim_screen/<phase>/run_config.json`
- `<output_dir>/uma_torchsim_screen/<phase>/run_state.json`
- `<output_dir>/uma_torchsim_screen/<phase>/batch_benchmark/batch_scaling.csv` (when benchmark enabled)

Plot set includes MSD/fits, RDF, CN traces + histogram, residence proxy, oxygen species counts, vacancy metrics, predicted-vs-experimental, and faceted T/P heatmaps.

Replica-aggregated error-band plots (`mean +/- std`) are also emitted, including:
- `*_msd_li_replicas_mean_std.png`
- `*_rdf_li_o_total_replicas_mean_std.png`
- `*_cn_time_series_replicas_mean_std.png`
- `*_density_retherm_equilibration.png` (NPT density evolution with equilibration cutoff marker and production-density average line)
