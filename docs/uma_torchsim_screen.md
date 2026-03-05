# UMA TorchSim Screen Workflow

`uma_torchsim_screen` is a GPU-first screening workflow for hydrothermal relithiation studies using TorchSim batched MD + FAIRChem UMA.

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

Full run:

```bash
hrw-uma-torchsim-screen \
  --electrode-manifest examples/uma_torchsim_screen/electrode_manifest.example.csv \
  --electrolyte-manifest examples/uma_torchsim_screen/electrolyte_manifest.example.csv \
  --output-dir runs/uma_screen \
  --phase both \
  --stage all \
  --device cuda \
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

## 2PT Export

Use stage `export-2pt` (or `--export-2pt`) to generate:

- `prod.lammpstrj` with `id type xu yu zu vx vy vz` (+ extras)
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

### pe/atom limitation

- `ke_atom` is always exported from masses + velocities.
- `pe_atom` is exported only when atomwise potential energies are available from model outputs.
- If unavailable, atomwise PE is not fabricated; `type_map.json` marks `pe_atom_available=false` and total PE remains in thermo logs.

## Output Layout

The workflow writes:

- `electrode/<condition_id>/replica_000/{retherm.h5md,prod.h5md,prod_thermo.csv,descriptors.json}`
- `electrolyte/<condition_id>/replica_000/{...}`
- `export2pt/<phase>/<condition_id>/replica_000/{prod.lammpstrj,type_map.json,2pt_metadata.json}`
- `merged/{features.csv,regression_summary.json,pred_vs_true.csv}`
- `plots/*.png`

Plot set includes MSD/fits, RDF, CN traces + histogram, residence proxy, oxygen species counts, vacancy metrics, predicted-vs-experimental, and faceted T/P heatmaps.
