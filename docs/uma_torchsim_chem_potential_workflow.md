# UMA TorchSim Chemical-Potential Workflow

This workflow runs TorchSim MD + py2pt to estimate Li chemical potential for either electrode or electrolyte systems.

CLI entry point:

```bash
hrw-uma-torchsim-chem-potential --help
```

## What it does

Given either an input structure directory or a manifest, it can:

1. Build a phase manifest for all structures and default/custom temperature-pressure combos.
2. Run TorchSim UMA MD (`NPT` rethermalization + `NVT` production) with batched single-GPU execution.
3. Export 2PT-ready artifacts from TorchSim outputs.
4. Generate py2pt `.ini` and `.grps` files per replica and run py2pt.
5. Produce summary plots:
   - `master_thermo_evolution_all_runs.png`
   - `master_final_stats_vs_concentration.png` (electrolyte)
   - `master_final_stats_vs_lithiation.png` (electrode)
   - `li_chemical_potential_vs_concentration.png` (electrolyte)
   - `li_chemical_potential_vs_lithiation.png` (electrode)

## Supported input formats

The `--input-dir` scan accepts:

- POSCAR (`POSCAR*`)
- LAMMPS data (`.data`, `.lammps`, `.lmp`)
- CIF (`.cif`)
- BGF (`.bgf`)
- Also supports `.xyz`, `.extxyz`, `.vasp`, `.pdb`

All inputs are converted to `extxyz` under:

- `<output_dir>/uma_torchsim_screen/prepared_structures/<phase>/`

## Default temperature-pressure combos

Defaults are encoded in Celsius/MPa:

- `220C, 2.02 MPa`
- `200C, 1.32 MPa`
- `160C, 0.46 MPa`
- `120C, 0.08 MPa`

Override with:

```bash
--tp-combos "220:2.02,200:1.32,160:0.46,120:0.08"
```

## Local run example

```bash
hrw-uma-torchsim-chem-potential \
  --system-type electrolyte \
  --input-dir /path/to/electrolyte_structures \
  --output-dir runs/uma_chem_potential \
  --replicas 15 \
  --device cuda \
  --model-name uma-s-1p2 \
  --py2pt-command py2pt
```

## Publication-classical full-MD runs

The repository now includes clone-visible publication input structures under:

- `results/publication/default_systems/electrolyte/LiOH_KOH_H2O/classical_forcefield/final_data_files`
- `results/publication/default_systems/electrode/LCO_mp-22526/classical_forcefield/POSCAR_directory`

Use the commands below to run MD plus py2pt with maximum practical parallelism on a single node:

- Keep MD enabled by not passing `--skip-md`.
- TorchSim autobatching/memory estimation is handled internally; set `--max-memory-scaler` only when you want to reuse a known value.
- Use all visible CPU cores for py2pt via `--py2pt-workers "$(nproc)"`.

Electrolyte (publication classical data files):

```bash
hrw-uma-torchsim-chem-potential \
  --system-type electrolyte \
  --input-dir results/publication/default_systems/electrolyte/LiOH_KOH_H2O/classical_forcefield/final_data_files \
  --output-dir runs/publication/uma_chem_potential_electrolyte \
  --device cuda \
  --replicas 15 \
  --model-name uma-s-1p2 \
  --py2pt-command py2pt \
  --py2pt-workers "$(nproc)"
```

Electrode (publication POSCAR directory):

```bash
hrw-uma-torchsim-chem-potential \
  --system-type electrode \
  --input-dir results/publication/default_systems/electrode/LCO_mp-22526/classical_forcefield/POSCAR_directory \
  --output-dir runs/publication/uma_chem_potential_electrode \
  --device cuda \
  --replicas 15 \
  --model-name uma-s-1p2 \
  --py2pt-command py2pt \
  --py2pt-workers "$(nproc)"
```

## Slurm-generation mode

To generate per-condition MD scripts plus postprocessing scripts:

```bash
hrw-uma-torchsim-chem-potential \
  --system-type electrode \
  --input-dir /path/to/electrode_structures \
  --output-dir runs/uma_chem_potential \
  --execution-mode slurm \
  --slurm-account <account>
```

Scripts are written under:

- `<output_dir>/uma_torchsim_screen/slurm/scripts/`
- `<output_dir>/uma_torchsim_screen/slurm/manifests/`

## Installing fork-pinned dependencies

For reproducible UMA/TorchSim/py2pt behavior matching this experiment:

```bash
pip install ".[uma-forks]"
```

This installs:

- `fairchem-core` from `feature/uma-per-atom-energy-inference`
- `torch-sim-atomistic` from `feature/per-atom-energy-reporting`
- `py2pt` from `fix/torchsim-flat-h5md-compat` in `rsilvabuarque/py2pt`
