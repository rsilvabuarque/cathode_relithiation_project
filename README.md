# Hydrothermal Cathode Relithiation Workflows

Repository backbone for a computational–experimental collaboration on hydrothermal relithiation. The project is designed to be material-agnostic while shipping with a default example pair:

- Electrode: LCO (LiCoO2)
- Electrolyte: LiOH/KOH + H2O

## Project goal

Build an end-to-end workflow to:

1. Generate structures for electrode/electrolyte datasets.
2. Run DFT (default: VASP) to create reference data.
3. Train ML interatomic potentials (initially DeepMD-kit; extensible to other trainers).
4. Run LAMMPS equilibrations over temperature, pressure, and ion concentration.
5. Run 2PT analysis to estimate equilibrium chemical potentials.
6. Compare electrode vs electrolyte chemical potential differences across conditions.
7. Correlate computed driving forces with measured relithiation rates and map optimal hydrothermal reactor conditions.

## Current status

This commit provides **backbone infrastructure only** and a detailed scaffold for the first active module:

- Electrode structure generation pipeline (interfaces, config schema, output layout, CLI).
- Default hydrothermal-oriented sampling ratios and retention policies.
- DIRECT subset selection placeholder.

Current executable stage includes pristine loading (MPID/local), delithiation candidate generation, rattling via hiPhive or MLFF-MD (UMA/MatGL), and DIRECT selection with plotting outputs.

## Repository layout

```text
cathode_relithiation_project/
├── README.md
├── pyproject.toml
├── docs/
│   ├── architecture.md
│   └── electrode_structure_generation.md
├── examples/
│   └── electrode_generation_lco_lioh.yaml
├── src/
│   └── hydrorelith/
│       ├── __init__.py
│       ├── cli.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── defaults.py
│       │   └── schemas.py
│       └── pipelines/
│           ├── __init__.py
│           ├── electrode_structure_generation.py
│           └── electrolyte_structure_generation.py
└── for_chat_gpt/
    └── user-provided context files only
```

## Electrolyte structure generation (amorphous builder + optional rattling)

New command:

```bash
hrw-electrolyte-generate --help
```

Key inputs are available on CLI (with publication defaults pre-populated):

- solvent template: `--solvent name=path` (default `H2O=default_structures/electrolyte_templates/H2O.cif`)
- ion templates: `--li-template`, `--k-template`, `--oh-template` (defaults in `default_structures/electrolyte_templates/`)
- concentration grid: `--li-k-concentrations "4/0,3.5/0.5,...,0/4"` (LiOH/KOH in mol/kg-solvent)
- build constraints: `--max-atoms` and `--structures-per-concentration`
- solvent density: `--solvent-density-g-cm3` (auto defaults to `1.0` for water-like solvent names)
- publication temperature/pressure defaults: `393 433 473 493 K` with `0.08 0.46 1.32 2.02 MPa`
- total UMA MD step budget: `--md-total-steps` (default `60000`, distributed across all MD runs)

The pipeline:

1. Builds amorphous cubic structures near target molalities while respecting max atoms.
2. Writes generation summary to `<output_dir>/electrolyte_generation_overview.json`.
3. Optionally applies hiPhive rattling or UMA-MD and then DIRECT down-selection.

For `--rattle-engine all`, each active engine now generates a full pre-DIRECT pool (`max_structures * oversampling_factor`) before option-wise DIRECT comparison (`hiphive`, `uma`, `combined`).

For quick generation-only checks, use `--skip-rattling`.

## SCF parallelization benchmark tool (Perlmutter CPU node)

New command:

```bash
hrw-scf-parallel-benchmark --help
```

Purpose:

- Discover DIRECT-filtered structures from electrode/electrolyte `best_training_set` folders.
- Sample a user-defined number of structures per system (default `3`).
- Generate VASP benchmark cases for MPI/OMP, `KPAR`, and `NCORE` sweeps on a full Perlmutter CPU node.
- Prepare a single Slurm script to run all benchmark cases (`run_all_parallel_benchmarks.slurm`).
- Parse `OUTCAR`/`OSZICAR`, then write per-structure and averaged performance data plus plots.

The tool uses INCAR templates from:

- `for_chat_gpt/performance_analysis_input_files/electrode/INCAR_ELECTRODE`
- `for_chat_gpt/performance_analysis_input_files/electrolyte/INCAR_ELECTROLYTE`

For electrode systems, `MAGMOM` token replacement is automatic for `N_Li`, `N_Co`, and `N_O` using the selected structure composition.

If `KPOINTS` templates are not present, the tool auto-generates gamma-only `KPOINTS` (`1x1x1`).
If `POTCAR` templates are missing, cases are still generated and Slurm runs are marked as `missing_potcar` until `POTCAR` files are added.

Generation example (3 structures per system, `NELM=15`):

```bash
hrw-scf-parallel-benchmark \
  --electrode-structures-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation \
  --electrolyte-structures-dir results/publication/default_systems/electrolyte/LiOH_KOH_H2O/structure_generation \
  --template-input-root for_chat_gpt/performance_analysis_input_files \
  --n-structures 3 \
  --scf-steps 15 \
  --output-dir results/performance/scf_parallelization
```

Run on Perlmutter:

```bash
cd results/performance/scf_parallelization
sbatch run_all_parallel_benchmarks.slurm
```

Analyze and plot after runs finish:

```bash
hrw-scf-parallel-benchmark \
  --analyze-only \
  --benchmark-root results/performance/scf_parallelization
```

Analysis outputs:

```text
<output_dir>/analysis/
├── records_per_case.csv
├── records_average.csv
├── summary.json
└── plots/
    ├── electrode_mpi_omp_vs_cores.png
    ├── electrode_kpar_sweep.png
    ├── electrode_ncore_sweep.png
    ├── electrode_parallel_efficiency.png
    ├── electrolyte_mpi_omp_vs_cores.png
    ├── electrolyte_kpar_sweep.png
    ├── electrolyte_ncore_sweep.png
    └── electrolyte_parallel_efficiency.png
```

## Electrode post-generation VASP workflow (prepare → submit/status → UMA-vs-VASP analysis)

New command:

```bash
hrw-electrode-vasp-workflow --help
```

Subcommands:

- `prepare-inputs`: build per-structure SCF case folders from `best_training_set`
- `submit`: submit pending (and optionally fizzled) cases via `sbatch`
- `status`: classify `completed/running/fizzled/pending`
- `analyze-uma-vs-vasp`: run UMA single-point checks on completed VASP outputs and generate lithiation-conditioned error plots

### 1) Prepare VASP inputs

Template-driven mode (uses provided `INCAR/KPOINTS/POTCAR`; applies composition-aware `MAGMOM` token substitution for `N_<Element>`):

```bash
hrw-electrode-vasp-workflow prepare-inputs \
  --structures-root results/publication/default_systems/electrode/LCO_mp-22526/structure_generation \
  --template-dir for_chat_gpt/VASP_input_templates_for_new_experiment \
  --output-dir results/electrode_vasp_workflow
```

Auto mode with `MPStaticSet` (no template directory):

```bash
hrw-electrode-vasp-workflow prepare-inputs \
  --structures-root results/publication/default_systems/electrode/LCO_mp-22526/structure_generation \
  --output-dir results/electrode_vasp_workflow \
  --potcar-spec
```

The command writes case folders under:

```text
<output_dir>/cases/<case_id>/
  POSCAR INCAR KPOINTS [POTCAR|POTCAR.spec] run_vasp.slurm run_manifest.json
```

Per-case Slurm scripts default to Perlmutter settings and can be overridden by CLI flags (`--perlmutter-account`, `--perlmutter-queue`, `--slurm-*`, `--vasp-module`, `--vasp-exe`).

### 2) Submit and monitor

Submit pending jobs only:

```bash
hrw-electrode-vasp-workflow submit --cases-root results/electrode_vasp_workflow
```

Resubmit fizzled jobs as well:

```bash
hrw-electrode-vasp-workflow submit --cases-root results/electrode_vasp_workflow --resubmit-fizzled
```

Check status summary and optionally write JSON:

```bash
hrw-electrode-vasp-workflow status \
  --cases-root results/electrode_vasp_workflow \
  --output-json results/electrode_vasp_workflow/status_summary.json
```

### 3) Compare UMA vs VASP and plot vs lithiation

```bash
hrw-electrode-vasp-workflow analyze-uma-vs-vasp \
  --cases-root results/electrode_vasp_workflow \
  --analysis-dir results/electrode_vasp_workflow/analysis \
  --model uma-s-1p2 \
  --task-name omat
```

Analysis outputs include:

```text
<analysis_dir>/
├── uma_vs_vasp_all_cases.csv
├── analysis_summary.json
├── uma_vs_vasp_delta_energy_per_atom_vs_lithiation.png
├── uma_vs_vasp_mean_force_vs_lithiation.png
├── uma_vs_vasp_rms_force_vs_lithiation.png
└── per_case/*.json
```

The lithiation plots show all points with transparency plus mean±std overlays at each lithiation percentage.

## UMA fine-tuning from `vasp_workflow` outputs (omat)

New command:

```bash
hrw-uma-finetune-vasp-workflow --help
```

Purpose:

- Convert completed VASP cases into labeled `extxyz` files.
- Build a reproducible `90/5/5` train/val/test split.
- Preserve diversity across temperatures and de-lithiation values parsed from case directory names (`T_<K>`, `lith_<pct>`).
- Run official FAIRChem UMA fine-tuning tooling for `omat`.
- Compare pre/post-fine-tuning MLFF quality against VASP labels.

Recommended references:

- https://fair-chem.github.io/
- https://fair-chem.github.io/fine-tuning/

Quick workflow:

```bash
# 1) Prepare split dataset
hrw-uma-finetune-vasp-workflow prepare-dataset \
  --cases-root results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/vasp_workflow \
  --output-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/dataset

# 2) Fine-tune UMA
hrw-uma-finetune-vasp-workflow run-finetune \
  --dataset-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/dataset \
  --output-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/finetune \
  --uma-task omat --regression-tasks ef --base-model uma-s-1p2

# 3) Pre/post analysis on test split
hrw-uma-finetune-vasp-workflow analyze-pre-post \
  --dataset-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/dataset \
  --fine-tuned-checkpoint <path_to_inference_ckpt.pt> \
  --output-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/analysis \
  --task-name omat --split test
```

Full details and end-to-end examples are in:

- `docs/uma_finetune_vasp_workflow.md`

## Electrode structure generation (scaffold)

Inputs accepted by the scaffold:

- Pristine structure path or Materials Project ID (`mp-...`)
- MPID cell mode toggle: `--conventional-unit-cell` / `--no-conventional-unit-cell` (default: conventional)
- Target ion (default `Li`, extensible to `Na`, etc.)
- Supercell size (default `[3, 3, 3]`)
- Output directory (default `./electrode_structures/`)
- Maximum final structures (default `600`)
- Pre-DIRECT pool size defaults to `max_structures * oversampling_factor` (e.g., `6000`)
- Minimum lithiation fraction (default `0.75`)
- Lithiation step for planned bins (default `0.05`)
- Maximum random ion-removal combinations per lithiation bin (default `200`)
- Rattle engine (`hiphive`, `uma`, `matgl`, or `all`; default `hiphive`)
  - Temporary behavior: MatGL rattling is currently deactivated; `all` runs hiPhive + UMA.
- For `hiphive`, rattle method (`mc`, `gaussian`, or `phonon`, default `mc`)
- Rattles per base structure (default `1`)
- Rattle amplitude at 300 K (default `0.01`, scales as $\sqrt{T/300}$)
- MC minimum distance (`d_min`, default `1.5` Å)
- MC iterations (`n_iter`, default `10`)
- Optional phonon-rattle mode (`rattle_method=phonon`) if `phonon_fc2_path` is provided
- Per-temperature and per-lithiation progress bars during rattling
- MLFF-MD options for UMA (`fairchem`) and MatGL with NVT/NPT controls
- UMA device defaults to GPU (`--uma-device cuda`) when not provided
- MD frame selection keeps a random fraction of sampled snapshots (`--md-frame-select-fraction`, default `0.10`)
- MD run length enforces at least 4× sampled snapshots before selection (`--md-min-step-multiplier`, default `4.0`)
- Total MLFF-MD step budget defaults to `60000` (`--md-total-steps-budget`) and is distributed across MD runs
- MD runs are resumable: completed base snapshots are cached under `<output_dir>/rattling_cache/` and reused after interruption
- MatGL model/backend controls (`--matgl-model-name`, `--matgl-backend auto|dgl|pyg`)
- MatGL code paths are retained but temporarily disabled at runtime pending robust shared environment compatibility with `fairchem-core>=2.15.0`.
- `md_execution=run` to execute immediately, or `--slurm-generate-only` to emit SLURM jobs
- Temperature strategy:
  - Fixed list default: `[393, 433, 473, 493]` K
  - Auto mode: 5 points between 250 K and 1.1 × melting temperature, forced inclusion of 300 K
- Optional pressure list (MPa) aligned with temperature entries; publication default is `[0.08, 0.46, 1.32, 2.02]` for NPT runs

Generation strategy:

- Generate delithiation combinations down to the minimum lithiation with capped, fast random sampling.
- Use one representative for 100% lithiated and single-vacancy states.
- Oversample pool by default to `10x` final target.
- Apply rattling via selected engine: hiPhive, UMA MD, MatGL MD, or all three.
- In `--rattle-engine all` mode, each active engine now generates a full pre-DIRECT pool (`max_structures * oversampling_factor`) instead of splitting that pool across engines.
- For `all` mode, three DIRECT candidates are evaluated: `hiphive`-only, `uma`-only, and `combined`; each produces a 600-structure training set (for default `max_structures=600`).
- The best option (highest mean DIRECT coverage score) is written to `<output_dir>/best_training_set/`.
- Apply `maml`-style DIRECT sampling to down-select robust training structures.
- Use `--skip-direct` if you need to emit the full pre-DIRECT pool.
- Preserve user-defined sampling ratios across lithiation bins and temperature bins.

DIRECT plotting outputs are written to:

```text
<output_dir>/direct_metrics/
├── explained_variance.png
├── pca_coverage_direct.png
├── pca_coverage_manual.png
└── coverage_scores.png
```

All-mode option-comparison analysis outputs (electrode and electrolyte) include:

```text
<output_dir>/direct_metrics/
├── source_contribution_by_option.png
├── selected_temperature_pressure_distribution.png
├── selected_lithiation_distribution.png                 # electrode
├── selected_li_concentration_distribution.png           # electrolyte
└── training_set_comparison.json
```

`training_set_comparison.json` now includes per-option source counts (`hiphive`, `uma`, `matgl/other`) for both pre-DIRECT pools and DIRECT-selected sets, plus DIRECT-selected distributions by temperature/pressure and composition key (lithiation for electrode, LiOH concentration for electrolyte).

In `all` mode, per-option metrics are written under:

```text
<output_dir>/direct_metrics/
├── option_hiphive/
├── option_uma/
├── option_combined/
└── training_set_comparison.json
```

Upfront generation overview (written before rattling starts):

```text
<output_dir>/generation_overview.json
```

This file includes:

- counts of delithiation candidates by unique missing-Li levels
- planned rattled structure counts per method/backend
- planned per-bin counts by temperature and lithiation fraction

Real-time MD progress/ETA outputs (updated during UMA/MatGL runs):

```text
<output_dir>/md_runtime_stats/
├── md_progress_uma.json
└── md_progress_matgl.json
```

These files track completed structures, current bin progress, effective generation rate, ETA to completion, and failure state (`status=failed`, `error_message`) if an MD backend exits with an exception.
For UMA with NPT, each bin also records `pressure_mpa` in `md_progress_uma.json`.
hiPhive and MD progress bars report base runs left for the current (temperature, lithiation) bin plus ETA from a moving average of recent base-run durations.

Per-run MD artifacts are saved under:

```text
<output_dir>/md_runs/
└── engine_<backend>/T_<temp>K/lith_<percent>pct/
  ├── base_<candidate_index>.extxyz
  └── base_<candidate_index>_properties.csv
```

Trajectory files are written in OVITO-readable EXTXYZ format.
The properties log includes step, temperature, pressure, volume, density, kinetic energy, potential energy, and total energy (UMA runs).

If `--slurm-generate-only` is used with MLFF-MD engines, scripts are written under:

```text
<output_dir>/slurm_jobs/
├── run_uma_T<temp>_lith_<percent>.slurm
├── run_matgl_T<temp>_lith_<percent>.slurm
├── ... (one script per MD temperature × lithiation bin by default)
├── run_uma_rattling.slurm (when `--slurm-combined-jobs` is passed)
├── run_matgl_rattling.slurm (when `--slurm-combined-jobs` is passed)
└── plot_direct_metrics.slurm
```

Default SLURM walltime is `1:00:00`.

Quick example (run directly on terminal, no SLURM files):

```bash
PYTHONPATH=src python -m hydrorelith.pipelines.electrode_structure_generation \
  --mpid mp-22526 \
  --target-ion Li \
  --rattle-engine all \
  --md-execution run \
  --max-structures 600 \
  --oversampling-factor 10 \
  --output-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation
```

Use `--skip-direct` if you want to keep the full pre-DIRECT pool instead of down-selecting.

Full publication command (electrode defaults for temperatures, pressures, structure counts, and MD budget):

```bash
PYTHONPATH=src hrw-electrode-generate \
  --mpid mp-22526 \
  --target-ion Li \
  --supercell 3 3 3 \
  --output-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation \
  --rattle-engine all \
  --uma-task-id omat \
  --uma-device cuda \
  --md-execution run \
  --md-ensemble npt
```

Full publication command (electrolyte defaults for templates, concentrations, temperatures, pressures, and MD budget):

```bash
hrw-electrolyte-generate \
  --output-dir results/publication/default_systems/electrolyte/LiOH_KOH_H2O/structure_generation \
  --rattle-engine all \
  --md-ensemble npt \
  --uma-device cuda
```

Output layout target:

```text
electrode_structures/
└── T_<temp>K/ or engine_<backend>/T_<temp>K/ (when `rattle_engine=all`)
  └── lith_<percent>pct/
    └── POSCAR (or .cif)
```

Backbone utility command (works before full generation internals exist):

```bash
hrw-electrode-generate --bootstrap-output-tree
```

## UMA TorchSim chemical-potential workflow

New command:

```bash
hrw-uma-torchsim-chem-potential --help
```

Purpose:

- Run TorchSim UMA MD (`NPT` rethermalization + `NVT` production) for electrode or electrolyte campaigns.
- Accept mixed structure inputs from one directory (`POSCAR*`, `.data`/`.lammps`/`.lmp`, `.cif`, `.bgf`, plus `.xyz`/`.extxyz`/`.vasp`/`.pdb`).
- Generate and run py2pt jobs for each replica using TorchSim trajectories and logs.
- Produce publication-style summary outputs including:
  - `master_thermo_evolution_all_runs.png`
  - `master_final_stats_vs_concentration.png` / `master_final_stats_vs_lithiation.png`
  - `li_chemical_potential_vs_concentration.png` / `li_chemical_potential_vs_lithiation.png`
- Optionally generate per-condition Slurm scripts for MD plus postprocessing scripts for py2pt and plots.

See full usage in:

- `docs/uma_torchsim_chem_potential_workflow.md`

Install fork-pinned dependencies for this experiment with:

```bash
pip install ".[uma-forks]"
```

## Quick start

```bash
pip install -e .
hrw --help
hrw-electrode-generate --help
```

`maml` is required for DIRECT sampling in both electrode and electrolyte pipelines and is installed automatically by `pip install -e .`.

Recommended clean install path for UMA + MatGL:

```bash
python -m pip install "fairchem-core==2.15.0" matgl maml
# Install a DGL wheel matching your active torch/CUDA stack for CHGNet/QET models.
# For fairchem-core 2.15.0, torch~=2.8.0 is expected.
# Follow MatGL + DGL install docs to pick the correct wheel URL for your platform.
python -m pip install "dgl>=2.2.0,<2.5"
python -m pip install -e .
```

Expected UMA package line after install: `fairchem-core 2.15.0`.

Dependency compatibility note:

- This project now aligns with `fairchem-core==2.15.0` and MatGL (`numpy>=2.0,<2.4`, `torch~=2.8.0`).
- Packages that still require `numpy<2` (for example, some older diffusion-analysis packages) must be installed in a separate environment.

For MPID-based structure loading, set `MP_API_KEY` in your environment.

MatGL/UMA model note:

MatGL and UMA recommended defaults:

- Electrode generation: MatGL `CHGNet-MatPES-PBE-2025.2.10-2.7M-PES` and UMA task `omat`.
- Electrolyte generation: MatGL `QET-MatQ-PES` and UMA task `omol`.

MatGL installation note (important for CHGNet/QET):

- CHGNet and QET are DGL-backed in MatGL, so install DGL in your environment and use `--matgl-backend dgl`.
- MatGL v2 defaults to PyG; DGL-backed models require backend selection via `MATGL_BACKEND=DGL` or `matgl.set_backend("DGL")` (the pipeline sets this automatically from `--matgl-backend`).
- Follow MatGL's installation guidance to choose a DGL wheel that matches your Torch/CUDA stack.

Run only the implemented next stage (pristine loading + delithiation generation):

```bash
hrw-electrode-generate --mpid mp-22526 --stop-after-delithiation
```

## Next implementation milestones

1. Implement DIRECT sampling adapter from descriptors.
2. Add serialization and job manifests for DFT workflows.
3. Expand to electrolyte and coupled thermodynamic optimization workflows.
