# Hydrothermal Cathode Relithiation Workflows

Repository backbone for a computational–experimental collaboration on hydrothermal relithiation. The project is designed to be material-agnostic while shipping with a default example pair:

- Electrode: LCO (LiCoO2)
- Electrolyte: LiOH/NaOH + H2O

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
│           └── electrode_structure_generation.py
└── for_chat_gpt/
    └── user-provided context files only
```

## Electrode structure generation (scaffold)

Inputs accepted by the scaffold:

- Pristine structure path or Materials Project ID (`mp-...`)
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
- MatGL model/backend controls (`--matgl-model-name`, `--matgl-backend auto|dgl|pyg`)
- MatGL code paths are retained but temporarily disabled at runtime pending robust shared environment compatibility with `fairchem-core>=2.15.0`.
- `md_execution=run` to execute immediately, or `--slurm-generate-only` to emit SLURM jobs
- Temperature strategy:
  - Fixed list default: `[250, 300, 600, 900, 1200]` K
  - Auto mode: 5 points between 250 K and 1.1 × melting temperature, forced inclusion of 300 K
- Optional pressure list (MPa) aligned with temperature entries; used as target external pressure for UMA when `--md-ensemble npt`

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

## Quick start

```bash
pip install -e .
hrw --help
hrw-electrode-generate --help
```

Recommended clean install path for UMA + MatGL:

```bash
python -m pip install "fairchem-core==2.15.0" matgl
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
