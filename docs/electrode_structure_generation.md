# Electrode Structure Generation (Backbone)

## Objective

Create a robust and diverse training set for electrode MLFF models by combining:

1. Controlled ion-removal configurations
2. Temperature-dependent structural rattling
3. DIRECT subset selection to enforce diversity and robustness

Rattling can be performed with three alternatives:

- hiPhive (gaussian / MC / phonon-rattle)
- UMA MLFF molecular dynamics (`fairchem-core>=2.15.0`, `omat` default task)
- MatGL MLFF molecular dynamics (`CHGNet-MatPES-PBE-2025.2.10-2.7M-PES` default model)

Temporary runtime note:

- MatGL rattling is currently deactivated in the pipeline to preserve stable operation with `fairchem-core>=2.15.0` in a single environment.
- `--rattle-engine all` currently executes hiPhive + UMA only.
- MatGL code remains in the codebase and can be re-enabled once environment compatibility is finalized.

## Current implementation defaults

The list below reflects the active defaults in `SamplingConfig` and pipeline CLI wiring.

- Source and output:
  - target ion: `Li`
  - supercell: `(3, 3, 3)`
  - output format: `poscar`
- Core sampling:
  - final structures: `600`
  - oversampling factor: `10`
  - minimum lithiation fraction: `0.75`
  - lithiation step: `0.05`
  - max removal combinations per fraction: `200`
- hiPhive rattling:
  - rattle method: `mc`
  - rattles per base structure: `1`
  - rattle std at 300 K: `0.01`
  - `d_min`: `1.5`
  - `n_iter`: `10`
  - max base structures per temperature/lithiation bin: `25`
  - phonon options: `phonon_fc2_path=None`, `phonon_qm_statistics=False`, `phonon_imag_freq_factor=1.0`
- MLFF-MD execution:
  - rattle engine default: `hiphive`
  - md execution mode: `run`
  - ensemble: `nvt`
  - for UMA with `ensemble=npt`, pressure is applied per temperature from `temperature.pressures_mpa` (defaults to `0.1 MPa` if omitted)
  - timestep: `1.0 fs`
  - steps: `500`
  - sample interval: `10`
  - Langevin friction: `0.001 / fs`
  - UMA model/device/task: `uma-s-1p1`, `cuda`, `omat`
  - MatGL model/backend: `CHGNet-MatPES-PBE-2025.2.10-2.7M-PES`, `dgl`
- DIRECT:
  - `direct_threshold_init`: `0.05`
  - DIRECT metric plots enabled by default
- Temperature defaults:
  - fixed temperatures: `(250, 300, 600, 900, 1200) K`
  - auto mode: `n_points=5`, `include_300k=True`, `melting_temperature_margin=1.10`

Additional runtime outputs enabled by default:

- pre-run generation plan: `<output_dir>/generation_overview.json`
- live MD runtime stats per backend:
  - `<output_dir>/md_runtime_stats/md_progress_uma.json`
  - `<output_dir>/md_runtime_stats/md_progress_matgl.json`

## Defaults for hydrothermal relithiation

These defaults prioritize realistic near-operating and near-equilibrium states while retaining high-temperature coverage:

- Final dataset size: `600`
- Oversampling factor before DIRECT: `10`
- Minimum lithiation: `75%`
- Temperatures: `[250, 300, 600, 900, 1200]` K
- Temperature retention weights:
  - 250 K: 0.15
  - 300 K: 0.35
  - 600 K: 0.25
  - 900 K: 0.15
  - 1200 K: 0.10
- Lithiation retention weights:
  - 1.00–0.95: 0.30
  - 0.95–0.90: 0.25
  - 0.90–0.85: 0.20
  - 0.85–0.80: 0.15
  - 0.80–0.75: 0.10
- Generation composition (pre-DIRECT):
  - Distinct ion-removal combinations: 20%
  - Rattled variants from those combinations: 80%

Rationale:

- Hydrothermal processes are often operated near moderate temperatures where kinetics and aqueous chemistry are active, so 300–600 K receives higher default weight.
- Near-lithiated states are strongly relevant for relithiation pathways and should be sampled more densely.
- High-temperature tails are retained to improve robustness and reduce extrapolation risk.

## Module responsibilities

- Parse and validate user input.
- Resolve temperature list from fixed/auto strategy.
- Build delithiation targets and removal combinations.
- Generate rattled candidates (phonon-rattle preferred for hiPhive).
- Generate MD-rattled candidates with UMA/MatGL as alternatives.
- Apply DIRECT-style sampler (maml interface) under retention constraints.
- Emit DIRECT quality plots (PCA coverage and feature coverage scores).
- Export structures into temperature/lithiation directory tree.
