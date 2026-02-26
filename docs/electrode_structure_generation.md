# Electrode Structure Generation (Backbone)

## Objective

Create a robust and diverse training set for electrode MLFF models by combining:

1. Controlled ion-removal configurations
2. Temperature-dependent structural rattling
3. DIRECT subset selection to enforce diversity and robustness

Rattling can be performed with three alternatives:

- hiPhive (gaussian / MC / phonon-rattle)
- UMA MLFF molecular dynamics (`fairchem`, `omat` default task)
- MatGL MLFF molecular dynamics (`CHGNet-MatPES-PBE-2025.2.10-2.7M-PES` default model)

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
