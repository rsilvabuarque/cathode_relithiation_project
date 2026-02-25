# Architecture Overview

## Design principles

- Material-agnostic core APIs; chemistry-specific defaults in config.
- Clear phase boundaries: generation, DFT, training, MD, 2PT, analysis.
- Reproducibility-first: explicit seeds, immutable manifests, deterministic splits.
- Tool-agnostic wrappers for trainer and engine extensibility.

## Planned workflow phases

1. Structure generation
2. DFT execution and parsing
3. MLFF training and hyperparameter optimization
4. MD equilibration in LAMMPS
5. 2PT chemical potential extraction
6. Coupled analysis with experimental relithiation rates

## Package map

- `hydrorelith.config`: schema and default policies.
- `hydrorelith.pipelines`: phase orchestration.
- Future:
  - `hydrorelith.dft`
  - `hydrorelith.training`
  - `hydrorelith.md`
  - `hydrorelith.twopt`
  - `hydrorelith.analysis`
