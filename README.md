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

Current executable stage includes pristine loading (MPID/local), delithiation candidate generation, and hiPhive-based temperature-dependent rattling.

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
- Rattle method (`mc` or `gaussian`, default `mc`)
- Rattles per base structure (default `1`)
- Rattle amplitude at 300 K (default `0.01`, scales as $\sqrt{T/300}$)
- MC minimum distance (`d_min`, default `1.5` Å)
- MC iterations (`n_iter`, default `10`)
- Optional phonon-rattle mode (`rattle_method=phonon`) if `phonon_fc2_path` is provided
- Per-temperature and per-lithiation progress bars during rattling
- Temperature strategy:
  - Fixed list default: `[250, 300, 600, 900, 1200]` K
  - Auto mode: 5 points between 250 K and 1.1 × melting temperature, forced inclusion of 300 K
- Optional pressure list (MPa) aligned with temperature entries for condition metadata tracking

Generation strategy:

- Generate delithiation combinations down to the minimum lithiation with capped, fast random sampling.
- Use one representative for 100% lithiated and single-vacancy states.
- Oversample pool by default to `10x` final target.
- Apply hiPhive rattling (phonon-rattle preferred where available).
- Apply DIRECT-style descriptor-based diversity sampling to down-select robust training structures.
- Use `--skip-direct` if you need to emit the full pre-DIRECT pool.
- Preserve user-defined sampling ratios across lithiation bins and temperature bins.

Output layout target:

```text
electrode_structures/
└── T_<temp>K/
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

For MPID-based structure loading, set `MP_API_KEY` in your environment.

Run only the implemented next stage (pristine loading + delithiation generation):

```bash
hrw-electrode-generate --mpid mp-22526 --stop-after-delithiation
```

## Next implementation milestones

1. Implement DIRECT sampling adapter from descriptors.
2. Add serialization and job manifests for DFT workflows.
3. Expand to electrolyte and coupled thermodynamic optimization workflows.
