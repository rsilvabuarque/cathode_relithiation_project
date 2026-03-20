# UMA fine-tuning from `vasp_workflow` outputs

This workflow converts completed VASP cases into a FAIRChem-compatible UMA fine-tuning dataset, runs fine-tuning for the `omat` task, and compares MLFF quality before/after fine-tuning.

Command entrypoint:

```bash
hrw-uma-finetune-vasp-workflow --help
```

Implemented in:

- `src/hydrorelith/pipelines/uma_finetune_vasp_workflow.py`

## Required FAIRChem install (official fine-tuning setup)

Per the FAIRChem fine-tuning docs, use a source checkout with the `dev` extras:

```bash
git clone https://github.com/facebookresearch/fairchem.git
pip install -e fairchem/packages/fairchem-core[dev]
```

Reference:

- https://fair-chem.github.io/fine-tuning/

If your shell defines `fairchem` as an alias, use the binary path directly for training launches, e.g.:

```bash
$(dirname $(which python))/fairchem -c <generated_yaml>
```

## Directory assumptions

Input can be either:

- a workflow root containing `cases/` (e.g., `.../vasp_workflow/`), or
- the `cases/` directory itself.

Each case directory is expected to look like:

```text
cases/T_433K__lith_87.50pct__POSCAR_000194/
  OUTCAR
  vasprun.xml
  POSCAR
  ...
```

Metadata extraction from directory names:

- Temperature: `T_<value>K` (e.g., `T_433K`)
- Lithiation: `lith_<value>pct` (e.g., `lith_87.50pct`)
- Delithiation used for stratification and reporting is computed as:
  - `delithiation_pct = 100 - lithiation_pct`

## 1) Prepare split dataset (90/5/5)

This step:

- reads final VASP-labeled structures (`vasprun.xml` preferred, `OUTCAR` fallback),
- writes single-frame labeled `extxyz` files,
- creates train/val/test splits,
- enforces split diversity for both temperature and delithiation,
- writes CSV manifests and split summary JSON.

```bash
hrw-uma-finetune-vasp-workflow prepare-dataset \
  --cases-root results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/vasp_workflow \
  --output-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/dataset \
  --train-frac 0.90 \
  --val-frac 0.05 \
  --test-frac 0.05 \
  --seed 7
```

Outputs:

```text
<dataset_dir>/
  dataset_summary.json
  split/{train.csv,val.csv,test.csv,all.csv}
  extxyz/{train/,val/,test/,raw/}
```

## 2) Run UMA fine-tuning (`omat`)

This step follows FAIRChem's recommended two-stage flow:

1. `python -m fairchem.core.scripts.create_uma_finetune_dataset`
2. `fairchem -c <generated_yaml> ...`

Reference pages used:

- https://fair-chem.github.io/
- https://fair-chem.github.io/fine-tuning/

Recommended defaults for this project scope:

- `--uma-task omat` (electrode solids)
- `--regression-tasks ef` (fit energies + forces)
- `--base-model uma-s-1p2`
- moderate training budget (`epochs=8`, `batch_size=4`, `lr=2e-4`) to reduce overfitting risk on limited in-domain VASP sets.

```bash
hrw-uma-finetune-vasp-workflow run-finetune \
  --dataset-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/dataset \
  --output-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/finetune \
  --uma-task omat \
  --regression-tasks ef \
  --base-model uma-s-1p2 \
  --epochs 8 \
  --batch-size 4 \
  --lr 2e-4 \
  --device auto
```

Dry-run mode (prints commands only):

```bash
hrw-uma-finetune-vasp-workflow run-finetune \
  --dataset-dir <dataset_dir> \
  --output-dir <finetune_dir> \
  --dry-run
```

Outputs:

- `<finetune_dir>/uma_sm_finetune_template.yaml`
- `<finetune_dir>/finetune_run_summary.json`
- FAIRChem run artifacts in `job.run_dir` (including `checkpoints/.../inference_ckpt.pt`)

## 3) Analyze pre vs post fine-tuning

This evaluates the baseline pretrained UMA model and the fine-tuned checkpoint against VASP labels on the selected split (default `test`).

```bash
hrw-uma-finetune-vasp-workflow analyze-pre-post \
  --dataset-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/dataset \
  --fine-tuned-checkpoint results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/finetune/runs/<run_id>/checkpoints/final/inference_ckpt.pt \
  --output-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow/analysis \
  --base-model uma-s-1p2 \
  --task-name omat \
  --split test
```

Key metrics reported for both pre/post models:

- `energy_mae_eV`
- `energy_per_atom_mae_eV`
- `mean_force_difference_mae_eV_per_A`
- `rms_force_difference_mae_eV_per_A`
- `max_force_difference_mae_eV_per_A`

Outputs:

```text
<analysis_dir>/
  pre_post_metrics.csv
  pre_post_summary.json
  pre_post_energy_per_atom_vs_delithiation.png
  pre_post_mean_force_vs_delithiation.png
  pre_post_rms_force_vs_delithiation.png
```

## One-command end-to-end run

```bash
hrw-uma-finetune-vasp-workflow run-all \
  --cases-root results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/vasp_workflow \
  --work-dir results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/uma_finetune_workflow \
  --uma-task omat \
  --base-model uma-s-1p2 \
  --epochs 8 \
  --batch-size 4 \
  --lr 2e-4
```

Use `--dry-run` on `run-all` to validate generated commands and paths before launching training.
