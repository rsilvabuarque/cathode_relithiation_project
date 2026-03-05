#!/bin/bash
#SBATCH -J uma_el_md
#SBATCH -A <account>
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Optional 4-GPU single-node configuration (one task per GPU):
# SBATCH --gpus-per-node=4
# SBATCH --ntasks-per-node=4

# Optional array mode for one condition per task:
# SBATCH --array=0-9

set -euo pipefail

# module load python/3.12
# source /path/to/venv/bin/activate

hrw-uma-torchsim-screen \
  --phase electrolyte \
  --stage md \
  --electrolyte-manifest electrolyte_manifest.csv \
  --output-dir runs/uma_torchsim \
  --device cuda
