#!/bin/bash
#SBATCH -J uma_analyze
#SBATCH -A <account>
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

set -euo pipefail

# module load python/3.12
# source /path/to/venv/bin/activate

hrw-uma-torchsim-screen \
  --phase both \
  --stage analyze,regress,plots \
  --output-dir runs/uma_torchsim \
  --electrode-reference-pristine path/to/pristine_reference.cif
