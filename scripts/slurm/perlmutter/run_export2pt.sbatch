#!/bin/bash
#SBATCH -J uma_export2pt
#SBATCH -A <account>
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

set -euo pipefail

# module load python/3.12
# source /path/to/venv/bin/activate

hrw-uma-torchsim-screen \
  --phase both \
  --stage export-2pt \
  --output-dir runs/uma_torchsim
