#!/usr/bin/env bash
#SBATCH --account=gts-skaramcheti3
#SBATCH --partition=gpu-v100,gpu-a100,gpu-h100,gpu-h200,gpu-l40s,gpu-rtx6000,gpu-rtxpro-blackwell
#SBATCH --job-name=aspire-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:INT@60

set -euo pipefail
cd /storage/project/r-skaramcheti3-0/ychen3868/workplace/aspire/aspire_so100
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec uv run --no-sync python -u scripts/learn_inventory_repair.py \
    --root outputs/inventory_repair_01 --model gpt-6-astra --rounds 4 --max-reported-tokens 80000
