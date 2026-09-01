#!/bin/bash
#SBATCH --job-name=itu-auction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:05:00

set -euo pipefail

export ITU_PYTHON="${ITU_PYTHON:-python}"
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$("$ITU_PYTHON" -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
export WORLD_SIZE="$SLURM_NTASKS"
export OMP_NUM_THREADS=1
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

cd "$SLURM_SUBMIT_DIR"
srun bash -c 'export RANK="$SLURM_PROCID"; export LOCAL_RANK="$SLURM_LOCALID"; exec "$ITU_PYTHON" experiments/demo_distributed.py'
