#!/bin/bash

#SBATCH --job-name=intranode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

export FI_PROVIDER=efa

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export WORLD_SIZE=$((${SLURM_NNODES} * 8))
export WORLD_LOCAL_SIZE=8

export PYTHONPATH=/pplx-kernels:$PYTHONPATH

mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

srun --mpi=pmix --cpu-bind=none --container-image ./vllm-ep.sqsh bash -c "RANK=\${SLURM_PROCID} NODE_RANK=\${SLURM_NODEID} python3 -m tests.bench_all_to_all"
