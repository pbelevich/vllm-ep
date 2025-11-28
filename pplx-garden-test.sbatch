#!/bin/bash

#SBATCH --job-name=intranode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

export FI_PROVIDER=efa

mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export NUM_NODES=${SLURM_NNODES}
export NUM_GPUS=8

export PYTHONPATH=/pplx-garden:$PYTHONPATH

srun -l \
    --container-image ./vllm-ep.sqsh \
    --mpi=pmix --cpu-bind=none \
    bash -c 'python3 -X faulthandler -m benchmarks.bench_all_to_all \
        --world-size $((NUM_NODES * NUM_GPUS)) \
        --nets-per-gpu 2 \
        --init-method=tcp://${MASTER_ADDR}:${MASTER_PORT} \
        --node-rank=${SLURM_NODEID} \
        --nvlink=8 \
        --max-num-tokens 128'
