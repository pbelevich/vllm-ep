#!/bin/bash

#SBATCH --job-name=intranode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

export FI_PROVIDER=efa

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export SLURM_GPUS_PER_NODE=8

mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

srun -l \
    --container-image ./vllm-ep.sqsh \
        torchrun \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --nproc_per_node=$SLURM_GPUS_PER_NODE \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$(hostname) \
            /uccl/ep/bench/test_internode.py \
                --num-tokens=4096 \
                --hidden=7168 \
                --num-topk=8 \
                --num-experts=288 \
                --test-ll-compatibility
