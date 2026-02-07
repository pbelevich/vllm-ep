#!/bin/bash
#SBATCH --job-name=vllm-qwen3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -x

export MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507

export MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export PORT=13579

export NUM_GPUS=8

export FI_PROVIDER=efa
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export PATH=/opt/nvshmem/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvshmem/lib:$LD_LIBRARY_PATH

export VLLM_USE_DEEP_GEMM=1

srun -l \
    --mpi=pmix --cpu-bind=none \
    --container-image ./vllm-ep.sqsh \
    --container-mounts=/local_scratch:/local_scratch \
    bash -c 'set -x; 
    mkdir -p /local_scratch/.cache/huggingface;
    export HF_HOME=/local_scratch/.cache/huggingface;
    hf download ${MODEL_NAME};
        vllm serve ${MODEL_NAME} \
            --port 8000 \
            --trust-remote-code \
            --data-parallel-size $((NUM_GPUS * SLURM_NTASKS)) \
            --data-parallel-size-local ${NUM_GPUS} \
            --tensor-parallel-size 1 \
            --api-server-count ${NUM_GPUS} \
            --enable-expert-parallel
    '
