#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=vllm-ep-multi-node-1rack
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=1 
#SBATCH --output logs_multi_node_1rack/%x_%j.out
#SBATCH --error logs_multi_node_1rack/%x_%j.err
#SBATCH --exclusive=topo
#SBATCH --wait-all-nodes=1

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

mkdir -p logs_multi_node_1rack

###########################
###### User Variables #####
###########################

# default variables for Enroot
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/ubuntu/vLLM-testing/vllm-ep.sqsh}"
: "${HF_HOME:=/fsx/ubuntu/.cache/huggingface}"

## Set libfabric flags to use EFA
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1 
export FI_EFA_FORK_SAFE=1

## Set vLLM Environment Variables
export VLLM_ALL2ALL_BACKEND=pplx
export VLLM_USE_DEEP_GEMM=1
export HF_TOKEN=$HF_TOKEN

## Set this flag for debugging EFA
export FI_LOG_LEVEL=warn

## NCCL Environment variables
export NCCL_DEBUG=INFO

# Model
export MODEL="deepseek-ai/DeepSeek-V3-0324"
export TP=1
export DP_TOTAL=72  # Total DP size across all nodes
export DP_NODE=4    # Local DP size per node


declare -a ARGS=(
    --container-image $IMAGE
    --container-mount-home
    --container-mounts /fsx/ubuntu/vLLM-testing:/workspace
    --container-mounts /dev/urandom:/dev/urandom
    --container-env HF_TOKEN
    --container-env VLLM_ALL2ALL_BACKEND
    --container-env VLLM_USE_DEEP_GEMM
    --container-env MODEL
    --container-env TP
    --container-env DP_TOTAL
    --container-env DP_NODE
)

# Get primary node IP
PRIMARY_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
PRIMARY_IP=$(srun --nodes=1 --ntasks=1 -w $PRIMARY_NODE hostname -I | awk '{print $1}')
echo "Primary node: $PRIMARY_NODE ($PRIMARY_IP)"
echo "Total nodes: 18 (72 GPUs)"

# Start primary node with API Servers and run benchmark once server is ready (assume once server on node 1 is ready, all others have indicated ready)
srun --nodes=1 --ntasks=1 --nodelist $PRIMARY_NODE "${ARGS[@]}" bash -c "
    cd /workspace

    # ISL/OSL combinations to test
    declare -a ISL_OSL_PAIRS=(
        \"128 128\"
        \"128 2048\"
        \"128 4096\"
        \"500 2000\"
        \"1024 2048\"
        \"2048 128\"
        \"2048 2048\"
        \"5000 500\"
        \"20000 2000\"
    )

    # Start server
    vllm serve \$MODEL \
        --trust-remote-code \
        --tensor-parallel-size \$TP \
        --enable-expert-parallel \
        --data-parallel-size \$DP_TOTAL \
        --data-parallel-size-local \$DP_NODE \
        --data-parallel-address $PRIMARY_IP \
        --data-parallel-rpc-port 13345 \
        --api-server-count=72 \
        --host 0.0.0.0 \
        --port 8000 &

    # Wait for primary server and give time for all secondary nodes to connect
    for i in {1..600}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "Primary server responding, waiting for all 18 nodes to connect..."
            sleep 240  # Give time for all 17 secondary nodes to join
            echo 'All servers should be ready'
            break
        fi
        sleep 10
    done

    # Run benchmarks
    for pair in \"\${ISL_OSL_PAIRS[@]}\"; do
        read -r isl osl <<< \"\$pair\"
        echo \"Testing ISL=\$isl, OSL=\$osl\"

        vllm bench serve \
            --model \$MODEL \
            --dataset-name random \
            --random-input-len \$isl \
            --random-output-len \$osl \
            --num-prompts 1000 \
            --ignore-eos \
            --trust-remote-code \
            > logs_multi_node_1rack/results_\${isl}_\${osl}_${SLURM_JOB_ID}.txt 2>&1
    done
" &

# Start remaining nodes as headless workers
WORKER_NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2))

for i in "${!WORKER_NODES[@]}"; do
    NODE=${WORKER_NODES[$i]}
    # NOTE: This is specific for GB200 (u-p6e-gb200x72)
    echo "Starting worker node $((i+1))/$((18-1)) ($NODE)"
    START_RANK=$(( (i+1) * 4 ))

    srun --nodes=1 --ntasks=1 --nodelist $NODE "${ARGS[@]}" --container-env START_RANK=$START_RANK bash -c "
        cd /workspace
        vllm serve \$MODEL \
            --trust-remote-code \
            --tensor-parallel-size \$TP \
            --enable-expert-parallel \
            --data-parallel-size \$DP_TOTAL \
            --data-parallel-size-local \$DP_NODE \
            --data-parallel-start-rank \$START_RANK \
            --data-parallel-address $PRIMARY_IP \
            --data-parallel-rpc-port 13345 \
            --headless
    " &
done

wait
