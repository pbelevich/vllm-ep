# vLLM Expert Parallel Deployment

https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment.html

## Set environment variables
```bash
GDRCOPY_VERSION=v2.5.1
EFA_INSTALLER_VERSION=1.45.0
NCCL_VERSION=v2.28.9-1
NCCL_TESTS_VERSION=v2.17.6
NVSHMEM_COMMIT=v3.4.5-0
TORCH_VERSION=2.9.1
VLLM_VERSION=0.11.2
PPLX_KERNELS_COMMIT=12cecfda252e4e646417ac263d96e994d476ee5d
DEEPGEMM_COMMIT=c9f8b34dcdacc20aa746b786f983492c51072870
DEEPEP_COMMIT=27e8e661857499068275dbaa09e4c15d67d51f81
PPLX_GARDEN_COMMIT=b3c4b59bab08bfb307ec7809b88a49ba8d53d633
UCCL_EP_COMMIT=d6f3089ca6ca0a4d353c446820d146c212c68630
VLLM_EP_CONTAINER_IMAGE_NAME_TAG="vllm-ep:latest"
```
## Build the container image
Don't build it on the slurm head node, it may kill it. Use `srun` to build it on a compute node.
```bash
srun bash -lc "docker build --progress=plain -f ./vllm-ep.Dockerfile \
  --build-arg=EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION} \
  --build-arg=NCCL_VERSION=${NCCL_VERSION} \
  --build-arg=NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION} \
  --build-arg=NVSHMEM_COMMIT=${NVSHMEM_COMMIT} \
  --build-arg=VLLM_VERSION=${VLLM_VERSION} \
  --build-arg=TORCH_VERSION=${TORCH_VERSION} \
  --build-arg=PPLX_KERNELS_COMMIT=${PPLX_KERNELS_COMMIT} \
  --build-arg=DEEPGEMM_COMMIT=${DEEPGEMM_COMMIT} \
  --build-arg=DEEPEP_COMMIT=${DEEPEP_COMMIT} \
  --build-arg=PPLX_GARDEN_COMMIT=${PPLX_GARDEN_COMMIT} \
  --build-arg=UCCL_EP_COMMIT=${UCCL_EP_COMMIT} \
  -t ${VLLM_EP_CONTAINER_IMAGE_NAME_TAG} . && \
  enroot import -o ./vllm-ep.sqsh dockerd://${VLLM_EP_CONTAINER_IMAGE_NAME_TAG}"
```

## Run the container on a single 8 GPU node
```bash
docker run --runtime nvidia --gpus all \
    -v "$HF_HOME":/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -e VLLM_ALL2ALL_BACKEND=pplx \
    -e VLLM_USE_DEEP_GEMM=1 \
    -p 8000:8000 \
    --ipc=host \
    ${VLLM_EP_CONTAINER_IMAGE_NAME_TAG} \
    vllm serve deepseek-ai/DeepSeek-R1-0528 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
```
## Expected logs:

FlashInfer Backend:
```
[topk_topp_sampler.py:50] Using FlashInfer for top-p & top-k sampling.
```

DeepGEMM Backend:
```
[fp8.py:512] Using DeepGemm kernels for Fp8MoEMethod.
```

PPLX Backend:
```
[cuda_communicator.py:81] Using PPLX all2all manager.
```
otherwise:
```
[cuda_communicator.py:77] Using naive all2all manager.
```
6. Benchmark
```
vllm bench serve \
    --model deepseek-ai/DeepSeek-R1-0528 \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 10000 \
    --ignore-eos
```
