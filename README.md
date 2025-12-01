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
VLLM_COMMIT=v0.11.2
PPLX_KERNELS_COMMIT=12cecfda252e4e646417ac263d96e994d476ee5d
DEEPGEMM_COMMIT=38f8ef73a48a42b1a04e0fa839c2341540de26a6
DEEPEP_COMMIT=27e8e661857499068275dbaa09e4c15d67d51f81
PPLX_GARDEN_COMMIT=f62aac558ef937340d17091a52011631d0c65147
UCCL_COMMIT=e4d2b2e00aed5c7b6b7d5f908a579e49ea538048
```
## Build the container image
Don't build it on the slurm head node, it may kill it. Use `srun` to build it on a compute node.
```bash
srun bash -lc "docker build --progress=plain -f ./vllm-ep.Dockerfile \
  --build-arg=EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION} \
  --build-arg=NCCL_VERSION=${NCCL_VERSION} \
  --build-arg=NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION} \
  --build-arg=NVSHMEM_COMMIT=${NVSHMEM_COMMIT} \
  --build-arg=VLLM_COMMIT=${VLLM_COMMIT} \
  --build-arg=TORCH_VERSION=${TORCH_VERSION} \
  --build-arg=PPLX_KERNELS_COMMIT=${PPLX_KERNELS_COMMIT} \
  --build-arg=DEEPGEMM_COMMIT=${DEEPGEMM_COMMIT} \
  --build-arg=DEEPEP_COMMIT=${DEEPEP_COMMIT} \
  --build-arg=PPLX_GARDEN_COMMIT=${PPLX_GARDEN_COMMIT} \
  --build-arg=UCCL_COMMIT=${UCCL_COMMIT} \
  -t vllm-ep:latest . && \
  enroot import -o ./vllm-ep.sqsh dockerd://vllm-ep:latest"
```

## NCCL Test
```bash
sbatch nccl-test.sbatch
```

## NVSHMEM Test
```bash
sbatch nvshmem-test.sbatch
```

## DeepEP Tests
```bash
sbatch deepep-test_intranode.sbatch
sbatch deepep-test_internode.sbatch
sbatch deepep-test_low_latency.sbatch
```

## PPLX Kernels Test
```bash
sbatch pplx-kernels-test.sbatch
```

## PPLX Garden Test
```bash
sbatch pplx-garden-test.sbatch
```

## UCCL-EP Tests
```bash
sbatch uccl-ep-test_intranode.sbatch
sbatch uccl-ep-test_internode.sbatch
sbatch uccl-ep-test_low_latency.sbatch
```

## DeepEP version
```bash
srun --container-image ./vllm-ep.sqsh \
  bash -c 'PYTHONPATH=$(echo /DeepEP/install/lib/python*/site-packages):$PYTHONPATH pip list | grep deep_ep'
```
expected output:
```
deep_ep                           1.2.1+27e8e66
```
## DeepEP UCCL-EP version
```bash
srun --container-image ./vllm-ep.sqsh \
  bash -c 'PYTHONPATH=$(echo /uccl/install/lib/python*/site-packages):$PYTHONPATH pip list | grep deep_ep'
```
expected output:
```
deep_ep                           0.1.0
```

## vLLM check for 
```bash
srun --container-image ./vllm-ep.sqsh \
  bash -c 'PYTHONPATH=$(echo /DeepEP/install/lib/python*/site-packages):$PYTHONPATH \
           python -c "from vllm.utils.import_utils import has_deep_ep, has_deep_gemm, has_pplx; \
                      print(f\"{has_deep_ep()=}\n{has_deep_gemm()=}\n{has_pplx()=}\")"'
```
or
```bash
srun --container-image ./vllm-ep.sqsh \
  bash -c 'PYTHONPATH=$(echo /uccl/install/lib/python*/site-packages):$PYTHONPATH \
           python -c "from vllm.utils.import_utils import has_deep_ep, has_deep_gemm, has_pplx; \
                      print(f\"{has_deep_ep()=}\n{has_deep_gemm()=}\n{has_pplx()=}\")"'
```
expected output:
```
has_deep_ep()=True
has_deep_gemm()=True
has_pplx()=True
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
    vllm-ep:latest \
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
