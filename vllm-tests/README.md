To test vLLM on GB200, we will use the instructions in [Expert Parallel Deployment (vLLM)](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html#single-node-deployment).
We will use the image defined [here](https://github.com/pbelevich/vllm-ep/blob/main/vllm-ep.Dockerfile).

## Note
```
EP=TP*DP
```

## Communication Backends for EP
vLLM supports three communication backends for EP:
<img width="764" height="274" alt="image" src="https://github.com/user-attachments/assets/7dc0a088-ce99-4ba0-96ff-1fba39bf6f02" />

## Tests
For this section, we will benchmark:
1. pplx backend on a single node (1 tray)
    1. Model: deepseek-ai/deepseek-moe-16b-base
2. pplx backend on multiple nodes (1 rack)
    1. Model: deepseek-ai/DeepSeek-V3-0324
3. pplx backend on multiple nodes (2 racks)
    1. Model: deepseek-ai/DeepSeek-V3-0324
4. deepep_low_latency  backend on multiple nodes (1 rack)
    1. Model: deepseek-ai/DeepSeek-V3-0324

We will not benchmark with the Expert Parallel load Balancer (EPLB). We will also not test Disaggregated Serving for this run. 

## Test 1

To run interactive test:
```bash
docker pull 159553542841.dkr.ecr.us-east-1.amazonaws.com/belevich/vllm-ep:latest

docker run --gpus all -it --rm \
    -v "$HF_HOME":/root/.cache/huggingface \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e VLLM_ALL2ALL_BACKEND=pplx \
    -e VLLM_USE_DEEP_GEMM=1 \
    -p 8000:8000 \
    --ipc=host \
    ${IMAGE} \
    vllm serve deepseek-ai/deepseek-moe-16b-base --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --enable-expert-parallel 
```
You can also use enroot:
```
enroot import -o ${SQUASH_FILE} dockerd://{IMAGE}
enroot start --rw \
    --mount "$HF_HOME":/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env VLLM_ALL2ALL_BACKEND=pplx \
    --env VLLM_USE_DEEP_GEMM=1 \
    ${SQUASH_FILE} \
    vllm serve deepseek-ai/deepseek-moe-16b-base \
    --trust-remote-code
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --host 0.0.0.0 \
    --port 8000
```
