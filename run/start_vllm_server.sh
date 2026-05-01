#!/usr/bin/env bash
# Launches a local vLLM server that exposes an OpenAI-compatible chat-completions
# endpoint on port 8000. Run this in a separate tmux pane / shell, then run
# run/run_react_easy_oss.sh from the discoveryworld env in another shell.
#
# First launch downloads ~64GB of Qwen3-32B weights into the HF cache; subsequent
# launches reuse the cached weights.

# # Use 2 specific GPUs                                                                                                                                              
# GPU_DEVICES=0,1 TP_SIZE=2 bash run/start_vllm_server.sh                                                                                                            
                                                                                                                                                                    
# # Use 4 specific GPUs                                                                                                                                              
# GPU_DEVICES=3,4,6,7 TP_SIZE=4 bash run/start_vllm_server.sh                                                                                                        
                                                                                                                                                                    
# # Single GPU (only works if it has ~64GB+ free for the model)                                                                                                      
# GPU_DEVICES=2 TP_SIZE=1 bash run/start_vllm_server.sh                                                                                                              
                                                                                                                                                                    
# Two rules:                                                                                                                                                         
# - TP_SIZE must equal the number of GPUs in GPU_DEVICES — vLLM tensor-parallel-shards the model across exactly that many devices.
# - GPU_MEM_UTIL is the fraction of each GPU's total memory vLLM is allowed to claim (default 0.4 → ~32 GB on an 80 GB A100). If your chosen GPUs have more headroom,
# bump it up (e.g. GPU_MEM_UTIL=0.7) for a larger KV cache and more concurrency. Check nvidia-smi --query-gpu=index,memory.free --format=csv,noheader first.        
                                                                                                                                                                    
# Defaults baked into the script: GPU_DEVICES=2,4,5,7, TP_SIZE=4, GPU_MEM_UTIL=0.4 — so plain bash run/start_vllm_server.sh uses those.


set -euo pipefail

# Use a vLLM-specific var so we don't collide with $MODEL from the recoma
# client side (which is `openai/Qwen/Qwen3-32B`, not the bare HF id).
VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen3-32B}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-4}
MAX_LEN=${MAX_LEN:-32768}
DTYPE=${DTYPE:-bfloat16}
# This box's A100s are usually shared. Default to a 4-GPU spread + 0.4
# memory utilization so vllm only claims ~32GB/GPU, fitting under whatever
# else is co-located. Override via env vars if you have a freer set.
GPU_DEVICES=${GPU_DEVICES:-2,4,5,7}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}

source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda activate vllm_server

# Park the HF model cache on /proj — /home/sruthi has a per-user quota that
# cannot fit the ~64GB of Qwen3-32B weights.
export HF_HOME=${HF_HOME:-/proj/vondrick3/sruthi/.cache/huggingface}
mkdir -p "${HF_HOME}"

export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}

echo "[start_vllm_server] model=${VLLM_MODEL} port=${PORT} tp=${TP_SIZE} max_len=${MAX_LEN} dtype=${DTYPE} gpus=${GPU_DEVICES} mem_util=${GPU_MEM_UTIL}"

vllm serve "${VLLM_MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_LEN}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --trust-remote-code
