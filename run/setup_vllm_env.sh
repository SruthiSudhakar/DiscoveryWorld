#!/usr/bin/env bash
# One-time setup: create an isolated conda env for the vLLM server.
# Kept separate from the `discoveryworld` env because vLLM's heavy CUDA-
# PyTorch deps conflict with the pinned `httpx<0.28` and
# `huggingface_hub<1.0` in agents/requirements.txt.
set -euo pipefail

ENV_NAME=vllm_server
PY_VERSION=3.11

# /home/sruthi has a per-user quota that can't fit vllm's deps (~6GB) +
# Qwen3-32B weights (~64GB). Park the uv + HF caches on /proj where there
# is plenty of space (~19TB free).
CACHE_ROOT=/proj/vondrick3/sruthi/.cache
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export HF_HOME="${CACHE_ROOT}/huggingface"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}"

source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup_vllm_env] '${ENV_NAME}' already exists; skipping creation."
else
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

conda activate "${ENV_NAME}"

# Wipe any prior install (e.g. vllm 0.20 built against CUDA 13, which fails
# on this box's CUDA 12.8 driver — `libcudart.so.13: cannot open shared
# object file`) before reinstalling.
pip uninstall -y vllm torch torchvision torchaudio || true

pip install --upgrade pip uv

# Pin to a vllm release whose own compiled extensions still target CUDA 12.x,
# and explicitly request the cu128 torch backend to match the driver here
# (CUDA Version: 12.8 per nvidia-smi). vllm >=0.20 switched its default wheels
# to CUDA 13.
#
# transformers is also pinned <5 because vllm 0.10.2 declares
# `transformers>=4.55.2` with no upper bound, but transformers 5.x removed
# `Qwen2Tokenizer.all_special_tokens_extended` which vllm still calls.
uv pip install "vllm==0.10.2" "transformers>=4.55.2,<5" --torch-backend=cu128

echo
echo "[setup_vllm_env] Done. Launch the server with: bash run/start_vllm_server.sh"
