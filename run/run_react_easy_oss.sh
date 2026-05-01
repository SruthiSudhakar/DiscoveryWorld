#ReAct easy all -- open-source LLM (Qwen3-32B served locally via vLLM)
#
# Prereqs (one-time):
#   bash run/setup_vllm_env.sh           # creates the `vllm_server` conda env
#
# Each run:
#   In a separate shell / tmux pane:  bash run/start_vllm_server.sh
#   Wait for "Application startup complete" before running this script.
#
# This script keeps run/run_react_easy.sh untouched so the GPT baseline
# stays runnable for direct comparison.
for t in "Archaeology Dating" "Combinatorial Chemistry"  "Proteomics" "Plant Nutrients" "Reactor Lab" "Space Sick" "Lost in Translation" "It's (not)"; do
 export DIFF=Easy
 export TASK=${t}
 export MAX_ENV_CALLS=100
 export SEED=123
 # `openai/<hf_id>` tells LiteLLM to use its OpenAI client; combined with
 # OPENAI_BASE_URL below it routes to the local vLLM server.
 export MODEL=openai/Qwen/Qwen3-32B
 export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8000/v1}
 export OPENAI_API_KEY=EMPTY
 # Short label used in output paths (avoids the `openai/` prefix and the slash).
 MODEL_LABEL=Qwen3-32B
 export OUTPUT_DIR=output_dir/baselines_apr29_oss/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL_LABEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/react_oss.jsonnet
done
