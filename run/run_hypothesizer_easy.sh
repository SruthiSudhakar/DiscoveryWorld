#Hypothesizer easy all
for t in "Archaeology Dating" "Combinatorial Chemistry"  "Proteomics" "Plant Nutrients" "Reactor Lab" "Space Sick" "Lost in Translation" "It's (not)"; do
 export DIFF=Easy
 export TASK=${t}
 export MAX_ENV_CALLS=100
 export SEED=123 # Used for GPT
 export MODEL=gpt-4.1-mini-2025-04-14
 export OUTPUT_DIR=output_dir/baselines/hypothesizer/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 export VARIATION=$1
 python agents/HypothesizerAgent.py \
  --output_dir="$OUTPUT_DIR" \
  --scenario="$TASK" \
  --difficulty="$DIFF" \
  --numSteps=$MAX_ENV_CALLS \
  --seed="$VARIATION" \
  --model="$MODEL" \
  --video \
  --maxCostDollars=125
done