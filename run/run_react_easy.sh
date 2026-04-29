#P+E easy all
for t in "Archaeology Dating" "Combinatorial Chemistry"  "Proteomics" "Plant Nutrients" "Reactor Lab" "Space Sick" "Lost in Translation" "It's (not)"; do
 export DIFF=Easy
 export TASK=${t}
 export MAX_ENV_CALLS=100
 export SEED=123 # Used for GPT
 export MODEL=gpt-4.1-mini-2025-04-14
 export OUTPUT_DIR=output_dir/baselines_apr29/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/react.jsonnet
done