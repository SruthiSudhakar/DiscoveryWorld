#P+E easy all
for t in "Lost in Translation" "It's (not)"; do
 export DIFF=Easy
 export TASK=${t}
 export MAX_ENV_CALLS=100
 export SEED=123 # Used for GPT
 export MODEL=gpt-4o-2024-05-13
 export OUTPUT_DIR=output_dir/baselines_apr29/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/react_one_seed.jsonnet
done