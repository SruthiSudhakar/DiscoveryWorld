# react smoke test
export DIFF=Normal
export TASK="Small Skills: Dialog Test"
export MAX_ENV_CALLS=1
export SEED=123                        # GPT sampling seed (distinct from scenario seed)
export MODEL=gpt-4.1-mini-2025-04-14
export OUTPUT_DIR=output_dir/smoke/react_dialog_s1

rm -rf "$OUTPUT_DIR"                   # recoma skips existing output; wipe to re-run
python agents/recoma/run_recoma.py \
  --output_dir "$OUTPUT_DIR" \
  --config agents/recoma/configs/react_smoke.jsonnet

#  ReAct smoke test on a real discovery task
export DIFF=Easy
export TASK="Space Sick"
export MAX_ENV_CALLS=50
export SEED=123
export MODEL=gpt-4.1-mini-2025-04-14
export OUTPUT_DIR=output_dir/smoke/react_spacesick_easy_s1

rm -rf "$OUTPUT_DIR"
python agents/recoma/run_recoma.py \
  --output_dir "$OUTPUT_DIR" \
  --config agents/recoma/configs/react_smoke.jsonnet




export DIFF=Easy
export TASK="Archaeology Dating"
export MAX_ENV_CALLS=100
export SEED=123 # Used for GPT
export MODEL=gpt-4.1-mini-2025-04-14
export OUTPUT_DIR=output_dir/baselines_apr29/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/react_one_seed.jsonnet
