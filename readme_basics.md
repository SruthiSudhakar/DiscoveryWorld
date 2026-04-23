# DiscoveryWorld — Hands-On Basics

A practical walkthrough for running DiscoveryWorld on this machine. Copy-paste recipes, ordered from fastest/free to slower/$$.

For conceptual background, see `README.md`. For per-agent usage, see `agents/README.md` and `agents/recoma/README.md`. This file is the "just show me what to type" complement.

---

## 0. One-time setup (already done)

The conda env and dependencies have been installed. Recorded here for reproducibility:

```bash
# Create env (Python 3.9 per the main README)
source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda create -y --name discoveryworld python=3.9
conda activate discoveryworld

# Install deps
cd /proj/vondrick3/sruthi/discovery/DiscoveryWorld
pip install -r agents/requirements.txt   # includes recoma @ pinned commit
pip install -e .                         # install discoveryworld package

# Two post-install pins (required, else recoma crashes at import or first API call)
pip install 'huggingface_hub<1.0'        # gradio 4.44 needs HfFolder, removed in HF Hub 1.x
pip install 'httpx<0.28'                 # openai 1.30.1 passes proxies= which httpx 0.28+ rejects
```

---

## 1. Activate the env (every new shell)

```bash
source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda activate discoveryworld
cd /proj/vondrick3/sruthi/discovery/DiscoveryWorld
```

Sanity check:
```bash
python -c "import discoveryworld, recoma; print('env ok')"
```

---

## 2. Random Agent (no API key, ~5 seconds)

Fastest way to watch the environment tick. Picks random actions; almost never solves anything — which is the point: it's the floor to compare smart agents against.

```bash
python agents/RandomAgent.py \
  --scenario="Small Skills: Dialog Test" \
  --difficulty=Normal \
  --seed=0 \
  --numSteps=25
```

Try a few different tasks to see how observations differ:
```bash
python agents/RandomAgent.py --scenario="Space Sick"       --difficulty=Challenge --seed=0 --numSteps=10
python agents/RandomAgent.py --scenario="Proteomics"       --difficulty=Normal    --seed=0 --numSteps=20
python agents/RandomAgent.py --scenario="Plant Nutrients"  --difficulty=Easy      --seed=0 --numSteps=10
```

Outputs (dropped in the repo root):
```bash
ls -t output_random_agent*.json | head -3
cat $(ls -t output_random_agent*.json | head -1) | python -m json.tool
```

Key fields: `completedSuccessfully`, `finalNormalizedScore`, `stepsPerSecond`.

---

## 3. ReAct smoke test (needs OpenAI key, ~$0.16, ~30 s)

First, export your key in the current shell (not written to disk):
```bash
export OPENAI_API_KEY='sk-...'
```

Run it:
```bash
export DIFF=Normal
export TASK="Small Skills: Dialog Test"
export MAX_ENV_CALLS=25
export SEED=123                        # GPT sampling seed (distinct from scenario seed)
export MODEL=gpt-4o-2024-05-13
export OUTPUT_DIR=output_dir/smoke/react_dialog_s1

rm -rf "$OUTPUT_DIR"                   # recoma skips existing output; wipe to re-run
python agents/recoma/run_recoma.py \
  --output_dir "$OUTPUT_DIR" \
  --config agents/recoma/configs/react_smoke.jsonnet
```

The smoke config (`react_smoke.jsonnet`) is a copy of `react.jsonnet` with `limit_seeds: [1]` so a single run does one scenario seed. The original `react.jsonnet` runs 5 seeds — use that for full paper reproduction, not smoke testing.

Inspect outputs:
```bash
cat "$OUTPUT_DIR/all_data.jsonl" | python -m json.tool
cat "$OUTPUT_DIR/Small Skills: Dialog Test_Normal_1_data.json" | python -m json.tool | head -40
ls -lh "$OUTPUT_DIR/"*.mp4            # video of the run (scp to watch locally)
```

Per-task JSON keys to look at in `*_data.json`:
- `metadata.final_scorecard[0].completedSuccessfully` — did the task pass?
- `metadata.final_scorecard[0].scoreNormalized` — procedural progress (0–1)
- `metadata.num_steps` — env ticks used
- `metadata.litellm.gpt-4o-2024-05-13.cost` — USD spent
- `metadata.litellm.gpt-4o-2024-05-13.calls` — LLM calls made

---

## 4. ReAct on a real discovery task (~$1–2, ~3–5 min)

Space Sick / Easy / 1 seed / 50 steps. Probably won't *finish* (Easy uses 100 steps in the paper), but you'll see the agent wander, talk to NPCs, and use instruments.

```bash
export DIFF=Easy
export TASK="Space Sick"
export MAX_ENV_CALLS=50
export SEED=123
export MODEL=gpt-4o-2024-05-13
export OUTPUT_DIR=output_dir/smoke/react_spacesick_easy_s1

rm -rf "$OUTPUT_DIR"
python agents/recoma/run_recoma.py \
  --output_dir "$OUTPUT_DIR" \
  --config agents/recoma/configs/react_smoke.jsonnet
```

Open `$OUTPUT_DIR/*_tracking.jsonl` to see every step's prompt, response, and action.

---

## 5. Plan+Execute for comparison (same task, different agent)

First make a smoke copy of its config (its default also runs 5 seeds):

```bash
cp agents/recoma/configs/plan_and_execute.jsonnet \
   agents/recoma/configs/plan_and_execute_smoke.jsonnet
sed -i 's/"limit_seeds": \[1, 2, 3, 4, 5\]/"limit_seeds": [1]/' \
   agents/recoma/configs/plan_and_execute_smoke.jsonnet
```

Run the same Space Sick task with Plan+Execute:
```bash
export DIFF=Easy
export TASK="Space Sick"
export MAX_ENV_CALLS=50
export SEED=123
export MODEL=gpt-4o-2024-05-13
export OUTPUT_DIR=output_dir/smoke/pne_spacesick_easy_s1

rm -rf "$OUTPUT_DIR"
python agents/recoma/run_recoma.py \
  --output_dir "$OUTPUT_DIR" \
  --config agents/recoma/configs/plan_and_execute_smoke.jsonnet
```

Compare the two `*_tracking.jsonl` files: ReAct interleaves one thought + one action per step; Plan+Execute writes a multi-step plan up front and then executes through it.

---

## 6. Hypothesizer (paper flagship, different launcher)

Uses a different entry point and expects the key in a file. Start cheap:

```bash
echo 'sk-...' > openai_key.txt           # HypothesizerAgent reads this file
python agents/HypothesizerAgent.py \
  --scenario="Small Skills: Dialog Test" \
  --difficulty=Normal \
  --seed=0 \
  --numSteps=25 \
  --model=gpt-4o-2024-05-13 \
  --video \
  --maxCostDollars=5
```

Outputs land in the repo root:
- `output_allhistory*.json` — step-by-step obs/action/knowledge trace
- `output_consolodatedKnowledge*.json` — periodic knowledge summaries
- `output_costAnalysis*.json` — per-call cost breakdown
- `output*.mp4` — rendered video

The `output_allhistory*.json` file is the most interesting read — you can watch the agent's working memory grow step by step.

---

## 7. Human GUI (only with an X display)

```bash
python scripts/userstudy.py
```

Will error with `pygame.error: No available video device` on a headless node. Options to actually play: `ssh -X` with X forwarding, run locally, or use VNC. Not viable on a compute-only server.

---

## 8. Knobs to vary

| Knob | Values | Notes |
|---|---|---|
| `--scenario` / `TASK` | 8 discovery tasks + 10 unit tests + Tutorial | See `agents/README.md` §1.1 for the full list, or `from discoveryworld.ScenarioMaker import SCENARIO_NAMES`. |
| `--difficulty` / `DIFF` | `Easy` / `Normal` / `Challenge` | Easy often gives shortcuts (e.g., species list provided); Challenge strips them. |
| `--seed` / scenario seed | 0–4 (official benchmark) | Varies NPC names, map layout, and the correct answer. Recoma configs use seeds 1–5; other agents use 0–4. |
| `--numSteps` / `MAX_ENV_CALLS` | integer | Paper uses 100 for Easy + unit tests, 1000 for Normal + Challenge. |
| `--model` / `MODEL` | any LiteLLM-compatible model | Paper baseline is `gpt-4o-2024-05-13`. |
| `SEED` (recoma env var) | integer | GPT *sampling* seed for reproducibility, **not** the scenario seed. |

Recoma configs also expose `max_history`, `max_output_length`, `max_llm_cost` (default $50/run safety cap) — see `agents/recoma/configs/react.jsonnet`.

---

## 9. Cost cheat-sheet (GPT-4o, rough)

| Run | ~Cost | ~Time |
|---|---|---|
| Random Agent (any scenario) | $0 | seconds |
| ReAct — Dialog Test, 25 steps | $0.15 | 30 s |
| ReAct — Space Sick Easy, 50 steps | $1–2 | 3–5 min |
| ReAct — Space Sick Easy, 100 steps, 5 seeds | $10–25 | 30–60 min |
| ReAct — Space Sick Normal, 1000 steps, 1 seed | $10–25 | ~1 h |
| Full paper sweep (8 tasks × 3 difficulties × 5 seeds × 2 agents) | $100s | days |

Always run one task end-to-end before launching a sweep. The per-run `max_llm_cost: 50.00` cap in the configs protects against a single runaway run, **not** against cumulative spend across runs.

---

## 10. Where outputs land

- **Random Agent:** `output_random_agent*.json` in repo root.
- **Hypothesizer:** `output_*` files + `logs/out*partXofY.zip` + `output*.mp4` in repo root.
- **Recoma (ReAct / Plan+Execute):** everything inside `$OUTPUT_DIR`:
  - `all_data.jsonl` — aggregated scorecard across scenarios
  - `<scenario>_<diff>_<seed>_data.json` — per-run detail
  - `<scenario>_<diff>_<seed>_tracking.jsonl` — prompt/response per step (the thing to read to *understand* what the agent did)
  - `<scenario>_<diff>_<seed>.mp4` — video
  - `predictions.json`, `source_config.json`, `logs/`, `files/`

---

## 11. Full paper sweep (only after you're comfortable)

When you're ready to reproduce the paper numbers, use the **un-modified** `react.jsonnet` / `plan_and_execute.jsonnet` (those run all 5 seeds). The command blocks for every difficulty tier are spelled out in `agents/recoma/README.md` §"All Experiments".

Broad shape:
- Small Skills (Normal): `TASK="Small Skills"`, `MAX_ENV_CALLS=100`
- Discovery Easy: loop over 8 tasks, `DIFF=Easy`, `MAX_ENV_CALLS=100`
- Discovery Normal / Challenge: loop over 8 tasks, `MAX_ENV_CALLS=1000`

Before launching the full sweep, dry-run a single Easy task with all 5 seeds to confirm scale and cost.

---

## 12. Security note

If you paste your API key into any shared transcript (chat, logs), rotate it afterward in the OpenAI dashboard. The `openai_key.txt` file in the repo root is `.gitignore`d but still plaintext on disk — treat it accordingly.
