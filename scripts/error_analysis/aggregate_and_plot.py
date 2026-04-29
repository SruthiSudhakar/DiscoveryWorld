"""Aggregate per-task error_analysis.json files and produce cross-task visuals.

Reads every error_analysis.json under the run root and emits:
  - aggregate.json / aggregate.csv : flat per-episode rows
  - per_task_score.png             : avg normalized score per task with episode dots
  - subscore_completion.png        : per-checkpoint completion rate, grouped by task
  - failure_mode_heatmap.png       : task x failure-mode counts
  - env_error_heatmap.png          : task x env-error counts
  - submit_behavior.png            : submitted vs hit-cap, premature vs honest
  - steps_vs_score.png             : scatter of episode length vs score
  - capability_gaps.png            : aggregated gap chart for the writeup
  - paper_metrics_triple.png       : Procedure / Knowledge / Completion bars
  - knowledge_*.png (if perf summary present)

Usage:
    python aggregate_and_plot.py --root <run_dir> [--out <out_dir>] [--label "<title suffix>"]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_p = argparse.ArgumentParser(description="Aggregate per-task error_analysis.json into cross-task plots.")
_p.add_argument("--root", required=True, help="Run root containing per-task subdirs.")
_p.add_argument("--out", default=None, help="Output dir (defaults to <root>/_aggregate).")
_p.add_argument("--label", default=None,
                help="Label used in plot titles (e.g. 'ReAct gpt-4.1-mini Easy'). "
                     "Defaults to the run-root directory name.")
_args = _p.parse_args()
ROOT = Path(_args.root)
OUT = Path(_args.out) if _args.out else (ROOT / "_aggregate")
LABEL = _args.label if _args.label else ROOT.name
OUT.mkdir(exist_ok=True)
print(f"[aggregate] root={ROOT}\n[aggregate] out={OUT}\n[aggregate] label={LABEL}")

TASK_DIRS = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name != "_aggregate"])

rows = []
task_summary = []
for td in TASK_DIRS:
    p = td / "error_analysis.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text())[0]
    task_summary.append({
        "task_dir": td.name,
        "task_name": data["task_name"],
        "n_episodes": data["n_episodes"],
        "avg_score_normalized": data["avg_score_normalized"],
        "max_env_calls": data["max_env_calls"],
        "failure_mode_prevalence": data.get("failure_mode_prevalence", {}),
        "env_error_prevalence": data.get("env_error_prevalence", {}),
    })
    for ep in data["episodes"]:
        rows.append({
            "task_dir": td.name,
            "task_name": data["task_name"],
            "ep_path": ep["tracking_path"].split("/")[-1].replace("_tracking.jsonl", ""),
            "n_steps": ep["n_steps"],
            "n_env_calls": ep["n_env_calls"],
            "hit_step_cap": ep["hit_step_cap"],
            "final_score": ep["final_score"],
            "max_score": ep["max_score"],
            "score_normalized": ep["score_normalized"],
            "subscores": ep["subscores"],
            "submax": ep["submax"],
            "missed_checkpoints": ep["missed_checkpoints"],
            "actions": ep["actions"],
            "n_failed_steps": ep["n_failed_steps"],
            "error_type_counts": ep["error_type_counts"],
            "max_repeat_run": ep["max_repeat_run"],
            "n_hallucinated_uuid_uses": ep["n_hallucinated_uuid_uses"],
            "submitted": ep["submitted"],
            "submit_text": ep.get("submit_text") or "",
            "failure_modes": ep["failure_modes"],
            "first_failures": ep.get("first_failures", []),
        })

df = pd.DataFrame(rows)

# --- optional: join per-episode knowledge-eval score from performanceSummary*.json
TASK_DIR_TO_PERF_NAME = {
    "Archaeology_Dating": "Archaeology Dating",
    "Combinatorial_Chemistry": "Combinatorial Chemistry",
    "It's_(not)": "It's (not) Rocket Science!",
    "Lost_in_Translation": "Lost in Translation",
    "Plant_Nutrients": "Plant Nutrients",
    "Proteomics": "Proteomics",
    "Reactor_Lab": "Reactor Lab",
    "Space_Sick": "Space Sick",
}
PERF_SUMMARY = None
for cand in ROOT.glob("performanceSummary*.json"):
    PERF_SUMMARY = cand
    break

knowledge_index = {}        # (task_dir, seed_int) -> dict with knowledge fields
knowledge_q_pass = defaultdict(lambda: {"pass": 0, "total": 0})  # task_dir -> per-question pass counts
perf_averages = {}          # task_dir -> avg fields
if PERF_SUMMARY is not None:
    print(f"[aggregate] joining knowledge eval from {PERF_SUMMARY.name}")
    perf = json.loads(PERF_SUMMARY.read_text())
    perf_name_to_dir = {v: k for k, v in TASK_DIR_TO_PERF_NAME.items()}
    for rec in perf.get("data", []):
        td = perf_name_to_dir.get(rec.get("taskName"))
        if td is None:
            continue
        seed = int(rec.get("seed", -1))
        knowledge_index[(td, seed)] = {
            "knowledge_score": rec.get("knowledgeEvaluationScoreNormalized"),
            "knowledge_error": rec.get("knowledgeEvaluationError", 0),
            "perf_normalized_score": rec.get("normalizedScore"),
            "perf_completed": rec.get("completedSuccessfully"),
        }
        # break out per-question pass-rate
        for ke in (rec.get("knowledgeEvaluation") or []):
            for q in ke.get("evaluation", []) or []:
                key = q.get("criticalQuestion", "?")[:80]
                knowledge_q_pass[(td, key)]
                knowledge_q_pass[(td, key)]["pass"] += int(q.get("evaluation") or 0)
                knowledge_q_pass[(td, key)]["total"] += 1
    for k, v in perf.get("averages", {}).items():
        # k looks like "Archaeology Dating_Easy_0steps"
        for pname, td in perf_name_to_dir.items():
            if k.startswith(pname + "_Easy"):
                perf_averages[td] = v
                break

# infer seed from ep filename (..._Easy_<seed>)
def _ep_to_seed(ep):
    try:
        return int(ep.split("_Easy_")[-1].split("_")[0])
    except Exception:
        return -1
df["seed"] = df["ep_path"].map(_ep_to_seed)
df["knowledge_score"] = df.apply(
    lambda r: (knowledge_index.get((r["task_dir"], int(r["seed"])), {}) or {}).get("knowledge_score"),
    axis=1,
)
df["knowledge_error_flag"] = df.apply(
    lambda r: (knowledge_index.get((r["task_dir"], int(r["seed"])), {}) or {}).get("knowledge_error"),
    axis=1,
)

df.to_json(OUT / "aggregate.json", orient="records", indent=2)
df_flat = df.copy()
for col in ["subscores", "submax", "missed_checkpoints", "actions", "error_type_counts", "failure_modes", "first_failures"]:
    df_flat[col] = df_flat[col].apply(json.dumps)
df_flat.to_csv(OUT / "aggregate.csv", index=False)

# --- Pretty task labels (short)
TASK_SHORT = {
    "Archaeology_Dating": "Archaeology",
    "Combinatorial_Chemistry": "Combinat. Chem.",
    "It's_(not)": "Rocket Sci.",
    "Lost_in_Translation": "Translation",
    "Plant_Nutrients": "Plant Nutr.",
    "Proteomics": "Proteomics",
    "Reactor_Lab": "Reactor",
    "Space_Sick": "Space Sick",
}
df["task_short"] = df["task_dir"].map(TASK_SHORT)
# Only include tasks we actually have error_analysis.json for, preserving canonical ordering.
_present_dirs = {ts["task_dir"] for ts in task_summary}
TASK_ORDER = [TASK_SHORT[td] for td in TASK_SHORT if td in _present_dirs]
_missing_dirs = [td for td in TASK_SHORT if td not in _present_dirs]
if _missing_dirs:
    print(f"[aggregate] skipping tasks without error_analysis.json: {_missing_dirs}")
if not TASK_ORDER:
    raise SystemExit("[aggregate] no tasks with error_analysis.json found under --root; run analyze_run.py first.")

# Detect step cap from data (max n_steps observed, rounded to a nice number)
_max_steps = int(df["n_steps"].max()) if len(df) else 0
STEP_CAP = _max_steps  # used as the dashed-line "cap" reference in steps-vs-score plot

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============ 1. avg score per task (with episode dots) ============
fig, ax = plt.subplots(figsize=(10, 5))
task_order = TASK_ORDER
xs = np.arange(len(task_order))
means = [df[df["task_short"] == t]["score_normalized"].mean() for t in task_order]
bars = ax.bar(xs, means, color="#5B8DEF", alpha=0.85, edgecolor="white")
for i, t in enumerate(task_order):
    pts = df[df["task_short"] == t]["score_normalized"].values
    jitter = (np.random.RandomState(i).rand(len(pts)) - 0.5) * 0.25
    ax.scatter(np.full_like(pts, i, dtype=float) + jitter, pts, color="#1E1E2C", s=24, zorder=3, alpha=0.9)
for i, m in enumerate(means):
    ax.text(i, m + 0.02, f"{m:.2f}", ha="center", fontsize=9, color="#1E1E2C")
ax.axhline(1.0, color="#999", lw=0.7, linestyle="--")
ax.set_xticks(xs)
ax.set_xticklabels(task_order, rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Normalized score (0=fail, 1=full)")
ax.set_title(f"{LABEL} · per-task average score (dots=episodes)")
fig.tight_layout()
fig.savefig(OUT / "per_task_score.png")
plt.close(fig)

# ============ 2. subscore completion rate (per-task per-checkpoint) ============
# Build matrix: rows = (task, checkpoint), value = mean(sub/submax)
subrows = []
for _, r in df.iterrows():
    for ck, mx in r["submax"].items():
        if mx == 0:
            continue
        subrows.append({
            "task_short": r["task_short"],
            "checkpoint": ck,
            "completion": r["subscores"].get(ck, 0) / mx,
        })
sdf = pd.DataFrame(subrows)
fig, ax = plt.subplots(figsize=(13, 6))
agg = sdf.groupby(["task_short", "checkpoint"])["completion"].mean().reset_index()
labels = []
vals = []
colors = []
palette = plt.cm.tab10.colors
task_to_color = {t: palette[i % 10] for i, t in enumerate(task_order)}
for t in task_order:
    sub = agg[agg["task_short"] == t]
    for _, row in sub.iterrows():
        labels.append(f"{t}\n{row['checkpoint']}")
        vals.append(row["completion"])
        colors.append(task_to_color[t])
xs = np.arange(len(labels))
ax.bar(xs, vals, color=colors, edgecolor="white")
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Completion rate (avg over episodes)")
ax.set_title(f"{LABEL} · per-checkpoint completion rate")
fig.tight_layout()
fig.savefig(OUT / "subscore_completion.png")
plt.close(fig)

# ============ 3. failure-mode heatmap (task x mode) ============
all_modes = set()
for ts in task_summary:
    all_modes.update(ts["failure_mode_prevalence"].keys())
all_modes = sorted(all_modes)
mat = np.zeros((len(task_order), len(all_modes)), dtype=int)
for i, t_short in enumerate(task_order):
    ts = next(x for x in task_summary if TASK_SHORT[x["task_dir"]] == t_short)
    for j, m in enumerate(all_modes):
        mat[i, j] = ts["failure_mode_prevalence"].get(m, 0)
fig, ax = plt.subplots(figsize=(max(8, len(all_modes) * 0.95), 5))
im = ax.imshow(mat, aspect="auto", cmap="Reds")
ax.set_xticks(np.arange(len(all_modes)))
ax.set_xticklabels(all_modes, rotation=35, ha="right")
ax.set_yticks(np.arange(len(task_order)))
ax.set_yticklabels(task_order)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v >= 3 else "#333", fontsize=9)
ax.set_title(f"{LABEL} · failure-mode prevalence (# episodes flagged)")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="# episodes")
fig.tight_layout()
fig.savefig(OUT / "failure_mode_heatmap.png")
plt.close(fig)

# ============ 4. env-error heatmap (task x error type) ============
all_errs = set()
for ts in task_summary:
    all_errs.update(ts["env_error_prevalence"].keys())
all_errs = sorted(all_errs)
emat = np.zeros((len(task_order), len(all_errs)), dtype=int)
for i, t_short in enumerate(task_order):
    ts = next(x for x in task_summary if TASK_SHORT[x["task_dir"]] == t_short)
    for j, e in enumerate(all_errs):
        emat[i, j] = ts["env_error_prevalence"].get(e, 0)
fig, ax = plt.subplots(figsize=(max(8, len(all_errs) * 1.5), 5))
im = ax.imshow(emat, aspect="auto", cmap="Oranges")
ax.set_xticks(np.arange(len(all_errs)))
ax.set_xticklabels(all_errs, rotation=15, ha="right")
ax.set_yticks(np.arange(len(task_order)))
ax.set_yticklabels(task_order)
for i in range(emat.shape[0]):
    for j in range(emat.shape[1]):
        v = emat[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v >= 25 else "#333", fontsize=9)
ax.set_title(f"{LABEL} · env-error totals (raw counts of failed action calls)")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="# failed env calls")
fig.tight_layout()
fig.savefig(OUT / "env_error_heatmap.png")
plt.close(fig)

# ============ 5. submit behavior matrix ============
# 4 quadrants: submitted+honest (score=1), submitted+premature, no_submit+hit_cap, no_submit+truncated
def classify(r):
    if r["submitted"]:
        if r["score_normalized"] >= 1.0 - 1e-6:
            return "submitted_full_credit"
        return "submitted_unfinished"
    if r["hit_step_cap"]:
        return "no_submit_hit_cap"
    return "no_submit_truncated"
df["status"] = df.apply(classify, axis=1)
cats = ["submitted_full_credit", "submitted_unfinished", "no_submit_hit_cap", "no_submit_truncated"]
mat = np.zeros((len(task_order), len(cats)), dtype=int)
for i, t in enumerate(task_order):
    counts = df[df["task_short"] == t]["status"].value_counts().to_dict()
    for j, c in enumerate(cats):
        mat[i, j] = counts.get(c, 0)
fig, ax = plt.subplots(figsize=(9, 5))
bottom = np.zeros(len(task_order))
colors_q = ["#52b788", "#e63946", "#f4a261", "#a8a8a8"]
for j, c in enumerate(cats):
    ax.bar(task_order, mat[:, j], bottom=bottom, color=colors_q[j], label=c, edgecolor="white")
    bottom += mat[:, j]
ax.set_ylabel("# episodes")
ax.set_title(f"{LABEL} · episode termination behavior")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_xticks(np.arange(len(task_order)))
ax.set_xticklabels(task_order, rotation=20, ha="right")
fig.tight_layout()
fig.savefig(OUT / "submit_behavior.png")
plt.close(fig)

# ============ 6. steps vs score scatter ============
fig, ax = plt.subplots(figsize=(8, 5))
for t in task_order:
    sub = df[df["task_short"] == t]
    ax.scatter(sub["n_steps"], sub["score_normalized"], label=t, s=60, alpha=0.85,
               color=task_to_color[t], edgecolor="white")
ax.set_xlabel("Episode length (steps)")
ax.set_ylabel("Normalized score")
ax.set_title(f"{LABEL} · effort vs. reward")
if STEP_CAP > 0:
    ax.axvline(STEP_CAP, ls="--", color="#999", lw=0.8)
    ax.text(STEP_CAP, 1.02, "max steps observed", color="#666", fontsize=8, ha="right")
ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
ax.set_ylim(-0.05, 1.1)
fig.tight_layout()
fig.savefig(OUT / "steps_vs_score.png")
plt.close(fig)

# ============ 7. capability-gap summary chart ============
GAP_DEFINITIONS = {
    "Verification before submit\n(no self-check that subgoals done)": [
        ("submitted_unfinished", lambda fm: "submitted_unfinished" in fm),
        ("premature_submit", lambda fm: "premature_submit" in fm),
    ],
    "Grounded memory of measurements\n(taking readings but not citing them)": [
        ("never_cited_correct_oldest_age", lambda fm: "never_cited_correct_oldest_age" in fm),
        ("dated_but_didnt_identify_oldest", lambda fm: "dated_but_didnt_identify_oldest" in fm),
    ],
    "Spatial / relative-position reasoning\n(flag-placement, neighbor tile)": [
        ("wrong_oldest_pick_or_misplaced_flag", lambda fm: "wrong_oldest_pick_or_misplaced_flag" in fm),
        ("missed_flag_subgoal", lambda fm: False),  # filled below from missed_checkpoints
    ],
    "Action-affordance / error-aware retry\n(repeats failing action, ignores error msg)": [
        ("stuck_in_action_loop", lambda fm: any(x.startswith("stuck_in_action_loop") for x in fm)),
        ("high_action_failure_rate", lambda fm: "high_action_failure_rate" in fm),
    ],
    "Step-budget management\n(loops until cap)": [
        ("hit_step_budget_unfinished", lambda fm: "hit_step_budget_unfinished" in fm),
    ],
}
gap_counts = {}
for gap, preds in GAP_DEFINITIONS.items():
    n = 0
    for _, r in df.iterrows():
        fm = r["failure_modes"]
        miss = r["missed_checkpoints"]
        if gap.startswith("Spatial"):
            spatial_miss = any(("Flag" in c) or ("flag" in c) or ("Place" in c) for c in miss)
            if spatial_miss or "wrong_oldest_pick_or_misplaced_flag" in fm:
                n += 1
        else:
            if any(p(fm) for _, p in preds):
                n += 1
    gap_counts[gap] = n
# sort gaps by count desc for clearer reading
gap_items = sorted(gap_counts.items(), key=lambda x: -x[1])
fig, ax = plt.subplots(figsize=(10, 6.5))
ys = np.arange(len(gap_items))
vals = [v for _, v in gap_items]
labels = [k for k, _ in gap_items]
total = len(df)
ax.barh(ys, vals, color="#7E22CE", alpha=0.85, edgecolor="white")
for i, v in enumerate(vals):
    ax.text(v + 0.4, i, f"  {v}/{total}  ({v/total*100:.0f}% of episodes)", va="center", fontsize=10)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, total)
ax.set_xlabel(f"# episodes (of {total} total) flagged with this capability gap")
ax.set_title(f"{LABEL} · capability gaps mapped from failure modes (ranked by prevalence)")
fig.tight_layout()
fig.savefig(OUT / "capability_gaps.png")
plt.close(fig)

# ============ 8. action-mix vs failure rate per task ============
mix = []
for _, r in df.iterrows():
    total_actions = sum(r["actions"].values()) or 1
    failed = r["n_failed_steps"]
    mix.append({"task_short": r["task_short"], "n_steps": r["n_steps"], "fail_rate": failed / max(r["n_steps"], 1)})
mdf = pd.DataFrame(mix)
fig, ax = plt.subplots(figsize=(9, 5))
mean_fail = mdf.groupby("task_short")["fail_rate"].mean().reindex(task_order)
ax.bar(task_order, mean_fail.values, color="#E63946", alpha=0.85, edgecolor="white")
for i, v in enumerate(mean_fail.values):
    ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
ax.set_ylabel("Failed env-action rate")
ax.set_title(f"{LABEL} · fraction of agent actions rejected by env")
ax.set_xticks(np.arange(len(task_order)))
ax.set_xticklabels(task_order, rotation=20, ha="right")
ax.set_ylim(0, max(0.6, mean_fail.max() + 0.1))
fig.tight_layout()
fig.savefig(OUT / "action_failure_rate.png")
plt.close(fig)

# ============ 9. per-episode failure grid ============
def base_mode(m):
    return m.split("(")[0]
all_eps = []
all_mode_set = set()
for _, r in df.iterrows():
    bm = sorted({base_mode(x) for x in r["failure_modes"]})
    all_mode_set.update(bm)
    all_eps.append((r["task_short"], r["ep_path"].split("_")[-1], bm, r["score_normalized"]))
modes_sorted = sorted(all_mode_set)
grid = np.zeros((len(all_eps), len(modes_sorted)), dtype=int)
for i, (_, _, bm, _s) in enumerate(all_eps):
    for j, m in enumerate(modes_sorted):
        grid[i, j] = 1 if m in bm else 0
fig, ax = plt.subplots(figsize=(max(10, len(modes_sorted) * 1.0), max(7, len(all_eps) * 0.22)))
im = ax.imshow(grid, aspect="auto", cmap="Reds", vmin=0, vmax=1.4)
ax.set_xticks(np.arange(len(modes_sorted)))
ax.set_xticklabels(modes_sorted, rotation=35, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(all_eps)))
ax.set_yticklabels([f"{t} ep{e}  ({s:.2f})" for t, e, _, s in all_eps], fontsize=8)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i, j]:
            ax.text(j, i, "X", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
# horizontal separators between tasks
prev = None
for i, (t, _, _, _) in enumerate(all_eps):
    if prev is not None and t != prev:
        ax.axhline(i - 0.5, color="black", lw=0.6)
    prev = t
ax.set_title(f"{LABEL} · per-episode failure-mode grid")
fig.tight_layout()
fig.savefig(OUT / "episode_failure_grid.png")
plt.close(fig)

# ============ 10. subscore completion heatmap (task x checkpoint) ============
# Use the union of checkpoint names per-task; build a long table.
# Pad each task row to its own checkpoints (different tasks have different ckpts).
fig, axes = plt.subplots(len(task_order), 1, figsize=(11, 1.3 * len(task_order) + 1))
for ax, t in zip(axes, task_order):
    sub = sdf[sdf["task_short"] == t]
    cks = list(sub["checkpoint"].unique())
    arr = sub.groupby("checkpoint")["completion"].mean().reindex(cks).values.reshape(1, -1)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cks)))
    ax.set_xticklabels(cks, fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels([t], fontsize=9)
    for j, v in enumerate(arr[0]):
        ax.text(j, 0, f"{v:.2f}", ha="center", va="center",
                color="black" if 0.25 < v < 0.85 else "white", fontsize=9)
fig.suptitle(f"{LABEL} · per-checkpoint completion (green=solved, red=collapsed)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "checkpoint_heatmap.png")
plt.close(fig)

# ============ paper-style triple-bar (Procedure / Knowledge / Completion) ============
# Compute completion (binary completedSuccessfully) per episode by re-reading scorecard
def _completed_for_ep(task_dir, ep_path):
    p = ROOT / task_dir / f"{ep_path}_tracking.jsonl"
    if not p.exists():
        return None
    last = None
    for line in open(p):
        last = json.loads(line)
    return 1 if (last["scorecard"][0].get("completedSuccessfully")) else 0
df["completed_successfully"] = df.apply(lambda r: _completed_for_ep(r["task_dir"], r["ep_path"]), axis=1)
means_proc = df.groupby("task_short")["score_normalized"].mean().reindex(task_order)
means_know = df.groupby("task_short")["knowledge_score"].mean().reindex(task_order)
means_comp = df.groupby("task_short")["completed_successfully"].mean().reindex(task_order)
fig, ax = plt.subplots(figsize=(12, 5.5))
xs = np.arange(len(task_order))
w = 0.27
ax.bar(xs - w, means_proc.values, w, label="Procedure (rubric / maxScore)", color="#5B8DEF", edgecolor="white")
ax.bar(xs,     means_know.values, w, label="Knowledge (critical Qs passed)", color="#7E22CE", edgecolor="white")
ax.bar(xs + w, means_comp.values, w, label="Completion (completedSuccessfully)", color="#52b788", edgecolor="white")
for i in range(len(task_order)):
    for off, vals in [(-w, means_proc.values), (0, means_know.values), (w, means_comp.values)]:
        v = vals[i]
        if np.isnan(v):
            ax.text(i + off, 0.02, "n/a", ha="center", fontsize=7, color="#888")
        else:
            ax.text(i + off, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(xs)
ax.set_xticklabels(task_order, rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Avg per task")
ax.set_title(f"{LABEL} · Procedure / Knowledge / Completion (paper Table-5 metrics)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "paper_metrics_triple.png")
plt.close(fig)

# also export the same numbers as a tasks × 3 table (csv + markdown + json + png)
paper_table_rows = []
for t_dir in TASK_SHORT:
    t = TASK_SHORT[t_dir]
    proc = means_proc.get(t)
    know = means_know.get(t)
    comp = means_comp.get(t)
    paper_table_rows.append({
        "task_dir": t_dir,
        "task": t,
        "Procedure": None if (proc is None or (isinstance(proc, float) and np.isnan(proc))) else round(float(proc), 3),
        "Knowledge": None if (know is None or (isinstance(know, float) and np.isnan(know))) else round(float(know), 3),
        "Completion": None if (comp is None or (isinstance(comp, float) and np.isnan(comp))) else round(float(comp), 3),
    })
# overall row (mean across episodes, weighted by # episodes per task)
n_per_task = df.groupby("task_short").size().reindex(task_order)
def _weighted_mean(series, n):
    s = pd.Series(series, index=task_order)
    mask = s.notna()
    if not mask.any():
        return None
    return float((s[mask] * n[mask]).sum() / n[mask].sum())
overall = {
    "task_dir": "OVERALL",
    "task": "OVERALL",
    "Procedure": round(_weighted_mean(means_proc.values, n_per_task), 3),
    "Knowledge": (None if not df["knowledge_score"].notna().any() else round(_weighted_mean(means_know.values, n_per_task), 3)),
    "Completion": round(_weighted_mean(means_comp.values, n_per_task), 3),
}
paper_table_rows.append(overall)

paper_df = pd.DataFrame(paper_table_rows)
paper_df.to_csv(OUT / "paper_metrics_table.csv", index=False)
paper_df.to_json(OUT / "paper_metrics_table.json", orient="records", indent=2)

# markdown
md_lines = ["| Task | Procedure | Knowledge | Completion |",
            "|---|---:|---:|---:|"]
def _fmt(v):
    return f"{v:.3f}" if v is not None else "n/a"
for r in paper_table_rows:
    label = r["task"] if r["task"] != "OVERALL" else "**OVERALL**"
    md_lines.append(f"| {label} | {_fmt(r['Procedure'])} | {_fmt(r['Knowledge'])} | {_fmt(r['Completion'])} |")
(OUT / "paper_metrics_table.md").write_text("\n".join(md_lines) + "\n")

# small table-as-image
fig, ax = plt.subplots(figsize=(7, 0.45 * (len(paper_table_rows) + 1) + 0.6))
ax.axis("off")
cell_text = [[r["task"], _fmt(r["Procedure"]), _fmt(r["Knowledge"]), _fmt(r["Completion"])] for r in paper_table_rows]
tbl = ax.table(cellText=cell_text, colLabels=["Task", "Procedure", "Knowledge", "Completion"],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.1, 1.4)
# bold last row
for j in range(4):
    tbl[(len(paper_table_rows), j)].set_text_props(weight="bold")
    tbl[(0, j)].set_text_props(weight="bold")
ax.set_title(f"{LABEL} · Procedure / Knowledge / Completion", pad=10)
fig.tight_layout()
fig.savefig(OUT / "paper_metrics_table.png", bbox_inches="tight")
plt.close(fig)

# ============ 11. KNOWLEDGE-EVAL plots (only if performanceSummary was found) ============
have_knowledge = df["knowledge_score"].notna().any()
if have_knowledge:
    # 11a. paired bar: avg task score vs avg knowledge score per task
    means_task = df.groupby("task_short")["score_normalized"].mean().reindex(task_order)
    means_know = df.groupby("task_short")["knowledge_score"].mean().reindex(task_order)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(task_order))
    w = 0.38
    ax.bar(xs - w/2, means_task.values, w, label="task score (norm)", color="#5B8DEF", edgecolor="white")
    ax.bar(xs + w/2, means_know.values, w, label="knowledge score (norm)", color="#7E22CE", edgecolor="white")
    for i, t in enumerate(task_order):
        v1 = means_task.values[i]
        v2 = means_know.values[i]
        if not np.isnan(v1):
            ax.text(i - w/2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=8)
        if not np.isnan(v2):
            ax.text(i + w/2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=8)
        else:
            ax.text(i + w/2, 0.02, "no data", ha="center", fontsize=7, color="#888")
    ax.set_xticks(xs)
    ax.set_xticklabels(task_order, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Avg normalized score")
    ax.set_title(f"{LABEL} · task score vs. knowledge-eval score")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "knowledge_vs_task_bars.png")
    plt.close(fig)

    # 11b. scatter: per-episode (task_score, knowledge_score), colored by task
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for t in task_order:
        sub = df[(df["task_short"] == t) & df["knowledge_score"].notna()]
        if sub.empty:
            continue
        # add small jitter so overlapping points are visible
        rs = np.random.RandomState(hash(t) % (2**32))
        jx = (rs.rand(len(sub)) - 0.5) * 0.04
        jy = (rs.rand(len(sub)) - 0.5) * 0.04
        ax.scatter(sub["score_normalized"].values + jx, sub["knowledge_score"].values + jy,
                   color=task_to_color[t], s=70, alpha=0.85, edgecolor="white", label=t)
    # quadrant lines at 0.5
    ax.axvline(0.5, color="#999", lw=0.7, ls="--")
    ax.axhline(0.5, color="#999", lw=0.7, ls="--")
    ax.set_xlabel("Task score (normalized)")
    ax.set_ylabel("Knowledge-eval score (normalized)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.text(0.02, 0.96, "knowledge ✓ but task ✗\n→ execution gap", fontsize=9, color="#444",
            transform=ax.transAxes, va="top")
    ax.text(0.55, 0.96, "knowledge ✓ AND task ✓\n→ true success", fontsize=9, color="#444",
            transform=ax.transAxes, va="top")
    ax.text(0.02, 0.16, "knowledge ✗ AND task ✗\n→ planning/reasoning gap", fontsize=9, color="#444",
            transform=ax.transAxes, va="top")
    ax.text(0.55, 0.16, "knowledge ✗ but task ✓\n→ luck / spec ambiguity", fontsize=9, color="#444",
            transform=ax.transAxes, va="top")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="center right", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"{LABEL} · per-episode insight vs execution (0.5 thresholds)")
    fig.tight_layout()
    fig.savefig(OUT / "knowledge_vs_task_scatter.png")
    plt.close(fig)

    # 11c. quadrant counts (insight × execution decomposition)
    sub = df[df["knowledge_score"].notna()].copy()
    def _qd(r):
        k = r["knowledge_score"]
        s = r["score_normalized"]
        return ("insight" if k >= 0.5 else "no-insight") + " · " + ("solved" if s >= 0.5 else "not-solved")
    sub["quadrant"] = sub.apply(_qd, axis=1)
    qcat = ["insight · solved", "insight · not-solved", "no-insight · solved", "no-insight · not-solved"]
    qtotals = {c: int((sub["quadrant"] == c).sum()) for c in qcat}
    qcolors = {"insight · solved": "#52b788", "insight · not-solved": "#f4a261",
               "no-insight · solved": "#a8a8a8", "no-insight · not-solved": "#e63946"}
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(qcat))
    ys = [qtotals[c] for c in qcat]
    ax.bar(xs, ys, color=[qcolors[c] for c in qcat], edgecolor="white")
    n_eval = sum(ys)
    for i, v in enumerate(ys):
        ax.text(i, v + 0.3, f"{v}/{n_eval}\n({v/n_eval*100:.0f}%)" if n_eval else f"{v}", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(qcat, rotation=15, ha="right")
    ax.set_ylim(0, max(ys) + 4)
    ax.set_ylabel(f"# episodes (of {n_eval} with knowledge eval)")
    ax.set_title(f"{LABEL} · insight × execution decomposition (0.5 thresholds)")
    fig.tight_layout()
    fig.savefig(OUT / "insight_execution_quadrants.png")
    plt.close(fig)

    # 11d. critical-question pass rate per task (stacked: pass, fail)
    if knowledge_q_pass:
        # collect per-task
        per_task = defaultdict(lambda: {"pass": 0, "total": 0})
        for (td, _q), counts in knowledge_q_pass.items():
            per_task[td]["pass"] += counts["pass"]
            per_task[td]["total"] += counts["total"]
        tasks_present = [t for t in TASK_SHORT if t in per_task]
        labels_p = [TASK_SHORT[t] for t in tasks_present]
        passes = [per_task[t]["pass"] for t in tasks_present]
        fails = [per_task[t]["total"] - per_task[t]["pass"] for t in tasks_present]
        fig, ax = plt.subplots(figsize=(10, 5))
        xs = np.arange(len(tasks_present))
        ax.bar(xs, passes, color="#52b788", edgecolor="white", label="answered (eval=1)")
        ax.bar(xs, fails, bottom=passes, color="#e63946", edgecolor="white", label="missing (eval=0)")
        for i, (p, t) in enumerate(zip(passes, [per_task[t]["total"] for t in tasks_present])):
            ax.text(i, t + 0.2, f"{p}/{t}", ha="center", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels_p, rotation=20, ha="right")
        ax.set_ylabel("# critical-question evaluations across episodes")
        ax.set_title(f"{LABEL} · critical-question pass rate")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "critical_question_pass.png")
        plt.close(fig)

# ============ summary print ============
print("\nTASK SUMMARY")
header = f"{'Task':25s} {'avg':>6s} {'epis':>5s} {'know':>6s}  failure-modes"
print(header)
for ts in task_summary:
    fm = ", ".join(f"{k}:{v}" for k, v in sorted(ts["failure_mode_prevalence"].items(), key=lambda x: -x[1]))
    k = perf_averages.get(ts["task_dir"], {}).get("avgKnowledgeScore")
    kstr = f"{k:.3f}" if k is not None else "  -- "
    print(f"{ts['task_dir']:25s} {ts['avg_score_normalized']:.3f} {ts['n_episodes']:5d} {kstr:>6s}  {fm}")

print(f"\nCapability gap rollup (across {len(df)} episodes):")
for k, v in gap_counts.items():
    label = k.split('\n')[0]
    print(f"  {label:60s}  {v:>2d}/{len(df)}")

print(f"\nWritten to {OUT}")
for f in sorted(OUT.iterdir()):
    if f.suffix == ".png":
        print("  ", f.name)
