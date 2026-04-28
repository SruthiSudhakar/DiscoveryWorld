"""
Task-agnostic error analysis for DiscoveryWorld runs.

Works on any agent (ReAct, plan_and_execute, ...) and any task whose run dir
contains:
  - one or more `<TaskName> <Difficulty>_<idx>_tracking.jsonl` (per episode)
  - a `files/` dir with `<TaskName> <Difficulty>_<idx>.json` LLM trajectories
  - a `source_config.json` (used to read max_env_calls / stopping conditions)

Generic per-episode analysis lives here; anything that needs task-specific
ground truth (e.g. mining numeric answers from criticalHypotheses) lives in
the `task_knowledge` plugin package and is dispatched by task name.

Usage:
python scripts/error_analysis/analyze_run.py \
  output_dir/reproduce/plan_and_execute/Easy_100env_gpt-4.1-mini-2025-04-14_s123/Proteomics


python scripts/error_analysis/analyze_run.py <run_dir>                                                                                           
python scripts/error_analysis/summarize.py <run_dir>/error_analysis.json                                                                         

"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from glob import glob
from typing import Any

from summarize import render_run_summary
from task_knowledge import for_task


def load_tracking(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_trajectory_files(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def trajectory_submit_action(traj: list[dict]) -> dict | None:
    """SUBMIT terminates a run; it appears in files/*.json under model=action."""
    for d in traj:
        if d.get("model") != "action":
            continue
        out = d.get("output") or ""
        try:
            obj = json.loads(out)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and obj.get("action") == "SUBMIT":
            return obj
    return None


def collect_uuids_in_observations(rows: list[dict]) -> set[int]:
    seen: set[int] = set()
    for r in rows:
        obs = r.get("observation") or {}
        ui = obs.get("ui") or {}
        for key in ("inventoryObjects", "accessibleEnvironmentObjects"):
            for o in ui.get(key, []) or []:
                if "uuid" in o:
                    seen.add(o["uuid"])
        nearby = (ui.get("nearbyObjects") or {}).get("objects", {}) or {}
        for direction, objs in nearby.items():
            for o in objs or []:
                if "uuid" in o:
                    seen.add(o["uuid"])
    return seen


def read_max_env_calls(run_dir: str, default: int = 100) -> int:
    cfg_path = os.path.join(run_dir, "source_config.json")
    if not os.path.exists(cfg_path):
        return default
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default
    for cond in (cfg.get("search", {}).get("stopping_conditions") or []):
        if "max_env_calls" in cond:
            return int(cond["max_env_calls"])
    return default


def slugify_checkpoint(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def analyze_episode(tracking_path: str, traj_path: str, max_env_calls: int) -> dict[str, Any]:
    rows = load_tracking(tracking_path)
    traj = load_trajectory_files(traj_path)

    final = rows[-1]
    final_sc = final["scorecard"][0]
    final_score = final_sc["score"]
    max_score = final_sc["maxScore"]
    score_norm = final_score / max_score if max_score else 0.0
    sub_status = {
        s["name"]: (s["score"], s["maxScore"], s.get("associatedNotes", ""))
        for s in final_sc["scoreCard"]
    }
    hypotheses = final_sc.get("criticalHypotheses", []) or []
    task_name = final_sc.get("taskName", "")

    action_steps = [r for r in rows if r.get("action") is not None]
    actions: Counter = Counter()
    failed_steps: list[dict] = []
    error_types: Counter = Counter()

    last_act_key: tuple | None = None
    repeat_run = 0
    max_repeat_run = 0

    seen_uuids = collect_uuids_in_observations(rows)
    hallucinated: list[tuple[int, str, int]] = []

    for r in action_steps:
        a = r["action"]
        name = a.get("action")
        actions[name] += 1
        arg1 = a.get("arg1")
        arg2 = a.get("arg2")

        key = (name, arg1, arg2)
        if key == last_act_key:
            repeat_run += 1
            if repeat_run > max_repeat_run:
                max_repeat_run = repeat_run
        else:
            repeat_run = 1
        last_act_key = key

        for arg in (arg1, arg2):
            if isinstance(arg, int) and arg not in seen_uuids:
                hallucinated.append((r["step"], name, arg))

        res = r.get("result") or {}
        errs = res.get("errors") or []
        success = str(res.get("success", "")).lower() == "true"
        if errs or not success:
            failed_steps.append({
                "step": r["step"], "action": name,
                "errors": errs, "success": res.get("success"),
            })
            for e in errs:
                e_low = e.lower()
                if "could not find object" in e_low and "accessible" in e_low:
                    error_types["target_not_accessible_or_unknown_uuid"] += 1
                elif "inventory" in e_low:
                    error_types["inventory_error"] += 1
                elif "blocked" in e_low or "cannot move" in e_low:
                    error_types["movement_blocked"] += 1
                else:
                    error_types["other_env_error"] += 1
            if not errs and not success:
                error_types["silent_failure"] += 1

    # Trajectory roles
    n_action_calls = sum(1 for d in traj if d.get("model") == "action")
    n_env_calls = sum(1 for d in traj if d.get("model") == "environment")
    planner_outputs = [d for d in traj if d.get("model") == "discworld_planner"]
    n_planner_calls = len(planner_outputs)
    initial_plan = (planner_outputs[0]["output"] if planner_outputs else None)

    submit_in_traj = trajectory_submit_action(traj)
    submitted = submit_in_traj is not None

    hit_step_cap = n_env_calls >= max_env_calls - 1
    premature_threshold = max(8, int(0.3 * max_env_calls))

    # Per-checkpoint missed list
    missed_checkpoints: list[str] = []
    for name, (score, mx, _notes) in sub_status.items():
        if mx and score < mx:
            missed_checkpoints.append(name)

    # Generic failure modes
    failure_modes: list[str] = []
    if submitted and final_score < max_score:
        failure_modes.append("submitted_unfinished")
        if len(action_steps) < premature_threshold:
            failure_modes.append("premature_submit")
    if hit_step_cap and final_score < max_score:
        failure_modes.append("hit_step_budget_unfinished")
    if max_repeat_run >= 3:
        failure_modes.append(f"stuck_in_action_loop(x{max_repeat_run})")
    if hallucinated:
        failure_modes.append("hallucinated_uuid")
    if action_steps and len(failed_steps) / len(action_steps) > 0.5:
        failure_modes.append("high_action_failure_rate")
    if not submitted and not hit_step_cap and final_score < max_score:
        failure_modes.append("trace_truncated_externally")

    # Task-specific knowledge
    plugin = for_task(task_name, os.path.basename(os.path.dirname(tracking_path) or ""))
    task_extra = plugin.analyze(rows, traj, hypotheses, sub_status)
    failure_modes.extend(task_extra.get("failure_modes", []))

    # Representative failure thoughts
    thoughts_around_failure: list[str] = []
    for fs in failed_steps[:5]:
        for r in action_steps:
            if r["step"] == fs["step"]:
                thought = (r["action"] or {}).get("thought", "")
                thoughts_around_failure.append(
                    f"step {fs['step']} ({fs['action']}): {thought[:300]}"
                )
                break

    return {
        "tracking_path": tracking_path,
        "task_name": task_name,
        "n_steps": len(action_steps),
        "n_action_calls": n_action_calls,
        "n_env_calls": n_env_calls,
        "n_planner_calls": n_planner_calls,
        "initial_plan": initial_plan,
        "max_env_calls": max_env_calls,
        "hit_step_cap": hit_step_cap,
        "final_score": final_score,
        "max_score": max_score,
        "score_normalized": score_norm,
        "subscores": {k: v[0] for k, v in sub_status.items()},
        "submax": {k: v[1] for k, v in sub_status.items()},
        "missed_checkpoints": missed_checkpoints,
        "actions": dict(actions),
        "n_failed_steps": len(failed_steps),
        "error_type_counts": dict(error_types),
        "max_repeat_run": max_repeat_run,
        "n_hallucinated_uuid_uses": len(hallucinated),
        "submitted": submitted,
        "submit_text": (submit_in_traj or {}).get("arg1"),
        "task_plugin": plugin.name,
        "task_extra": task_extra.get("extra", {}),
        "failure_modes": failure_modes,
        "critical_hypotheses": hypotheses,
        "first_failures": failed_steps[:5],
        "thoughts_around_failure": thoughts_around_failure,
    }


def analyze_run(run_dir: str) -> dict[str, Any]:
    tracking_files = sorted(glob(os.path.join(run_dir, "*_tracking.jsonl")))
    max_env_calls = read_max_env_calls(run_dir)
    episodes = []
    for tp in tracking_files:
        base = os.path.basename(tp).replace("_tracking.jsonl", "")
        traj = os.path.join(run_dir, "files", base + ".json")
        episodes.append(analyze_episode(tp, traj, max_env_calls))

    fm_counts: Counter = Counter()
    for ep in episodes:
        for fm in ep["failure_modes"]:
            fm_counts[fm.split("(")[0]] += 1

    err_counts: Counter = Counter()
    for ep in episodes:
        for k, v in ep["error_type_counts"].items():
            err_counts[k] += v

    avg = sum(ep["score_normalized"] for ep in episodes) / max(len(episodes), 1)

    return {
        "run_dir": run_dir,
        "task_name": episodes[0]["task_name"] if episodes else None,
        "max_env_calls": max_env_calls,
        "n_episodes": len(episodes),
        "avg_score_normalized": avg,
        "failure_mode_prevalence": dict(fm_counts),
        "env_error_prevalence": dict(err_counts),
        "episodes": episodes,
    }


def render_run_report(result: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("=" * 80)
    out.append(f"RUN: {result['run_dir']}")
    out.append(f"  task: {result['task_name']}  episodes: {result['n_episodes']}  avg(norm): {result['avg_score_normalized']:.3f}")
    out.append(f"  failure-mode prevalence: {result['failure_mode_prevalence']}")
    out.append(f"  env-error prevalence: {result['env_error_prevalence']}")
    out.append("")
    for ep in result["episodes"]:
        name = os.path.basename(ep["tracking_path"]).replace("_tracking.jsonl", "")
        out.append(f"-- {name} --")
        out.append(f"   score {ep['final_score']}/{ep['max_score']} ({ep['score_normalized']:.3f})  steps={ep['n_steps']} env={ep['n_env_calls']} planner={ep['n_planner_calls']} hit_cap={ep['hit_step_cap']}")
        out.append(f"   subscores: {ep['subscores']}")
        out.append(f"   missed: {ep['missed_checkpoints']}")
        out.append(f"   actions: {ep['actions']}")
        out.append(f"   failed_steps: {ep['n_failed_steps']}  err_types: {ep['error_type_counts']}")
        out.append(f"   submitted: {ep['submitted']}  submit_text: {ep['submit_text']}")
        out.append(f"   plugin: {ep['task_plugin']}  task_extra: {ep['task_extra']}")
        if ep["initial_plan"]:
            preview = ep["initial_plan"].replace("\n", " ")[:300]
            out.append(f"   initial_plan: {preview}")
        out.append(f"   FAILURE MODES: {ep['failure_modes']}")
        if ep["thoughts_around_failure"]:
            out.append("   sample failure thoughts:")
            for t in ep["thoughts_around_failure"][:3]:
                out.append(f"     - {t}")
        out.append("")
    return "\n".join(out)


def print_run_report(result: dict[str, Any]) -> None:
    print(render_run_report(result))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", help="Run dirs (e.g. .../Archaeology_Dating)")
    p.add_argument(
        "--json-out",
        help=(
            "Write all results to this single JSON file. "
            "If omitted, each run_dir gets its own error_analysis.json written inside it."
        ),
    )
    args = p.parse_args()

    results = []
    for d in args.run_dirs:
        r = analyze_run(d)
        results.append(r)
        print_run_report(r)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote structured results to {args.json_out}")

    for d, r in zip(args.run_dirs, results):
        if not args.json_out:
            json_path = os.path.join(d, "error_analysis.json")
            with open(json_path, "w") as f:
                json.dump([r], f, indent=2)
            print(f"Wrote structured results to {json_path}")
            report_path = os.path.join(d, "error_analysis_report.txt")
            with open(report_path, "w") as f:
                f.write(render_run_report(r) + "\n")
            print(f"Wrote detailed report to {report_path}")
        summary_text = render_run_summary(r)
        print(summary_text)
        if os.path.isdir(d):
            summary_path = os.path.join(d, "error_analysis_summary.txt")
            with open(summary_path, "w") as f:
                f.write(summary_text + "\n")
            print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
