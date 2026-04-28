"""Print a compact per-episode summary table from results.json (task-agnostic).

Also writes each run's summary to <run_dir>/error_analysis_summary.txt so it
sits next to the error_analysis.json that analyze_run.py drops there by default.
"""
import json
import os
import re
import sys


def render_run_summary(run: dict) -> str:
    out: list[str] = []
    out.append("")
    out.append("=" * 100)
    out.append(f"RUN: {run['run_dir']}")
    out.append(
        f"task: {run.get('task_name')}  "
        f"avg_score(norm)={run['avg_score_normalized']:.3f}  "
        f"max_env_calls={run.get('max_env_calls')}"
    )
    out.append("")

    eps = run["episodes"]
    if not eps:
        out.append("(no episodes)")
        return "\n".join(out)

    submax = eps[0].get("submax", {})
    checkpoint_names = list(submax.keys())

    headers = ["episode", "score", "steps", "submit", "cap", "planner"] + [n[:14] for n in checkpoint_names] + ["failure_modes"]
    widths = [10, 7, 6, 8, 5, 8] + [16] * len(checkpoint_names) + [80]
    line = "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    out.append(line)
    out.append("-" * len(line))

    for ep in eps:
        base = ep["tracking_path"].split("/")[-1].replace("_tracking.jsonl", "")
        m = re.search(r"_(\d+)$", base)
        epname = f"ep{m.group(1)}" if m else base
        sub = ep["subscores"]
        cells = [
            epname,
            f"{ep['final_score']}/{ep['max_score']}",
            str(ep["n_steps"]),
            "Y" if ep["submitted"] else "N",
            "Y" if ep["hit_step_cap"] else "N",
            str(ep.get("n_planner_calls", 0)),
        ]
        for n in checkpoint_names:
            cells.append(f"{sub.get(n, 0)}/{submax.get(n, 0)}")
        cells.append(", ".join(ep["failure_modes"]))
        out.append("".join(f"{c:<{w}}" for c, w in zip(cells, widths)))

    if any(ep.get("initial_plan") for ep in eps):
        out.append("")
        out.append("Initial plans (planner output):")
        for ep in eps:
            ip = ep.get("initial_plan")
            if not ip:
                continue
            base = ep["tracking_path"].split("/")[-1].replace("_tracking.jsonl", "")
            m = re.search(r"_(\d+)$", base)
            epname = f"ep{m.group(1)}" if m else base
            preview = ip.replace("\n", " ")[:240]
            out.append(f"  {epname}: {preview}")

    if any(ep.get("task_extra") for ep in eps):
        out.append("")
        out.append("Task-specific extras:")
        for ep in eps:
            extra = ep.get("task_extra") or {}
            if not extra:
                continue
            base = ep["tracking_path"].split("/")[-1].replace("_tracking.jsonl", "")
            m = re.search(r"_(\d+)$", base)
            epname = f"ep{m.group(1)}" if m else base
            out.append(f"  {epname}: {extra}")

    out.append("")
    out.append("Failure-mode prevalence across episodes:")
    for k, v in sorted(run["failure_mode_prevalence"].items(), key=lambda kv: -kv[1]):
        out.append(f"  {v}/{run['n_episodes']:>2}  {k}")
    out.append("")
    out.append("Env-error prevalence:")
    for k, v in sorted(run["env_error_prevalence"].items(), key=lambda kv: -kv[1]):
        out.append(f"  {v:>3}  {k}")

    return "\n".join(out)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "scripts/error_analysis/results.json"
    with open(src) as f:
        runs = json.load(f)

    for run in runs:
        text = render_run_summary(run)
        print(text)
        run_dir = run.get("run_dir")
        if run_dir and os.path.isdir(run_dir):
            out_path = os.path.join(run_dir, "error_analysis_summary.txt")
            with open(out_path, "w") as f:
                f.write(text + "\n")
            print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
