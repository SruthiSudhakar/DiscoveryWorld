#!/usr/bin/env python3
"""Render a single self-contained HTML visualization of one DiscoveryWorld run.

Supports both `react` and `plan_and_execute` agents (auto-detected from
`source_config.json`, falling back to inspecting the trace). For plan_and_execute
the output additionally interleaves the planner LLM calls (which produce the
high-level subtask sequence) with the executor action calls, so the run reads
top-to-bottom as: plan -> N executor actions -> re-plan -> N more actions -> ...

Per action turn it shows:
  - the rendered game frame (with grid) the agent saw at decision time
  - the **actual reconstructed LLM prompt**, with world-state observation,
    valid actions, teleport destinations, and interactable objects substituted
    into the agent's prompt template (`react_prompt.txt` for react,
    `executer_prompt.txt` for plan_and_execute) — the recoma trace stores only
    the rolling `input_str`, not the final Jinja-rendered prompt
  - the LLM's JSON action response
  - the environment's result

Per plan turn (plan_and_execute only) it shows:
  - the planner's input (task + plan history of prior subtasks/results)
  - the planner's output (the next subtask) — rendered through
    `planner_prompt.txt` so the full prompt the planner saw is visible

The action-prompt reconstruction loads the same DiscoveryWorld scenario once at
startup to grab env-static fields (known actions, additional instructions,
teleport destinations) and uses the per-step `observation.ui` from the tracking
JSONL for everything else. Planner-prompt reconstruction needs no env loading.

Usage:
  python visualize_run.py "<path to *_data.json>" [--output <out.html>]
                          [--no-reconstruct]

Use --no-reconstruct to skip env-loading and show only the recoma trace's raw
`input_str` for action turns (task + history). Plan turns are unaffected.

Fast Mode:
python visualize_run.py \
  "output_dir/reproduce/plan_and_execute/Easy_100env_gpt-4.1-mini-2025-04-14_s123/Proteomics/Proteomics_Easy_1_data.json" \
  --no-reconstruct

Full Mode:
python visualize_run.py \
  "output_dir/reproduce/plan_and_execute/Easy_100env_gpt-4.1-mini-2025-04-14_s123/Proteomics/Proteomics_Easy_1_data.json"

"""

import argparse
import copy
import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

REACT_TEMPLATE_PATH = REPO_ROOT / "agents" / "recoma" / "prompts" / "react_prompt.txt"
EXECUTER_TEMPLATE_PATH = REPO_ROOT / "agents" / "recoma" / "prompts" / "executer_prompt.txt"
PLANNER_TEMPLATE_PATH = REPO_ROOT / "agents" / "recoma" / "prompts" / "planner_prompt.txt"


def load_data_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def find_trace_json(data_json_path: Path) -> Path:
    stem = data_json_path.name
    if not stem.endswith("_data.json"):
        raise ValueError(f"Expected '*_data.json' filename, got {stem!r}")
    base = stem[: -len("_data.json")]
    return data_json_path.parent / "files" / f"{base}.json"


def find_tracking_jsonl(data_json_path: Path) -> Path:
    stem = data_json_path.name
    base = stem[: -len("_data.json")]
    return data_json_path.parent / f"{base}_tracking.jsonl"


def load_trace(path: Path) -> list:
    with path.open() as f:
        return json.load(f)


def load_tracking(path: Path) -> list:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_action_output(output: str) -> dict:
    try:
        return json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return {}


def summarize_action(out_obj: dict) -> str:
    if not out_obj:
        return "(unparseable)"
    parts = [str(out_obj.get("action", "?"))]
    if "arg1" in out_obj:
        parts.append(f"arg1={out_obj['arg1']}")
    if "arg2" in out_obj:
        parts.append(f"arg2={out_obj['arg2']}")
    return " ".join(parts)


def summarize_result(result: dict):
    if not isinstance(result, dict):
        return None, "(no result)", []
    succ = result.get("success")
    if isinstance(succ, str):
        succ_bool = succ.lower() == "true"
    elif isinstance(succ, bool):
        succ_bool = succ
    else:
        succ_bool = None
    errors = result.get("errors", []) or []
    label = "OK" if succ_bool else ("ERROR" if succ_bool is False else "?")
    return succ_bool, label, errors


def img_data_url(vision: dict) -> str:
    if not isinstance(vision, dict):
        return ""
    for key in ("base64_with_grid", "base64_no_grid"):
        v = vision.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def detect_agent_type(data_path: Path, trace: list) -> tuple[str, Path]:
    """Returns (agent_type, action_template_path).

    Prefers source_config.json's `models.action.prompt_file`; falls back to
    inspecting whether the trace contains any `discworld_planner` entries.
    """
    cfg_path = data_path.parent / "source_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            prompt_file = (cfg.get("models", {}).get("action", {})
                              .get("prompt_file") or "")
            if "executer" in prompt_file:
                return "plan_and_execute", EXECUTER_TEMPLATE_PATH
            if "react" in prompt_file:
                return "react", REACT_TEMPLATE_PATH
        except (json.JSONDecodeError, OSError):
            pass
    has_planner = any(
        isinstance(e, dict) and e.get("model") == "discworld_planner"
        for e in trace
    )
    if has_planner:
        return "plan_and_execute", EXECUTER_TEMPLATE_PATH
    return "react", REACT_TEMPLATE_PATH


def build_turns(trace: list, tracking: list) -> list:
    """Walk the trace once; produce a flat ordered list of typed turns.

    Each turn dict carries `kind` ∈ {"plan", "action", "final_state"}:
      - plan: planner LLM call, no env step. Has `input` (task + plan history)
              and `output` (next subtask).
      - action: action LLM call, paired with the next `environment` trace entry
                (when present) and the next tracking row whose `action` is not
                null. Has `input`, `output`, `env_output` (str|None), and
                `track` (dict|None).
      - final_state: last tracking row (the one with `action: null`) — the
                     post-episode observation. Just `track`.
    """
    turns = []
    n_real_track = sum(1 for r in tracking if r.get("action") is not None)
    track_idx = 0
    i = 0
    while i < len(trace):
        e = trace[i]
        if not isinstance(e, dict):
            i += 1
            continue
        model = e.get("model")
        if model == "discworld_planner":
            turns.append({
                "kind": "plan",
                "input": e.get("input", ""),
                "output": e.get("output", ""),
            })
            i += 1
        elif model == "action":
            nxt = trace[i + 1] if i + 1 < len(trace) else None
            has_env = isinstance(nxt, dict) and nxt.get("model") == "environment"
            track = None
            env_output = None
            if has_env:
                env_output = nxt.get("output")
                if track_idx < n_real_track:
                    track = tracking[track_idx]
                    track_idx += 1
            turns.append({
                "kind": "action",
                "input": e.get("input", ""),
                "output": e.get("output", ""),
                "env_output": env_output,
                "track": track,
            })
            i += 2 if has_env else 1
        else:
            # `environment` consumed by previous action, or unknown — skip
            i += 1
    if tracking and tracking[-1].get("action") is None:
        turns.append({"kind": "final_state", "track": tracking[-1]})
    return turns


def render_planner_prompt(input_str: str) -> str:
    """Render planner_prompt.txt with input_str substituted in. Cheap — needs
    no DiscoveryWorld env loading because the planner template only uses
    {{ input_str }}."""
    from jinja2 import Template
    with PLANNER_TEMPLATE_PATH.open() as f:
        tpl = Template(f.read())
    return tpl.render(input_str=input_str)


class PromptReconstructor:
    """Renders an action prompt template the same way the live agent does,
    using per-step `observation.ui` from tracking + env-static fields fetched
    once at startup. The template path is selected by the caller based on
    agent type (react_prompt.txt vs executer_prompt.txt)."""

    def __init__(self, scenario_name: str, difficulty: str, random_seed: int,
                 template_path: Path):
        from jinja2 import Template
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
        from discoveryworld.ScenarioMaker import SCENARIO_NAMES
        from agents.HypothesizerAgent import mkShortInteractableObjectList

        self.mkShortInteractableObjectList = mkShortInteractableObjectList

        with template_path.open() as f:
            self.template = Template(f.read())

        diff2id = {"Easy": 1, "Normal": 2, "Challenge": 3, "Test": 4}
        thread_id = diff2id.get(difficulty, 1) * 1000 + random_seed * 100
        if scenario_name in SCENARIO_NAMES:
            thread_id += SCENARIO_NAMES.index(scenario_name)

        self.env = DiscoveryWorldAPI(threadID=thread_id)
        ok = self.env.loadScenario(
            scenarioName=scenario_name,
            difficultyStr=difficulty,
            randomSeed=random_seed,
            numUserAgents=1,
        )
        if not ok:
            raise RuntimeError(f"Failed to load scenario {scenario_name}/{difficulty}/{random_seed}")

        self.known_actions_str = json.dumps(
            self.env.listKnownActions(limited=False), indent=4, sort_keys=True
        )
        self.teleport_destinations_str = json.dumps(
            self.env.listTeleportLocationsDict(), indent=4, sort_keys=True
        )
        self.additional_instructions = self.env.additionalActionDescriptionString()

    def render(self, observation: dict, input_str: str) -> str:
        """Reproduce populate_template_dictionary using the recorded observation."""
        ui = observation.get("ui", {}) or {}
        agent_loc = ui.get("agentLocation", {}) or {}
        dialog_box = ui.get("dialog_box", {}) or {}
        in_dialog = bool(dialog_box.get("is_in_dialog", False))

        # Mirror discoveryworld_promptlm.populate_template_dictionary:
        # the prompt's "observation" excludes vision and taskProgress.
        observation_no_vision = copy.deepcopy(observation)
        observation_no_vision.pop("vision", None)
        if "ui" in observation_no_vision:
            observation_no_vision["ui"].pop("taskProgress", None)
        observation_str = json.dumps(observation_no_vision, indent=4, sort_keys=True)

        interactable_objects = self.mkShortInteractableObjectList(observation)
        dialog_box_str = json.dumps(dialog_box, indent=4, sort_keys=True)

        return self.template.render(
            input_str=input_str,
            facing_direction=agent_loc.get("faceDirection", ""),
            valid_dirs=agent_loc.get("directions_you_can_move", []),
            additional_instructions=self.additional_instructions,
            observation=observation_str,
            known_actions=self.known_actions_str,
            teleport_destinations=self.teleport_destinations_str,
            interactable_objects=interactable_objects,
            in_dialog=in_dialog,
            dialog_box=dialog_box_str,
        )


CSS = """
:root { --fg: #1a1a1a; --muted: #666; --bg: #fafafa; --card: #fff;
        --border: #e0e0e0; --accent: #2c5282; --ok: #2f855a; --err: #c53030;
        --code-bg: #f5f5f5; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 1.5rem; background: var(--bg); color: var(--fg);
       line-height: 1.4; }
header { max-width: 1400px; margin: 0 auto 1.5rem; padding: 1rem 1.25rem;
         background: var(--card); border: 1px solid var(--border);
         border-radius: 8px; }
header h1 { margin: 0 0 0.5rem; font-size: 1.4rem; }
header .meta { color: var(--muted); font-size: 0.9rem; }
header .task { margin-top: 0.75rem; padding: 0.5rem 0.75rem;
               background: var(--code-bg); border-radius: 4px;
               font-size: 0.9rem; }
header .note { margin-top: 0.5rem; font-size: 0.8rem; color: var(--muted);
               font-style: italic; }
.turns { max-width: 1400px; margin: 0 auto; }
details.turn { background: var(--card); border: 1px solid var(--border);
               border-radius: 6px; margin-bottom: 0.5rem; }
details.turn[open] { border-color: var(--accent); }
summary { padding: 0.6rem 1rem; cursor: pointer; font-weight: 500;
          display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
          list-style: none; }
summary::-webkit-details-marker { display: none; }
summary .step-num { color: var(--accent); font-weight: 700; min-width: 4rem; }
summary .action { font-family: ui-monospace, Menlo, monospace; font-size: 0.85rem;
                  color: var(--fg); }
summary .ok { color: var(--ok); font-size: 0.8rem; font-weight: 600; }
summary .err { color: var(--err); font-size: 0.8rem; font-weight: 600; }
summary .stats { margin-left: auto; color: var(--muted); font-size: 0.8rem; }
details.turn.plan { background: #faf5ff; }
details.turn.plan[open] { border-color: #6b46c1; }
details.turn.plan summary .step-num { color: #6b46c1; }
details.turn.plan summary .kind { background: #6b46c1; color: white;
                                   font-size: 0.7rem; padding: 0.1rem 0.4rem;
                                   border-radius: 3px; letter-spacing: 0.04em; }
details.turn.action summary .kind { background: var(--accent); color: white;
                                     font-size: 0.7rem; padding: 0.1rem 0.4rem;
                                     border-radius: 3px; letter-spacing: 0.04em; }
details.turn.final summary .kind { background: var(--muted); color: white;
                                    font-size: 0.7rem; padding: 0.1rem 0.4rem;
                                    border-radius: 3px; letter-spacing: 0.04em; }
.body { padding: 1rem; border-top: 1px solid var(--border);
        display: grid; grid-template-columns: minmax(300px, 480px) 1fr;
        gap: 1rem; }
.body.plan-body { grid-template-columns: 1fr; }
.frame img { width: 100%; height: auto; border: 1px solid var(--border);
             border-radius: 4px; image-rendering: pixelated; background: #000; }
.frame .no-img { padding: 2rem; text-align: center; color: var(--muted);
                 background: var(--code-bg); border-radius: 4px; }
.io { display: flex; flex-direction: column; gap: 0.75rem; min-width: 0; }
.io section { min-width: 0; }
.io h3 { margin: 0 0 0.4rem; font-size: 0.8rem; text-transform: uppercase;
         color: var(--muted); letter-spacing: 0.04em; }
pre { margin: 0; padding: 0.75rem; background: var(--code-bg);
      border: 1px solid var(--border); border-radius: 4px;
      font-family: ui-monospace, Menlo, monospace; font-size: 0.78rem;
      white-space: pre-wrap; word-break: break-word; max-height: 480px;
      overflow: auto; }
pre.prompt { max-height: 600px; }
pre.output { background: #fffbe6; border-color: #f0d878; }
pre.envresult { background: #f0f5ff; border-color: #c0d4f0; max-height: 200px; }
pre.history { background: #f7fafc; border-color: #cbd5e0; max-height: 320px;
              font-size: 0.75rem; }
.errors { color: var(--err); font-size: 0.85rem; margin-top: 0.25rem; }
details.subdetail { margin-top: 0.25rem; }
details.subdetail summary { padding: 0.3rem 0; font-size: 0.8rem;
                            color: var(--muted); cursor: pointer; }
@media (max-width: 900px) {
  .body { grid-template-columns: 1fr; }
}
"""


def _render_action_turn(turn, turn_idx, action_idx, reconstructor, open_attr):
    input_str = turn["input"] if isinstance(turn["input"], str) else json.dumps(turn["input"], indent=2)
    output = turn["output"] if isinstance(turn["output"], str) else json.dumps(turn["output"], indent=2)
    out_obj = parse_action_output(output) if isinstance(output, str) else {}

    track = turn.get("track") or {}
    track_step = track.get("step")
    track_step_str = f"tracking step {track_step}" if track_step is not None else "no tracking row (action not executed)"
    action_field = track.get("action") if track else None
    result = track.get("result")
    if result is None and turn.get("env_output"):
        try:
            result = json.loads(turn["env_output"])
        except (json.JSONDecodeError, TypeError):
            result = turn["env_output"]
    succ_bool, succ_label, errors = summarize_result(result if isinstance(result, dict) else {})
    action_label = summarize_action(action_field if isinstance(action_field, dict) else out_obj)

    thought = out_obj.get("thought", "") if isinstance(out_obj, dict) else ""
    observation = track.get("observation") or {}
    vision = observation.get("vision") or {}
    img_url = img_data_url(vision)

    full_prompt = None
    if reconstructor is not None and observation:
        try:
            full_prompt = reconstructor.render(observation, input_str)
        except Exception as e:
            full_prompt = f"(reconstruction failed for this step: {e!r})"

    prompt_chars = len(full_prompt) if full_prompt is not None else len(input_str)
    output_chars = len(output)

    succ_class = "ok" if succ_bool else ("err" if succ_bool is False else "")
    frame_html = (
        f'<img src="{html.escape(img_url, quote=True)}" alt="game frame at step {track_step}" />'
        if img_url else '<div class="no-img">(no rendered frame for this step)</div>'
    )
    errors_html = ""
    if errors:
        errs = "<br>".join(html.escape(str(e)) for e in errors)
        errors_html = f'<div class="errors">errors: {errs}</div>'

    if isinstance(result, (dict, list)):
        result_pretty = json.dumps(result, indent=2)
    elif result is None:
        result_pretty = "(action was not executed against the environment)"
    else:
        result_pretty = str(result)

    if full_prompt is not None:
        prompt_section = f"""<section>
        <h3>LLM prompt (full, reconstructed)</h3>
        <pre class="prompt">{html.escape(full_prompt)}</pre>
        <details class="subdetail">
          <summary>show recoma input_str (task + action history) substituted into <code>{{{{ input_str }}}}</code></summary>
          <pre class="history">{html.escape(input_str)}</pre>
        </details>
      </section>"""
    else:
        prompt_section = f"""<section>
        <h3>LLM prompt — recoma input_str only (task + history)</h3>
        <pre class="prompt">{html.escape(input_str)}</pre>
      </section>"""

    thought_html = ""
    if thought:
        thought_html = f"""<section>
        <h3>Thought (extracted from output)</h3>
        <pre class="output">{html.escape(thought)}</pre>
      </section>
"""

    return f"""<details class="turn action"{open_attr}>
  <summary>
    <span class="step-num">#{turn_idx}</span>
    <span class="kind">ACTION {action_idx}</span>
    <span class="action">{html.escape(action_label)}</span>
    <span class="{succ_class}">{html.escape(succ_label)}</span>
    <span class="stats">prompt {prompt_chars:,} chars · output {output_chars:,} chars · {track_step_str}</span>
  </summary>
  <div class="body">
    <div class="frame">
      {frame_html}
    </div>
    <div class="io">
      {prompt_section}
      <section>
        <h3>LLM response (output)</h3>
        <pre class="output">{html.escape(output)}</pre>
      </section>
      {thought_html}<section>
        <h3>Environment result</h3>
        <pre class="envresult">{html.escape(result_pretty)}</pre>
        {errors_html}
      </section>
    </div>
  </div>
</details>"""


def _render_plan_turn(turn, turn_idx, plan_idx, planner_prompt_available, open_attr):
    input_str = turn["input"] if isinstance(turn["input"], str) else json.dumps(turn["input"], indent=2)
    output = turn["output"] if isinstance(turn["output"], str) else json.dumps(turn["output"], indent=2)

    full_prompt = None
    if planner_prompt_available:
        try:
            full_prompt = render_planner_prompt(input_str)
        except Exception as e:
            full_prompt = f"(planner prompt render failed: {e!r})"

    if full_prompt is not None:
        prompt_section = f"""<section>
        <h3>Planner prompt (full, rendered through planner_prompt.txt)</h3>
        <pre class="prompt">{html.escape(full_prompt)}</pre>
        <details class="subdetail">
          <summary>show recoma input_str (task + plan history) substituted into <code>{{{{ input_str }}}}</code></summary>
          <pre class="history">{html.escape(input_str)}</pre>
        </details>
      </section>"""
    else:
        prompt_section = f"""<section>
        <h3>Planner input (task + plan history)</h3>
        <pre class="prompt">{html.escape(input_str)}</pre>
      </section>"""

    output_summary = output.splitlines()[0][:120] if output else ""
    return f"""<details class="turn plan"{open_attr}>
  <summary>
    <span class="step-num">#{turn_idx}</span>
    <span class="kind">PLAN {plan_idx}</span>
    <span class="action">{html.escape(output_summary)}</span>
    <span class="stats">input {len(input_str):,} chars · output {len(output):,} chars</span>
  </summary>
  <div class="body plan-body">
    <div class="io">
      {prompt_section}
      <section>
        <h3>Planner output (next subtask)</h3>
        <pre class="output">{html.escape(output)}</pre>
      </section>
    </div>
  </div>
</details>"""


def _render_final_state_turn(turn, turn_idx):
    track = turn.get("track") or {}
    track_step = track.get("step", "?")
    observation = track.get("observation") or {}
    vision = observation.get("vision") or {}
    img_url = img_data_url(vision)
    frame_html = (
        f'<img src="{html.escape(img_url, quote=True)}" alt="final game frame" />'
        if img_url else '<div class="no-img">(no rendered frame)</div>'
    )
    return f"""<details class="turn final">
  <summary>
    <span class="step-num">#{turn_idx}</span>
    <span class="kind">FINAL STATE</span>
    <span class="action">post-episode observation</span>
    <span class="stats">tracking step {track_step}</span>
  </summary>
  <div class="body">
    <div class="frame">
      {frame_html}
    </div>
    <div class="io">
      <section>
        <h3>Note</h3>
        <pre class="envresult">The episode has ended. This is the final game frame; no further action was issued.</pre>
      </section>
    </div>
  </div>
</details>"""


def render_html(data, turns, agent_type, action_template_path, reconstructor, reconstruct_error):
    md = data.get("metadata", {}) or {}
    final_scorecard = (md.get("final_scorecard") or [{}])[0]
    score = final_scorecard.get("score", "?")
    max_score = final_scorecard.get("maxScore", "?")
    norm = final_scorecard.get("scoreNormalized")
    norm_str = f"{norm:.3f}" if isinstance(norm, (int, float)) else str(norm)
    completed_ok = final_scorecard.get("completedSuccessfully")

    plan_count = sum(1 for t in turns if t["kind"] == "plan")
    action_count = sum(1 for t in turns if t["kind"] == "action")
    env_step_count = sum(1 for t in turns if t["kind"] == "action" and t.get("track"))

    template_label = action_template_path.relative_to(REPO_ROOT)
    if reconstructor is not None:
        recon_note = (
            f"Agent: <strong>{agent_type}</strong>. Action prompts reconstructed via "
            f"<code>{html.escape(str(template_label))}</code>; planner prompts (if any) "
            f"via <code>agents/recoma/prompts/planner_prompt.txt</code>. The raw recoma "
            f"<code>input_str</code> appears as a collapsed sub-section in each turn."
        )
    elif reconstruct_error:
        recon_note = (
            f"Agent: <strong>{agent_type}</strong>. <strong>Action-prompt reconstruction unavailable</strong> "
            f"({html.escape(reconstruct_error)}); showing recoma <code>input_str</code> only for action turns. "
            f"Planner prompts (if any) are still rendered via "
            f"<code>agents/recoma/prompts/planner_prompt.txt</code> since they need no env loading."
        )
    else:
        recon_note = (
            f"Agent: <strong>{agent_type}</strong>. Reconstruction skipped (--no-reconstruct). "
            f"Showing recoma <code>input_str</code> only for action turns."
        )

    header_html = f"""<header>
  <h1>{html.escape(str(data.get('scenario_name', '?')))} · {html.escape(str(data.get('difficulty', '?')))} · seed {html.escape(str(data.get('random_seed', '?')))}</h1>
  <div class="meta">
    Score: <strong>{html.escape(str(score))}/{html.escape(str(max_score))}</strong>
    (normalized {html.escape(norm_str)}) ·
    Env steps: <strong>{env_step_count}</strong> ·
    Action LLM calls: <strong>{action_count}</strong> ·
    Plan LLM calls: <strong>{plan_count}</strong> ·
    Completed successfully: <strong>{html.escape(str(completed_ok))}</strong>
  </div>
  <div class="task">{html.escape(str(data.get('task_description', '')))}</div>
  <div class="note">{recon_note}</div>
</header>
"""

    turn_blocks = []
    plan_idx = 0
    action_idx = 0
    planner_prompt_available = PLANNER_TEMPLATE_PATH.exists()
    for i, turn in enumerate(turns):
        open_attr = " open" if i == 0 else ""
        if turn["kind"] == "plan":
            plan_idx += 1
            turn_blocks.append(_render_plan_turn(
                turn, i + 1, plan_idx, planner_prompt_available, open_attr
            ))
        elif turn["kind"] == "action":
            action_idx += 1
            turn_blocks.append(_render_action_turn(
                turn, i + 1, action_idx, reconstructor, open_attr
            ))
        elif turn["kind"] == "final_state":
            turn_blocks.append(_render_final_state_turn(turn, i + 1))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(str(data.get('scenario_name', 'run')))} · seed {html.escape(str(data.get('random_seed', '')))}</title>
<style>{CSS}</style>
</head>
<body>
{header_html}
<div class="turns">
{chr(10).join(turn_blocks)}
</div>
</body>
</html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description="Visualize a DiscoveryWorld run as a static HTML.")
    p.add_argument("data_json", help="Path to <Task>_<Difficulty>_<seed>_data.json")
    p.add_argument("--output", "-o", default=None,
                   help="Output HTML path (default: <run>/<base>_visualization.html)")
    p.add_argument("--no-reconstruct", action="store_true",
                   help="Skip env-loading; show only the recoma input_str (task+history).")
    args = p.parse_args(argv)

    data_path = Path(args.data_json).expanduser().resolve()
    if not data_path.exists():
        print(f"error: {data_path} not found", file=sys.stderr)
        return 1

    trace_path = find_trace_json(data_path)
    tracking_path = find_tracking_jsonl(data_path)

    missing = [p for p in (trace_path, tracking_path) if not p.exists()]
    if missing:
        for m in missing:
            print(f"error: required sibling file missing: {m}", file=sys.stderr)
        return 1

    data = load_data_json(data_path)
    raw_trace = load_trace(trace_path)
    tracking = load_tracking(tracking_path)

    agent_type, action_template_path = detect_agent_type(data_path, raw_trace)
    turns = build_turns(raw_trace, tracking)

    plan_count = sum(1 for t in turns if t["kind"] == "plan")
    action_count = sum(1 for t in turns if t["kind"] == "action")
    paired = sum(1 for t in turns if t["kind"] == "action" and t.get("track"))
    n_real_track = sum(1 for r in tracking if r.get("action") is not None)
    print(f"detected agent: {agent_type} (action template: "
          f"{action_template_path.relative_to(REPO_ROOT)})", file=sys.stderr)
    if action_count != n_real_track:
        print(f"note: {action_count} action LLM calls, {paired} paired with env steps "
              f"(== {n_real_track} executed tracking rows). The {action_count - paired} "
              f"unpaired action(s) were not executed (parse failure or terminal SUBMIT).",
              file=sys.stderr)

    reconstructor = None
    reconstruct_error = None
    if not args.no_reconstruct:
        try:
            print("loading env to reconstruct action prompts (this takes ~2s) ...", file=sys.stderr)
            reconstructor = PromptReconstructor(
                scenario_name=str(data.get("scenario_name")),
                difficulty=str(data.get("difficulty")),
                random_seed=int(data.get("random_seed", 0)),
                template_path=action_template_path,
            )
            print("env loaded; rendering ...", file=sys.stderr)
        except Exception as e:
            reconstruct_error = f"{type(e).__name__}: {e}"
            print(f"warning: action prompt reconstruction unavailable: {reconstruct_error}",
                  file=sys.stderr)
            print("falling back to recoma input_str only for action turns. Re-run with the",
                  file=sys.stderr)
            print("discoveryworld conda env active to enable reconstruction, or pass",
                  file=sys.stderr)
            print("--no-reconstruct to suppress this warning.", file=sys.stderr)

    out_html = render_html(data, turns, agent_type, action_template_path,
                           reconstructor, reconstruct_error)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        base = data_path.name[: -len("_data.json")]
        out_path = data_path.parent / f"{base}_visualization.html"

    out_path.write_text(out_html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.1f} MB, {plan_count} plan turns, "
          f"{action_count} action turns, {len(tracking)} tracking rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
