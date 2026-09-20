"""Bounded LLM propose/execute/repair loop; no hand-authored task solution."""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from aspire_so100.context import build_prompt, read_usage
from aspire_so100.skill_report import build_report

ROOT = Path(__file__).resolve().parents[2]
DEBUG = (51, 52, 53)
VALIDATION = (61, 62)

API = """Write Python defining run(api). Imports: math and numpy only.
Task: blue_block on table, small_red_block on blue_block, green_block on red.
Each object is a free rigid body. Never teleport, attach, or modify physics.
api.get_observation() returns task, end_effector_position, robot_joint_positions,
objects[name]: position, quaternion_wxyz, linear_velocity, angular_velocity,
half_extent_world. Units: meters, radians, seconds. Poses are simulator ground
truth in this SO-100 adaptation; they are not vision estimates.
api.move_end_effector(position, seconds=1.2, wrist_roll=None): position-only IK;
wrist_roll optionally fixes the wrist joint angle, NOT world object yaw.
api.move_to_joint_positions(five_arm_angles, seconds=1.2)
api.open_gripper(), api.close_gripper(), api.set_gripper(angle, seconds=.6)
Jaw range -0.174..1.75; open uses .8, close uses .25. Calibrate from outcomes.
api.wait(seconds=.5)
api.get_overhead_rgbd(): optional on-demand simulated overhead camera (requires
working OpenGL renderer). Returns rgb uint8[240,320,3], depth_m float[240,320]
(optical-axis meters), valid_depth bool[240,320], intrinsics 3x3 and
camera_to_world 4x4 (camera axes right/down/forward), time in simulation seconds.
For pixel (u,v), world_point = camera_to_world @ [inv(intrinsics) @ [u,v,1] * depth_m[v,u], 1].
Selected initial/outcome side RGB, depth previews and wrist RGB are attached
to the coding LLM prompt. Raw metric depth stays in local NPZ files.
Ground-truth poses remain auxiliary inputs; this is not an image-only policy.
api.check_structure(names, duration=2.): names must be a bottom-up prefix of
[blue_block,small_red_block,green_block]. Returns stable, reasons, blocks.
Check after releasing and retracting, before adding the next object. Stop on
instability. A passing two-block check for at least 2 seconds is mandatory
before completing the tower. The evaluator independently tests the full structure for 5 s.
Success requires support contacts, flat faces within 8 degrees, low velocities,
less than 3 mm drift and a 3 mm upper-substack COM margin on support faces.
The original single-object policy is NOT supplied: discover your own program.
Do not access api internals, evaluator, filesystem, network or external tools
from the generated Python. No imports except math/numpy. Use only this API.
Nested helpers may use nonlocal for enclosing-function state; global is forbidden.
"""


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def schema(fields):
    return {"type": "object", "properties": {k: {"type": "string"} for k in fields},
            "required": list(fields), "additionalProperties": False}


def resolve_codex(explicit=None):
    """Resolve Codex from an override, PATH, or the VS Code extension install."""
    override = explicit or os.environ.get("ASPIRE_CODEX_BIN")
    if override:
        resolved = shutil.which(override) if os.sep not in override else override
        path = Path(resolved or override).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
        raise FileNotFoundError(f"Codex executable is not usable: {path}")
    found = shutil.which("codex")
    if found:
        return Path(found).resolve()
    candidates = []
    for base in (Path.home() / ".vscode-server/extensions", Path.home() / ".vscode/extensions"):
        candidates.extend(base.glob("openai.chatgpt-*/bin/*/codex"))
    candidates = [p for p in candidates if p.is_file() and os.access(p, os.X_OK)]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime).resolve()
    raise FileNotFoundError(
        "Codex CLI was not found on PATH or in the VS Code extension. "
        "Pass --codex-bin /path/to/codex or set ASPIRE_CODEX_BIN."
    )


def ask_llm(prompt, directory, fields, model=None, timeout=300, codex_bin=None, image_paths=()):
    executable = resolve_codex(codex_bin)
    directory.mkdir(parents=True)
    if not image_paths:
        from aspire_so100.tower import TowerEnv
        from aspire_so100.rgbd import save_visual_observation
        env = TowerEnv(directory / "initial_scene")
        try:
            env.reset(DEBUG[0])
            image_paths = [save_visual_observation(env, directory / "initial_scene.png", "Initial scene (no execution)")]
        finally:
            env.close()
    write_json(directory / "images.json", {"attachments": list(image_paths),
        "layout": "Each image: side RGB | depth preview (white .25m, black 1m) | wrist RGB"})
    (directory / "prompt.txt").write_text(prompt)
    write_json(directory / "schema.json", schema(fields))
    command = [str(executable), "exec", "--ephemeral", "--sandbox", "read-only",
               "--skip-git-repo-check", "--cd", str(directory),
               "--output-schema", str(directory / "schema.json"),
               "--output-last-message", str(directory / "response.json"), "--json"]
    if model:
        command += ["--model", model]
    for path in image_paths:
        command += ["--image", str(path)]
    command += ["-"]
    with (directory / "agent.jsonl").open("w") as stdout, (directory / "agent.stderr").open("w") as stderr:
        subprocess.run(command, input=prompt, text=True, stdout=stdout, stderr=stderr,
                       check=True, timeout=timeout)
    response = json.loads((directory / "response.json").read_text())
    if set(response) != set(fields) or not all(isinstance(v, str) for v in response.values()):
        raise ValueError("Invalid LLM response schema")
    return response


def validate_policy(source):
    """API-policy lint, NOT a security sandbox. Trials also use bwrap."""
    tree = ast.parse(source)
    banned = {"open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars", "__import__", "breakpoint", "input"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module]
            if any(x not in ("math", "numpy") for x in modules):
                raise ValueError("Only math/numpy imports are allowed")
        if isinstance(node, ast.Name) and (node.id in banned or node.id.startswith("__")):
            raise ValueError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in {"model", "data", "scene", "reset", "evaluate", "load", "save", "fromfile", "tofile", "ctypes"}):
            raise ValueError(f"Forbidden attribute: {node.attr}")
        if isinstance(node, ast.Global):
            raise ValueError("Global declarations are not allowed")
    if not any(isinstance(x, ast.FunctionDef) and x.name == "run" for x in tree.body):
        raise ValueError("Policy must define run(api)")
    # Parsing alone does not check whether a nonlocal binding actually exists.
    # Compile without executing so invalid scope declarations fail before trials.
    compile(tree, "<generated-policy>", "exec")


def worker(candidate, output, seed):
    from aspire_so100.tower import TowerEnv, TowerAPI, ORDER
    from aspire_so100.stacking_env import _jsonable
    from dataclasses import asdict
    source = candidate.read_text()
    validate_policy(source)
    namespace = {}
    exec(compile(source, str(candidate), "exec"), namespace)
    env = TowerEnv(output)
    error = None
    try:
        env.reset(seed)
        from aspire_so100.rgbd import save_visual_observation
        initial_image = save_visual_observation(env, output / "initial.png", "Before policy")
        try:
            namespace["run"](TowerAPI(env))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            env.failure = error
        env.check_structure(list(ORDER), duration=5.)
        metrics = env.evaluate()
        final_image = save_visual_observation(env, output / "outcome.png", "After policy and stability evaluation")
        env.write_trace(metrics, error)
        result = {"seed": seed, "error": error, "metrics": asdict(metrics),
                  "visual_observations": {"initial": initial_image, "outcome": final_image}}
        write_json(output / "result.json", _jsonable(result))
    finally:
        env.close()


def sandbox_command(command, output):
    if not shutil.which("bwrap"):
        raise RuntimeError("bwrap is required for generated-policy trials")
    # Read-only host packages; only this trial output and temporary storage writable.
    # Hide credentials in the user's home. The project resolves under /storage/project.
    return ["bwrap", "--die-with-parent", "--unshare-net", "--new-session",
            "--ro-bind", "/", "/", "--tmpfs", str(Path.home()),
            "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev",
            "--bind", str(output), str(output), "--chdir", str(output), *command]


def run_trials(candidate, seeds, output, timeout):
    results = []
    for seed in seeds:
        trial = output / f"seed_{seed}"
        trial.mkdir(parents=True)
        command = [sys.executable, "-m", "aspire_so100.agent_loop", "--worker",
                   "--candidate", str(candidate), "--output", str(trial), "--seed", str(seed)]
        clean_env = {k: v for k, v in os.environ.items() if k in
                     {"PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "VIRTUAL_ENV"}}
        clean_env.update(HOME="/tmp", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MUJOCO_GL="egl")
        try:
            with (trial / "stdout.log").open("w") as stdout, (trial / "stderr.log").open("w") as stderr:
                process = subprocess.run(sandbox_command(command, trial), env=clean_env,
                                         stdout=stdout, stderr=stderr, timeout=timeout)
            if not (trial / "result.json").exists():
                message = (trial / "stderr.log").read_text()[-3000:]
                # Sandbox/runtime infrastructure failures must not become policy scores.
                raise RuntimeError(f"Trial did not produce a result (exit {process.returncode}): {message}")
            result = json.loads((trial / "result.json").read_text())
        except subprocess.TimeoutExpired:
            result = {"seed": seed, "error": "trial timeout", "metrics": {"success": False, "score": 0.}}
        trace = trial / "trace.json"
        if trace.exists():
            result["trace"] = str(trace)
        results.append(result)
    summary = {"successes": sum(r["metrics"]["success"] for r in results),
               "trials": len(results), "mean_score": sum(r["metrics"]["score"] for r in results) / len(results),
               "results": results}
    write_json(output / "results.json", summary)
    return summary


ACTOR = ("You are the ASPIRE robot-program actor. Return the requested JSON; do not edit files or run tools. "
         "Develop an executable hypothesis using the API below. Diagnose failures from the evidence. "
         "Reuse working code and revise the failing behavior. Keep the hypothesis concise. "
         "Do not change task success criteria.\n")


class BudgetStop(RuntimeError):
    pass


def select_visual_observations(history):
    from aspire_so100.context import select_prior
    paths, labels = [], []
    for entry in select_prior(history):
        result = next((r for r in entry.get("results", []) if r.get("visual_observations")), None)
        if result is None:
            continue
        for phase in ("initial", "outcome"):
            path = Path(result["visual_observations"][phase])
            if not path.is_file():
                raise FileNotFoundError(f"Missing visual observation: {path}")
            paths.append(str(path.resolve()))
            labels.append({"image":len(paths), "policy_round":Path(entry["candidate"]).parent.name,
                           "seed":result["seed"], "phase":phase})
    return paths, labels


def budgeted_call(prompt, directory, fields, args, state):
    image_paths, labels = select_visual_observations(state.get("history", []))
    prompt += "\nVisual observations: " + json.dumps(labels or "Initial scene will be attached; old attempts may have no images.")
    prompt += "\nEach sheet: side RGB, depth (white .25m / black 1m), wrist RGB. Use images with execution evidence to diagnose grasp/placement errors."
    if len(prompt) > args.max_prompt_chars:
        raise BudgetStop(f"Prompt exceeds --max-prompt-chars={args.max_prompt_chars}; no call made")
    calls = state.setdefault("llm_calls", [])
    if any(call.get("usage") is None for call in calls):
        raise BudgetStop("Previous model usage is unknown; stopping before another call")
    used = sum(call["usage"].get("input_tokens", 0) + call["usage"].get("output_tokens", 0) for call in calls)
    if args.max_reported_tokens and used >= args.max_reported_tokens:
        raise BudgetStop(f"Reported token threshold reached: {used} >= {args.max_reported_tokens}")
    record = {"directory": str(directory), "prompt_chars": len(prompt), "usage": None,
              "image_count": len(image_paths) or 1}
    calls.append(record)
    print(f"LLM: {len(prompt):,} prompt characters; {record['image_count']} image sheets; {used:,} tokens reported so far", flush=True)
    try:
        return ask_llm(prompt, directory, fields, args.model, args.llm_timeout, args.codex_bin, image_paths)
    finally:
        record["usage"] = read_usage(directory / "agent.jsonl")
        if directory.exists():
            write_json(directory / "usage.json", record)
        print(f"LLM usage: {record['usage'] or 'unknown'}", flush=True)
        write_json(args.output.resolve() / "campaign.json", state)


def update_best_policy(state, output):
    """Atomically refresh the stable best-policy path after every scored round."""
    if not state.get("history"):
        return None
    best = max(state["history"], key=lambda x: (x["successes"], x["mean_score"]))
    source = Path(best["candidate"])
    if not source.is_file():
        raise FileNotFoundError(f"Best candidate is missing: {source}")
    destination = output / "best_policy.py"
    temporary = output / ".best_policy.py.tmp"
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    state.update(best_candidate=str(source), best_policy=str(destination),
                 best_successes=best["successes"], best_mean_score=best["mean_score"])
    return best


def campaign(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = ROOT / "skills" / "learned"
    library.mkdir(parents=True, exist_ok=True)
    history = []
    source_campaign = None
    if args.continue_from:
        source_campaign = args.continue_from.resolve()
        previous = json.loads(source_campaign.read_text())
        history = previous.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError("--continue-from must contain a non-empty campaign history")
        for entry in history:
            candidate = Path(entry.get("candidate", ""))
            if not candidate.is_file():
                raise FileNotFoundError(f"Previous candidate is missing: {candidate}")
            if not all(key in entry for key in ("successes", "trials", "mean_score", "results")):
                raise ValueError("Previous campaign history is incomplete")
    start_round = len(history)
    state = {"status": "running", "backend": "codex exec", "model": args.model or "CLI configured default",
             "debug_seeds": DEBUG, "validation_seeds": VALIDATION,
             "heldout_seeds": "1–10 remain unused", "rounds": args.rounds,
             "round_mode": "until_pass_or_interrupt" if args.rounds == 0 else "bounded",
             "history": history,
             "context_mode": "compact", "max_prompt_chars": args.max_prompt_chars,
             "max_reported_tokens": args.max_reported_tokens, "llm_calls": []}
    if source_campaign:
        state.update(source_campaign=str(source_campaign), inherited_rounds=start_round)
    update_best_policy(state, output)
    write_json(output / "campaign.json", state)
    write_json(output / "skill_progress.json", build_report(state))
    try:
        round_indices = itertools.count(start_round) if args.rounds == 0 else range(start_round, start_round + args.rounds)
        for index in round_indices:
            if state["history"] and max(e["successes"] for e in state["history"]) == len(DEBUG):
                break
            round_dir = output / f"round_{index:03d}"
            prompt = build_prompt(ACTOR + API, state["history"], library, max(1, args.max_prompt_chars - 2048))
            response = budgeted_call(prompt, round_dir / "proposal", ("hypothesis", "code"), args, state)
            candidate = round_dir / "candidate.py"
            candidate.write_text(response["code"])
            try:
                validate_policy(response["code"])
            except (ValueError, SyntaxError) as exc:
                entry = {"candidate": str(candidate), "hypothesis": response["hypothesis"], "successes": 0,
                         "mean_score": 0., "trials": 0, "results": [], "generation_error": str(exc)}
            else:
                print(f"Round {index + 1}: testing LLM candidate", flush=True)
                entry = {"candidate": str(candidate), "hypothesis": response["hypothesis"],
                         **run_trials(candidate, DEBUG, round_dir / "debug", args.trial_timeout)}
            state["history"].append(entry)
            write_json(round_dir / "skill_update.json", build_report({"history": [entry]})["rounds"][0] | {"round": index + 1})
            best = update_best_policy(state, output)
            write_json(output / "campaign.json", state)
            write_json(output / "skill_progress.json", build_report(state))
            if "generation_error" in entry:
                print(f"Round {index + 1}: candidate rejected before simulation: {entry['generation_error']}", flush=True)
            else:
                print(f"Round {index + 1}: {entry['successes']}/{entry['trials']} passed", flush=True)
            if entry["successes"] == len(DEBUG):
                break
        best = update_best_policy(state, output)
        if best["successes"] == len(DEBUG):
            validation = run_trials(Path(best["candidate"]), VALIDATION, output / "validation", args.trial_timeout)
            state["validation"] = validation
            if validation["successes"] == len(VALIDATION):
                prefix = ("Act as ASPIRE skill curator. Distill only reusable lessons supported by this candidate's "
                          "debug and independent validation results. Do not claim broader generalization.\n" + API +
                          "\nValidation:\n" + json.dumps(validation))
                prompt = build_prompt(prefix, [best], library, max(1, args.max_prompt_chars - 2048))
                review = budgeted_call(prompt, output / "curator",
                                       ("failure_signature", "when_to_apply", "repair", "limitations"), args, state)
                digest = hashlib.sha256(Path(best["candidate"]).read_bytes()).hexdigest()
                review.update(policy_sha256=digest, evidence=str(output / "campaign.json"), debug_seeds=DEBUG, validation_seeds=VALIDATION)
                write_json(library / f"tower_{digest[:12]}.json", review)
                state["promoted_skill"] = str(library / f"tower_{digest[:12]}.json")
        state["status"] = "completed" if state.get("promoted_skill") else "budget_exhausted_or_validation_failed"
    except BudgetStop as exc:
        state.update(status="budget_stopped", error=str(exc))
        print(str(exc), flush=True)
    except KeyboardInterrupt:
        state.update(status="interrupted")
        raise
    except Exception as exc:
        state.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        update_best_policy(state, output)
        write_json(output / "campaign.json", state)
        write_json(output / "skill_progress.json", build_report(state))
    print(f"Campaign: {output / 'campaign.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3,
                        help="Additional proposals; 0 keeps proposing until 3/3 or interruption")
    parser.add_argument("--model", help="Optional Codex model; otherwise uses CLI configuration")
    parser.add_argument("--codex-bin", help="Codex executable; auto-detected from PATH or the VS Code extension")
    parser.add_argument("--llm-timeout", type=int, default=300)
    parser.add_argument("--trial-timeout", type=int, default=120)
    parser.add_argument("--max-prompt-chars", type=int, default=32000,
                        help="Maximum supplied prompt characters per call (not provider tokens)")
    parser.add_argument("--max-reported-tokens", type=int, default=100000,
                        help="Stop before the next call after this input+output token threshold; 0 disables it")
    parser.add_argument("--preview-from", type=Path,
                        help="Build the next compact prompt from an existing campaign.json; no LLM or simulation")
    parser.add_argument("--continue-from", type=Path,
                        help="Inherit policy history from a completed campaign; --rounds is the additional limit")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=51, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.candidate, args.output, args.seed)
    else:
        if args.rounds < 0 or args.max_prompt_chars < 1 or args.max_reported_tokens < 0:
            parser.error("rounds/token limit must be nonnegative and prompt limit must be positive")
        if args.preview_from and args.continue_from:
            parser.error("--preview-from and --continue-from cannot be used together")
        if args.preview_from:
            state = json.loads(args.preview_from.read_text())
            prompt = build_prompt(ACTOR + API, state["history"], ROOT / "skills" / "learned", args.max_prompt_chars)
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "prompt.txt").write_text(prompt)
            print(f"Offline preview: {len(prompt):,} characters; no model called. {args.output / 'prompt.txt'}")
            return
        campaign(args)


if __name__ == "__main__":
    main()
