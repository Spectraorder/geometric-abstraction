"""Local, bounded selection of robot execution evidence for policy repair.

Inspired by GPT-Policy's selective context construction; no upstream code is
copied. Full precision traces stay on disk. This module makes no model calls.
"""
from collections import Counter
import json
import math
from pathlib import Path


def compact(value):
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    return value


def encode(value):
    return json.dumps(compact(value), separators=(",", ":"), ensure_ascii=False)


def snapshot(state):
    return {key: state[key] for key in
            ("end_effector_position", "robot_joint_positions", "objects") if key in state}


def event_score(event):
    """Prefer errors and physical object transitions over repeated observations."""
    if event.get("error"):
        return 1000.
    before = event.get("before", {}).get("objects", {})
    after = event.get("after", {}).get("objects", {})
    score = 0.
    for name in before.keys() & after.keys():
        p, q = before[name].get("position", []), after[name].get("position", [])
        score += math.dist(p, q) if len(p) == len(q) else 0.
        a = before[name].get("quaternion_wxyz", [])
        b = after[name].get("quaternion_wxyz", [])
        if len(a) == len(b) == 4:
            score += min(math.dist(a, b), math.dist(a, [-x for x in b])) * .1
    return score


def summarize_trace(trace, event_limit=6):
    primitives = trace.get("primitives", [])
    summary = {"primitive_count": len(primitives),
               "primitive_counts": dict(Counter(p["primitive"] for p in primitives)),
               "checks": trace.get("stability_checks", [])[-8:]}
    if not primitives:
        return summary
    summary["initial_state"] = snapshot(primitives[0].get("before", {}))
    summary["final_state"] = snapshot(primitives[-1].get("after", {}))
    ranked = sorted(range(len(primitives)), key=lambda i: event_score(primitives[i]), reverse=True)
    selected = sorted(ranked[:event_limit])
    events = []
    for i in selected:
        p = primitives[i]
        before, after = p.get("before", {}), p.get("after", {})
        changed = {name: {"before": state, "after": after.get("objects", {}).get(name)}
                   for name, state in before.get("objects", {}).items()
                   if event_score({"before": {"objects": {name: state}},
                                   "after": {"objects": {name: after.get("objects", {}).get(name, {})}}}) > .002}
        events.append({"index": i, "primitive": p["primitive"], "inputs": p.get("inputs"),
                       "error": p.get("error"), "changed_objects": changed,
                       "ee_before": before.get("end_effector_position"),
                       "ee_after": after.get("end_effector_position")})
    summary["selected_events"] = events
    summary["omitted_events"] = len(primitives) - len(selected)
    return compact(summary)


def feedback(entry, event_limit=6):
    value = {k: entry[k] for k in ("hypothesis", "successes", "trials", "mean_score", "generation_error") if k in entry}
    # Keep full source: truncating a policy would make the repair ambiguous.
    value["code"] = Path(entry["candidate"]).read_text()
    groups = {}
    for result in entry.get("results", []):
        trace = json.loads(Path(result["trace"]).read_text()) if result.get("trace") else {}
        reasons = sorted({r for c in trace.get("stability_checks", []) for r in c.get("reasons", [])})
        # Group only matching outcomes; differences across seeds remain visible.
        signature = encode([result.get("error"), result.get("metrics"), reasons])
        if signature not in groups:
            groups[signature] = {"seeds": [], "metrics": result.get("metrics"),
                                 "error": result.get("error"), "reasons": reasons,
                                 "representative_seed": result["seed"],
                                 "trace": summarize_trace(trace, event_limit)}
        groups[signature]["seeds"].append(result["seed"])
    value["outcome_groups"] = list(groups.values())
    return value


def select_prior(history):
    if not history:
        return []
    # Include the newest error even when its score is lower than older policies.
    best = max(history, key=lambda x: (x["successes"], x["mean_score"]))
    latest = history[-1]
    return [latest] if best["candidate"] == latest["candidate"] else [best, latest]


def build_prompt(prefix, history, library, max_chars=32000):
    prior = select_prior(history)
    # This library is currently tower-only. Bound memory growth to three skills.
    memory = []
    for path in sorted(library.glob("tower_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        skill = json.loads(path.read_text())
        lesson = {k: skill[k] for k in ("failure_signature", "when_to_apply", "repair", "limitations") if k in skill}
        if len(encode(memory + [lesson])) <= 4000:
            memory.append(lesson)
        if len(memory) == 3:
            break
    for event_limit in (6, 3, 0):
        payload = {"validated_skills": memory, "previous_attempts": [feedback(e, event_limit) for e in prior],
                   "context_notes": "Rounded to 4 decimals. Matching seed outcomes grouped; representative transitions selected. "
                                    "Unshown intermediate events may differ. Full traces retained locally. "
                                    "Code is complete. Selected events are evidence, not proof of a successful grasp."}
        prompt = prefix + "\nExecution context:\n" + encode(payload)
        if len(prompt) <= max_chars:
            return prompt
    raise ValueError(f"Context exceeds {max_chars} characters even without intermediate events. "
                     "No model call made; increase --max-prompt-chars to include the complete policies.")


def read_usage(path):
    """Read observed Codex completion counters; absence is unknown, never zero."""
    turns = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
                if not isinstance(usage, dict) or not all(
                    isinstance(usage.get(k), int) and usage[k] >= 0
                    for k in ("input_tokens", "output_tokens")
                ):
                    return None
                turns.append(usage)
    if not turns:
        return None
    keys = {k for usage in turns for k in usage if k.endswith("tokens")}
    return {k: sum(u.get(k, 0) for u in turns) for k in keys}
