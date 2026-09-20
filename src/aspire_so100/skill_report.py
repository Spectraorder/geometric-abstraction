"""Show proposed and demonstrated robot capabilities for each learning round."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


PAIR = ["blue_block", "small_red_block"]


def round_update(entry, round_number):
    reasons = Counter()
    stable_pair_seeds = []
    max_placements = 0
    placement_seeds = []
    runtime_errors = []
    for result in entry.get("results", []):
        seed = result.get("seed")
        metrics = result.get("metrics", {})
        placed = int(metrics.get("placed_blocks", 0))
        if placed > max_placements:
            max_placements, placement_seeds = placed, [seed]
        elif placed and placed == max_placements:
            placement_seeds.append(seed)
        if result.get("error"):
            runtime_errors.append({"seed": seed, "error": result["error"]})
        trace_path = result.get("trace")
        if not trace_path or not Path(trace_path).is_file():
            continue
        trace = json.loads(Path(trace_path).read_text())
        seen = set()
        for check in trace.get("stability_checks", []):
            seen.update(check.get("reasons", []))
            if check.get("stable") and check.get("blocks") == PAIR and check.get("duration_s", 0) >= 2:
                stable_pair_seeds.append(seed)
        reasons.update(seen)
    if entry.get("generation_error"):
        status = "rejected"
    elif entry.get("trials") and entry.get("successes") == entry.get("trials"):
        status = "debug_passed"
    elif max_placements or stable_pair_seeds:
        status = "partial"
    else:
        status = "failed"
    demonstrated = []
    if stable_pair_seeds:
        demonstrated.append({"capability": "stable_red_on_blue_2s", "seeds": sorted(set(stable_pair_seeds))})
    if entry.get("successes"):
        successful = [r["seed"] for r in entry.get("results", []) if r.get("metrics", {}).get("success")]
        demonstrated.append({"capability": "complete_stable_tower", "seeds": successful})
    return {
        "round": round_number,
        "candidate_status": status,
        "proposed_update": entry.get("hypothesis", ""),
        "debug_passes": entry.get("successes", 0),
        "debug_trials": entry.get("trials", 0),
        "mean_score": entry.get("mean_score", 0),
        "max_placements": max_placements,
        "max_placement_seeds": sorted(set(placement_seeds)),
        "demonstrated_capabilities": demonstrated,
        "top_remaining_failures": [{"reason": reason, "seeds": count}
                                   for reason, count in reasons.most_common(6)],
        "generation_error": entry.get("generation_error"),
        "runtime_errors": runtime_errors,
        "candidate": entry.get("candidate"),
    }


def build_report(campaign):
    return {
        "campaign_status": campaign.get("status"),
        "validated_skill_promoted": campaign.get("promoted_skill"),
        "rounds": [round_update(entry, number) for number, entry in enumerate(campaign.get("history", []), 1)],
    }


def markdown(report):
    lines = [f"Campaign: {report['campaign_status']}",
             f"Validated skill promoted: {report['validated_skill_promoted'] or 'no'}", ""]
    for item in report["rounds"]:
        lines.extend([
            f"Round {item['round']} — {item['candidate_status']} "
            f"({item['debug_passes']}/{item['debug_trials']} full passes, score {item['mean_score']:.3f})",
            f"  Proposed: {item['proposed_update']}",
        ])
        if item["demonstrated_capabilities"]:
            shown = ", ".join(f"{x['capability']} on seeds {x['seeds']}" for x in item["demonstrated_capabilities"])
            lines.append(f"  Demonstrated: {shown}")
        else:
            lines.append("  Demonstrated: no stable placement passed the evaluator")
        if item["top_remaining_failures"]:
            failures = "; ".join(f"{x['reason']} ({x['seeds']} seeds)" for x in item["top_remaining_failures"])
            lines.append(f"  Remaining: {failures}")
        if item["generation_error"]:
            lines.append(f"  Rejected: {item['generation_error']}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    report = build_report(json.loads(args.campaign.read_text()))
    print(json.dumps(report, indent=2) if args.json else markdown(report))


if __name__ == "__main__":
    main()
