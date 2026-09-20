from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from aspire_so100.stacking_env import StackingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = PROJECT_ROOT / "policies" / "stack_small_red_on_blue.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stack_small_red_on_blue"
SEED_SPLITS = {
    "debug": tuple(range(51, 56)),
    "heldout": tuple(range(1, 11)),
}


def _load_candidate(path: Path) -> tuple[str, Callable[[StackingEnv], None]]:
    module_name = f"aspire_candidate_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise TypeError(f"{path} must define run(api)")
    return path.stem, run


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            seeds.extend(range(start, end + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return tuple(seeds)


def evaluate_candidate(
    candidate_path: Path,
    seeds: tuple[int, ...],
    output_dir: Path,
    render: bool,
) -> dict:
    candidate_name, candidate = _load_candidate(candidate_path)
    candidate_dir = output_dir / candidate_name
    results = []
    env = StackingEnv(candidate_dir, render=render)
    try:
        for seed in seeds:
            seed_dir = candidate_dir / f"seed_{seed:04d}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            env.output_dir = seed_dir
            env.reset(seed)
            error = None
            try:
                candidate(env)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            env.wait(0.8)
            metrics = env.evaluate()
            trace_path = env.write_trace(metrics, error=error)
            results.append(
                {
                    "seed": seed,
                    "error": error,
                    "metrics": asdict(metrics),
                    "trace": str(trace_path.relative_to(output_dir)),
                }
            )
    finally:
        env.close()

    successes = sum(result["metrics"]["success"] for result in results)
    mean_score = sum(result["metrics"]["score"] for result in results) / len(results)
    return {
        "candidate": candidate_name,
        "candidate_path": str(candidate_path.resolve()),
        "successes": successes,
        "trials": len(results),
        "success_rate": successes / len(results),
        "mean_score": mean_score,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ASPIRE-style code-as-policy candidates on SO-100 stacking."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=Path,
        help="Python file defining run(api); repeat to compare candidates.",
    )
    parser.add_argument("--split", choices=SEED_SPLITS, default="debug")
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        help="Explicit comma/range list, for example 51-55 or 51,53,55.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--render", action="store_true", help="Save pre/post RGB frames.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidates = args.candidate or [DEFAULT_CANDIDATE]
    seeds = args.seeds or SEED_SPLITS[args.split]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{args.split}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "task": "stack small_red_block on blue_block",
        "split": args.split,
        "seeds": seeds,
        "render": args.render,
        "created_at": timestamp,
        "host": os.uname().nodename,
        "candidates": [],
    }
    for candidate_path in candidates:
        print(f"Evaluating {candidate_path} on {len(seeds)} {args.split} seed(s)...")
        result = evaluate_candidate(candidate_path, seeds, output_dir, args.render)
        manifest["candidates"].append(result)
        print(
            f"  {result['candidate']}: {result['successes']}/{result['trials']} "
            f"success, mean score={result['mean_score']:.3f}"
        )

    manifest["candidates"].sort(
        key=lambda item: (item["success_rate"], item["mean_score"]), reverse=True
    )
    manifest["best_candidate"] = manifest["candidates"][0]["candidate"]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
