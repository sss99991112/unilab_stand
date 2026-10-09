"""Compare saved recovery actors on identical seeded resets; no training or video."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.deploy.diagnose_unilab_g1_recovery_policy import run_branch

from unilab.base import registry


def summarize(episodes):
    if not episodes or not all(row["full_episode"] for row in episodes):
        raise ValueError("checkpoint comparison requires complete episodes")
    if any(row["assistance_fraction"] != 0 or row["assisted_steps"] for row in episodes):
        raise ValueError("checkpoint comparison must be unassisted")
    successes = [row for row in episodes if row["success"]]
    times = [row["first_success_step"] * row["ctrl_dt_seconds"] for row in successes]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "mean_longest_stable_seconds": float(
            np.mean([row["longest_stable_seconds"] for row in episodes])
        ),
        "max_longest_stable_seconds": max(row["longest_stable_seconds"] for row in episodes),
        "successful_and_stable_at_end": sum(
            row["success"] and row["stable_at_episode_end"] for row in episodes
        ),
        "mean_success_time_seconds": float(np.mean(times)) if times else None,
        "mean_max_base_height_m": float(np.mean([row["max_base_height_m"] for row in episodes])),
        "mean_episode_reward": float(np.mean([row["episode_reward"] for row in episodes])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iterations", nargs="+", type=int, default=[3000, 6000, 10000])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 32 or args.seed < 0:
        parser.error("require 1..32 episodes and nonnegative seed")
    if (
        not 1 <= len(args.iterations) <= 8
        or len(set(args.iterations)) != len(args.iterations)
        or min(args.iterations) < 0
    ):
        parser.error("require 1..8 distinct nonnegative checkpoint iterations")
    if args.output.exists():
        parser.error("output already exists; choose a new result filename")
    run_dir = args.run_dir.resolve(strict=True)
    paths = [
        (run_dir / f"model_{iteration}.pt").resolve(strict=True) for iteration in args.iterations
    ]
    saved_config = json.loads((run_dir / "run_config.json").read_text())["config"]
    seeds = list(range(args.seed, args.seed + args.episodes))
    initial_by_seed = {}
    models = []
    registry.ensure_registries()
    torch.set_num_threads(1)
    for path in paths:
        checkpoint = torch.load(path, map_location=args.device, weights_only=True)
        episodes = []
        for seed in seeds:
            initial, actor_obs, row = run_branch(
                saved_config,
                checkpoint,
                assisted=False,
                seed=seed,
                device=args.device,
                steps=0,
            )
            if seed in initial_by_seed:
                first_state, first_obs = initial_by_seed[seed]
                np.testing.assert_array_equal(first_state, initial)
                np.testing.assert_array_equal(first_obs, actor_obs)
            else:
                initial_by_seed[seed] = (initial, actor_obs)
            episodes.append(row)
            print(
                f"{path.name}: seed={seed} success={row['success']} "
                f"longest_stable={row['longest_stable_seconds']:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        models.append(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "summary": summarize(episodes),
                "episodes": episodes,
            }
        )
        del checkpoint
    highest_rate = max(model["summary"]["success_rate"] for model in models)
    report = {
        "status": "COMPARISON_COMPLETED",
        "run_dir": str(run_dir),
        "seeds": seeds,
        "same_initial_states_and_actor_observations": True,
        "highest_success_rate": highest_rate,
        "checkpoints_with_highest_success_rate": [
            model["checkpoint"]
            for model in models
            if model["summary"]["success_rate"] == highest_rate
        ],
        "scope": "unassisted deterministic-policy evaluation; success uses task stability criterion; no training; reward is reference only",
        "models": models,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for model in models:
        stats = model["summary"]
        print(
            f"{Path(model['checkpoint']).name}: success={stats['successes']}/{stats['episodes']} "
            f"({stats['success_rate']:.1%}), mean_longest_stable={stats['mean_longest_stable_seconds']:.2f}s, "
            f"successful_and_stable_at_end={stats['successful_and_stable_at_end']}/{stats['episodes']}, "
            f"mean_max_height={stats['mean_max_base_height_m']:.3f}m"
        )
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
