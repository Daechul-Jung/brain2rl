"""
download_d4rl.py — Download D4RL offline RL datasets.

D4RL is maintained by Farama Foundation:
  https://github.com/Farama-Foundation/D4RL

Install:
  pip install d4rl   # or: pip install "d4rl @ git+https://github.com/Farama-Foundation/D4RL"

Usage:
  python scripts/download_d4rl.py                        # download defaults
  python scripts/download_d4rl.py --datasets hopper-medium-v2 ant-medium-v2
  python scripts/download_d4rl.py --list                 # list all available datasets

Downloaded datasets are cached to ~/.d4rl/datasets/ automatically by the d4rl library.
"""

from __future__ import annotations

import argparse
import sys


DEFAULT_DATASETS = [
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "walker2d-medium-v2",
    "halfcheetah-medium-replay-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-replay-v2",
    "halfcheetah-expert-v2",
    "hopper-expert-v2",
]

ALL_DATASETS = DEFAULT_DATASETS + [
    "ant-medium-v2",
    "ant-medium-replay-v2",
    "antmaze-umaze-v0",
    "antmaze-medium-play-v0",
    "kitchen-partial-v0",
    "kitchen-complete-v0",
    "pen-human-v1",
    "hammer-human-v1",
    "door-human-v1",
    "relocate-human-v1",
]


def download(dataset_id: str) -> None:
    try:
        import d4rl  # noqa: F401
        import gym as old_gym
    except ImportError:
        print(
            "[error] d4rl is not installed.\n"
            "Install with: pip install d4rl\n"
            "  or: pip install 'git+https://github.com/Farama-Foundation/D4RL'\n"
            "  (requires mujoco_py or mujoco>=2.1 with dm_control)",
            file=sys.stderr,
        )
        sys.exit(1)

    env_name = dataset_id.rsplit("-", 1)[0]  # e.g. "halfcheetah-medium-v2" → "halfcheetah-medium-v2"
    print(f"[download] {dataset_id} ...", end=" ", flush=True)
    try:
        env = old_gym.make(dataset_id)
        dataset = env.get_dataset()  # triggers download
        n = len(dataset["observations"])
        print(f"OK  ({n:,} transitions)")
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Download D4RL offline RL datasets")
    p.add_argument(
        "--datasets", nargs="+", default=None,
        metavar="DATASET_ID",
        help="Specific dataset IDs to download (default: download all defaults)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Download all known D4RL datasets (including robot dexterous hand tasks)",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print available dataset IDs and exit",
    )
    args = p.parse_args()

    if args.list:
        print("Default datasets:")
        for d in DEFAULT_DATASETS:
            print(f"  {d}")
        print("\nAll datasets (--all):")
        for d in ALL_DATASETS:
            print(f"  {d}")
        return

    if args.all:
        targets = ALL_DATASETS
    elif args.datasets:
        targets = args.datasets
    else:
        targets = DEFAULT_DATASETS

    print(f"Downloading {len(targets)} dataset(s) ...\n")
    for ds in targets:
        download(ds)
    print("\nDone. Datasets are cached in ~/.d4rl/datasets/")


if __name__ == "__main__":
    main()
