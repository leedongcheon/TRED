"""RL data (verl RLHFDataset parquet) in the Search layout: the question goes to the environment through
env_kwargs; the actor prompt itself is built by TKGQAEnvironmentManager at every step.

    python -m scripts.data.make_rl_data --dataset MultiTQ     # rl.jsonl -> train, heldout.jsonl -> val
"""

import argparse
import json
import os
import sys

import pandas as pd

from agent_system.environments.env_package.tkgqa.offline import env_kwargs
from scripts.common import REPO_ROOT


def rows(path: str):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            q = json.loads(line)
            yield {
                "data_source": q["dataset"],
                "prompt": [{"role": "user", "content": q["question"]}],  # placeholder; the env builds the prompt
                "env_kwargs": env_kwargs(q),
                "extra_info": {"index": i, "uid": q["uid"], "qtype": q["qtype"]},
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", default="rl.jsonl")
    parser.add_argument("--val", default="heldout.jsonl")
    parser.add_argument("--split_dir", default=None, help="default: data/tkgqa/splits/<dataset>")
    parser.add_argument("--out_dir", default=None, help="default: data/tkgqa/rl/<dataset>")
    args = parser.parse_args()
    split_dir = args.split_dir or os.path.join(REPO_ROOT, "data", "tkgqa", "splits", args.dataset)
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "data", "tkgqa", "rl", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    for name, split in (("train", args.train), ("val", args.val)):
        df = pd.DataFrame(list(rows(os.path.join(split_dir, split))))
        df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
        print(f"{name}: {len(df)} questions from {split} -> {out_dir}/{name}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
