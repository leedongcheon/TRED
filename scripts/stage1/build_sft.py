"""Build the Stage 1 (Temporal-Operation SFT) set from the teacher runs (scripts/stage1/teacher.py).

Keeps passing attempts only (F1 = 1 and the teacher filter: invalid actions and failed FINDs in between are allowed,
the same invalid error twice is not), one trajectory per question (the shortest; a FIND-retry trajectory may be
preferred to reach the retry quota), with a cap per question type. Every step of a trajectory becomes one example,
except invalid steps (they stay in the history of the later steps, whose targets show the recovery): prompt = the
actor prompt at that step (no few-shot block), response = "<action> OP </action>" (offline.normalize_response); loss
applies to the response only (verl SFT trainer). Train/val are split by question. Every chosen trajectory is replayed
in a fresh TKGQAEnvironmentManager and its prompts must match the stored ones exactly.

    python -m scripts.stage1.build_sft data/tkgqa/stage1/MultiTQ/pilot.jsonl data/tkgqa/stage1/MultiTQ/main.jsonl \\
        --out_dir data/tkgqa/sft/stage1_raw/MultiTQ
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd

from agent_system.environments.env_package.tkgqa.offline import env_kwargs, make_manager, normalize_response, prompt_version
from scripts.common import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_trajectories", type=int, default=3000)
    parser.add_argument("--retry_min", type=float, default=0.10)
    parser.add_argument("--retry_max", type=float, default=0.20)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fill_beyond_cap", action="store_true",
                        help="when some question types run out, fill the rest from other types (off: the per-type cap holds)")
    parser.add_argument("--no_verify", action="store_true", help="skip the replay check of the stored prompts")
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "datasets"))
    parser.add_argument("--store_root", default=os.path.join(REPO_ROOT, "stores"))
    parser.add_argument("--device", default="cpu", help="device for the bge-m3 grounder used by the replay check")
    return parser.parse_args()


def verify_prompts(chosen: Dict[str, Dict], dataset: str, args) -> int:
    """Replay every chosen trajectory's responses from scratch; the number whose prompts differ."""
    from tkgqa.grounder import Grounder
    from tkgqa.kg import TemporalKG

    kg = TemporalKG.load(dataset, args.data_root)
    grounder = Grounder.from_store_root(kg, args.store_root, device=args.device)
    bad = 0
    for uid, r in chosen.items():
        manager = make_manager(kg, grounder, 1, max_steps=r["max_steps"])
        obs, _ = manager.reset([env_kwargs(r)])
        prompts = [obs["text"][0]]
        for step in r["steps"][:-1]:
            prompts.append(manager.step([step["response"]])[0]["text"][0])
        if prompts != r["prompts"]:
            bad += 1
            print(f"replay mismatch: {uid}", file=sys.stderr)
    return bad


def cost(record: Dict) -> tuple:
    return (record["num_steps"], record["usage"]["completion_tokens"], record["attempt"])


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    by_uid: Dict[str, List[Dict]] = defaultdict(list)
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["passed"] and r.get("prompts"):
                    by_uid[r["uid"]].append(r)
    if not by_uid:
        print("no passing trajectories", file=sys.stderr)
        return 1
    versions = {(r.get("prompt_version"), r.get("max_steps")) for rs in by_uid.values() for r in rs}
    if len(versions) != 1:
        print(f"inputs mix prompt versions / max steps: {sorted(map(str, versions))}", file=sys.stderr)
        return 1
    if next(iter(versions))[0] != prompt_version():
        print(f"inputs were generated with prompt version {next(iter(versions))[0]}, the current one is "
              f"{prompt_version()}", file=sys.stderr)
        return 1

    shortest = {uid: min(rs, key=cost) for uid, rs in by_uid.items()}
    shortest_retry = {uid: min(retry, key=cost) for uid, rs in by_uid.items() if (retry := [r for r in rs if r["find_retry"]])}

    by_type: Dict[str, List[str]] = defaultdict(list)
    for uid, r in shortest.items():
        by_type[r["qtype"]].append(uid)
    for uids in by_type.values():
        rng.shuffle(uids)
    target = min(args.max_trajectories, len(shortest))
    cap = math.ceil(target / len(by_type))
    chosen: Dict[str, Dict] = {}
    per_type: Counter = Counter()

    def fill(pool_of, limit):
        progress = True
        while len(chosen) < limit and progress:
            progress = False
            for qtype, uids in sorted(by_type.items()):
                if per_type[qtype] >= cap or len(chosen) >= limit:
                    continue
                for uid in uids:
                    if uid not in chosen and (record := pool_of(uid)) is not None:
                        chosen[uid] = record
                        per_type[qtype] += 1
                        progress = True
                        break

    fill(shortest_retry.get, math.floor(args.retry_max * target))  # retry trajectories first, up to the max share
    fill(shortest.get, target)
    if len(chosen) < target and args.fill_beyond_cap:  # some types ran out under the cap; fill from the rest
        cap = target
        fill(shortest.get, target)

    retry_share = sum(r["find_retry"] for r in chosen.values()) / len(chosen)
    if retry_share < args.retry_min:
        print(f"warning: FIND-retry share {retry_share:.1%} is below {args.retry_min:.0%}", file=sys.stderr)

    if not args.no_verify:
        datasets = {r["dataset"] for r in chosen.values()}
        assert len(datasets) == 1, datasets
        bad = verify_prompts(chosen, next(iter(datasets)), args)
        if bad:
            print(f"{bad} of {len(chosen)} trajectories do not replay to their stored prompts", file=sys.stderr)
            return 1
        print(f"replay check: {len(chosen)} trajectories reproduce their prompts", file=sys.stderr)

    uids = sorted(chosen)
    rng.shuffle(uids)
    n_val = max(1, int(len(uids) * args.val_frac)) if len(uids) > 1 else 0
    val_uids = set(uids[:n_val])
    rows = {"train": [], "val": []}
    for uid, r in chosen.items():
        split = "val" if uid in val_uids else "train"
        for k, (step, prompt) in enumerate(zip(r["steps"], r["prompts"])):
            if step["status"] == "invalid":  # not a target; the next step's prompt shows it and the fix
                continue
            rows[split].append({
                "prompt": prompt,
                "response": normalize_response(step["response"]),
                "uid": uid,
                "dataset": r["dataset"],
                "qtype": r["qtype"],
                "step": k,
                "num_steps": r["num_steps"],
                "find_retry": r["find_retry"],
                "prompt_version": r["prompt_version"],
                "fewshot_version": r["fewshot_version"],
                "attempt": r["attempt"],
                "success": float(r["score"]["success"]),
            })

    os.makedirs(args.out_dir, exist_ok=True)
    for split, data in rows.items():
        pd.DataFrame(data).to_parquet(os.path.join(args.out_dir, f"{split}.parquet"), index=False)
    manifest = {
        "inputs": args.inputs,
        "prompt_version": next(iter(versions))[0],
        "max_steps": next(iter(versions))[1],
        "questions_with_pass": len(shortest),
        "trajectories": len(chosen),
        "val_trajectories": len(val_uids),
        "examples": {k: len(v) for k, v in rows.items()},
        "find_retry_share": retry_share,
        "fewshot_version": sorted({r["fewshot_version"] for r in chosen.values()}),
        "attempt": dict(sorted(Counter(r["attempt"] for r in chosen.values()).items())),
        "with_invalid_steps": sum(r.get("invalid_steps", 0) > 0 for r in chosen.values()) / len(chosen),
        "teacher_mean_steps_per_qtype": {t: sum(r["num_steps"] for r in chosen.values() if r["qtype"] == t)
                                         / sum(r["qtype"] == t for r in chosen.values())
                                         for t in sorted({r["qtype"] for r in chosen.values()})},
        "qtype": dict(Counter(r["qtype"] for r in chosen.values()).most_common()),
        "num_steps": dict(sorted(Counter(r["num_steps"] for r in chosen.values()).items())),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
