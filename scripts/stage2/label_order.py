"""The labelling order of a Stage 2 collection: distinct trajectories (uid + action sequence, as scripts.stage2.label),
N_SUCC successful ones first, then every failed one; both type-stratified - round robin over the question types, random
within a type (seed) - so a budget cut keeps the types balanced.
    python -m scripts.stage2.label_order data/tkgqa/stage2/MultiTQ [--n_succ 700] [--seed 2026]
writes <dir>/analyze_order.jsonl
"""
import argparse
import collections
import glob
import json
import os
import random


def rr(pool, rng, n=None):
    by = collections.defaultdict(list)
    for line, t in pool:
        by[t["qtype"]].append(line)
    for v in by.values():
        rng.shuffle(v)
    out, types = [], sorted(by)
    while any(by.values()) and (n is None or len(out) < n):
        for k in types:
            if by[k] and (n is None or len(out) < n):
                out.append(by[k].pop())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--n_succ", type=int, default=700)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    seen, succ, fail = set(), [], []
    for p in sorted(glob.glob(os.path.join(args.dir, "trajectories.shard*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            t = json.loads(line)
            key = (t["uid"], tuple(a or "" for a in t["actions"]))
            if key in seen:
                continue
            seen.add(key)
            (succ if t["success"] >= 1.0 else fail).append((line if line.endswith("\n") else line + "\n", t))
    rng = random.Random(args.seed)
    order = rr(succ, rng, args.n_succ) + rr(fail, rng)
    with open(os.path.join(args.dir, "analyze_order.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(order)
    s = collections.Counter(json.loads(l)["qtype"] for l in order[:min(args.n_succ, len(succ))])
    print(f"distinct trajectories {len(seen)}: successes {len(succ)} (taken {min(args.n_succ, len(succ))}: {dict(sorted(s.items()))}), "
          f"failures {len(fail)}; order {len(order)} -> {args.dir}/analyze_order.jsonl")


if __name__ == "__main__":
    main()
