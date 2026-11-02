"""Actor rows from the accepted decision-level skills as whole trajectories: for each accepted decision-level skill, the trajectory that
succeeded under it in the continuation verification - the agent's own steps 0..k-1, then from the failure step k the
policy's steps with the decision-level skill in context, each row with the plain prompt (no skill block) - from
<dir>/prescription_continuations.jsonl (scripts.stage2.verify). Trajectories are taken round robin over failure types
(random within a type, seed) until the rows reach --rows; 5% of the questions go to val. Written together with the
own-trajectory replay rows of --own (source "own" / "corrected") as one replay set for scripts.stage2.build_sft
--replay_dir (--replay = its train rows).
    python -m scripts.stage2.corrected --dir data/tkgqa/stage2/MultiTQ --own data/tkgqa/sft/own_replay/MultiTQ \\
        --rows 1000000 --out data/tkgqa/sft/replay/MultiTQ
"""
import argparse
import collections
import json
import os
import random


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--own", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, required=True); ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    C = [json.loads(l) for l in open(os.path.join(a.dir, "prescription_continuations.jsonl"), encoding="utf-8")]
    rng = random.Random(a.seed)
    uids = sorted({c["uid"] for c in C}); rng.shuffle(uids)
    val_u = set(uids[: max(1, int(len(uids) * a.val_frac))])
    stats = {}
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "val"):
        pool = [c for c in C if (c["uid"] in val_u) == (split == "val")]
        cap = a.rows if split == "train" else max(1, round(a.rows * a.val_frac))
        by = collections.defaultdict(list)
        for c in sorted(pool, key=lambda c: (c["uid"], c["rollout"])):
            by[c["failure_type"]].append(c)
        for v in by.values():
            rng.shuffle(v)
        chosen, n = [], 0
        while n < cap and any(by.values()):
            for t in sorted(by):
                if by[t] and n < cap:
                    c = by[t].pop(); chosen.append(c); n += len(c["prefix"]) + len(c["continuation"])
        rows = [{"prompt": p, "response": r, "uid": c["uid"], "qtype": c["qtype"], "step": j, "program": -1, "source": "corrected",
                 "prompt_version": c["prompt_version"]}
                for c in chosen for j, (p, r) in enumerate(c["prefix"] + c["continuation"])]
        own = pd.read_parquet(os.path.join(a.own, f"{split}.parquet")); own["source"] = "own"
        df = pd.concat([own, pd.DataFrame(rows)], ignore_index=True)
        df.to_parquet(os.path.join(a.out, f"{split}.parquet"), index=False)
        ops = collections.Counter(r.split("=", 1)[-1].strip().split("(")[0].replace("<action>", "").strip()
                                  for c in chosen for r in [c["continuation"][0][1]] if c["continuation"])
        stats[split] = {"own": len(own), "corrected_rows": len(rows), "corrected_trajectories": len(chosen), "available": len(pool),
                        "by_failure_type": dict(collections.Counter(c["failure_type"] for c in chosen).most_common()),
                        "by_qtype": dict(collections.Counter(c["qtype"] for c in chosen).most_common()),
                        "op_at_the_failure_step": dict(ops.most_common())}
    m = {"dir": a.dir, "own": a.own, "rows_cap": a.rows, "splits": stats, "seed": a.seed}
    json.dump(m, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
