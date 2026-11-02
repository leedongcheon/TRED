"""Stage 2 replay rows from the policy's own correct trajectories (the Stage 2 collection, not the teacher). Pool: the
collection's successful trajectories (F1 = 1) without an invalid action; per question, one trajectory per distinct
operation skeleton (quoted strings blanked, result-variable numbers dropped; the shortest). Selection of --n step
examples, in pairs: a question with >= 2 distinct correct trajectories, two of them, and the step where they first
differ, from both (nearly the same state with two different correct actions: the replay keeps the decision open instead
of setting one trajectory per question). 5% of the questions go to val (the same procedure, n * 5% rows). Columns as
the Stage 1 SFT sets (prompt, response, uid, qtype, step, program, source="own", prompt_version), for
scripts.stage2.corrected --own.
    python -m scripts.stage2.own_replay --dir data/tkgqa/stage2/MultiTQ --n 131 --out data/tkgqa/sft/own_replay/MultiTQ
"""
import argparse
import collections
import glob
import json
import os
import random
import re


from scripts.common import REPO_ROOT as REPO


def skeleton(actions):
    return tuple(re.sub(r"\b([fts])\d+\b", r"\1", re.sub(r'"[^"]*"', '"_"', a or "")) for a in actions)


def pick(progs, uids, n, rng):
    """pairs: questions with >= 2 distinct correct trajectories, in random order; for each, two of them (random) and
    the step where their skeletons first differ - the decision on which both are correct - taken from both (two rows,
    nearly the same state, two different correct actions). Pairs where one trajectory is a prefix of the other are skipped."""
    rows = []
    for u in uids:
        if len(rows) + 2 > n:
            break
        ps = progs[u]
        if len(ps) < 2:
            continue
        a, b = rng.sample(range(len(ps)), 2)
        ka, kb = ps[a][3], ps[b][3]
        d = next((i for i in range(min(len(ka), len(kb))) if ka[i] != kb[i]), None)
        if d is None:
            continue
        rows += [(u, a, d, ps[a]), (u, b, d, ps[b])]
    return rows


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    progs = collections.defaultdict(dict)  # uid -> skeleton -> (qtype, [(prompt, response)], prompt_version)
    n_succ = 0
    for p in sorted(glob.glob(os.path.join(REPO, args.dir, "trajectories.shard*.jsonl"))):
        for t in map(json.loads, open(p, encoding="utf-8")):
            if t["success"] < 1.0 or any(s.get("status") == "invalid" for s in t["steps"]):
                continue
            n_succ += 1
            key, ex = skeleton(t["actions"]), [(s["prompt"], s["response"]) for s in t["steps"]]
            cur = progs[t["uid"]].get(key)
            if cur is None or len(ex) < len(cur[1]):
                progs[t["uid"]][key] = (t["qtype"], ex, t["prompt_version"], key)
    progs = {u: list(v.values()) for u, v in progs.items()}
    rng = random.Random(args.seed)
    uids = sorted(progs); rng.shuffle(uids)
    nval = max(1, int(len(uids) * args.val_frac))
    out = os.path.join(REPO, args.out); os.makedirs(out, exist_ok=True)
    stats = {}
    for split, us, n in (("val", uids[:nval], max(1, round(args.n * args.val_frac))), ("train", uids[nval:], args.n)):
        rows = pick(progs, us, n, rng)
        df = pd.DataFrame([{"prompt": ex[k][0], "response": ex[k][1], "uid": u, "qtype": qt, "step": k, "program": j, "source": "own",
                            "prompt_version": pv} for u, j, k, (qt, ex, pv, _) in rows])
        df.to_parquet(os.path.join(out, f"{split}.parquet"), index=False)
        per_q = collections.Counter(u for u, *_ in rows)
        progs_q = collections.defaultdict(set)
        for u, j, *_ in rows:
            progs_q[u].add(j)
        stats[split] = {"rows": len(rows), "questions": len(per_q), "rows_per_question": dict(sorted(collections.Counter(per_q.values()).items())),
                        "questions_with_rows_from_2plus_programs": sum(len(v) >= 2 for v in progs_q.values()),
                        "by_qtype": dict(sorted(collections.Counter(qt for _, _, _, (qt, _, _, _) in rows).items())),
                        "decision_step": dict(sorted(collections.Counter(k for _, _, k, _ in rows).items()))}
    manifest = {"dir": args.dir, "successful_trajectories_used": n_succ, "questions_with_a_program": len(progs),
                "questions_with_2plus_programs": sum(len(v) >= 2 for v in progs.values()), "programs": sum(len(v) for v in progs.values()),
                "selection": stats, "seed": args.seed}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
