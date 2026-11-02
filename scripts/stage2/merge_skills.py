"""The Stage 2 analyzer targets: one skills file for scripts.stage2.build_sft --skills.
  success trajectories: the labeller's analysis with its episode-level skill accepted by the TKGQA filter; step_skill "".
  failed trajectories: only those whose decision-level skill passed the checks, the executor pre-check and the
    continuation verification (skills.validated.jsonl; step_skill = the normalised decision-level skill); an
    episode-level skill that failed the TKGQA filter becomes "" (the decision-level skill alone is the target).
Questions whose gold answer is finer than the question asks ("gold_finer_than_question" in
data/tkgqa/unsolvable/<dataset>.json) are left out: a scoring artefact, not a decision to correct.
    python -m scripts.stage2.merge_skills data/tkgqa/stage2/MultiTQ [--dataset MultiTQ]
writes <dir>/skills.final.jsonl (or --out) ("accepted" = goes into the SFT set)
"""
import argparse
import json
import os
from collections import Counter

from scripts.common import REPO_ROOT

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dir")
ap.add_argument("--dataset", default="MultiTQ")
ap.add_argument("--out", default="skills.final.jsonl")
args = ap.parse_args()
d = args.dir
noise = set()
upath = os.path.join(REPO_ROOT, "data", "tkgqa", "unsolvable", f"{args.dataset}.json")
if os.path.exists(upath):
    noise = {u for v in json.load(open(upath)).get("gold_finer_than_question", {}).values() for u in v}
validated = {(r["uid"], r["rollout"]): r for r in map(json.loads, open(os.path.join(d, "skills.validated.jsonl"), encoding="utf-8"))}
out, stats = [], Counter()
for r in map(json.loads, open(os.path.join(d, "skills.jsonl"), encoding="utf-8")):
    key = (r["uid"], r["rollout"])
    if r["uid"] in noise:
        stats["left out: gold finer than the question"] += 1
        r["accepted"] = False
    elif r["success"] >= 1.0:
        r["accepted"] = bool(r["accepted"] and r["parsed"])
        if r["accepted"]:
            r["parsed"]["step_skill"] = ""
        stats[f"success {'in' if r['accepted'] else 'out'}"] += 1
    elif key in validated:
        v = validated[key]
        keep = r["accepted"]
        r["parsed"] = dict(v["parsed"], episode_skill=v["parsed"]["episode_skill"] if keep else "")
        stats["failure in (validated prescription)" + ("" if keep else ", episode skill blanked")] += 1
        r["accepted"] = True
    else:
        stats["failure out (no validated prescription)"] += 1
        r["accepted"] = False
    out.append(r)
with open(os.path.join(d, args.out), "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(dict(stats))
