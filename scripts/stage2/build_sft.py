"""Build the Stage 2 (Hindsight-Skill SFT) set from the collected trajectories and the final analyzer targets
(skills.final.jsonl).

(a) analyzer examples: prompt = the analyzer prompt of the trajectory as SEED's analyzer renders it (seed.analysis, the
    same function RL calls), response = the accepted JSON (episode_summary, failure_step = the selected step,
    failure_type, episode_skill = the episode-level hindsight skill, step_skill = the decision-level hindsight skill),
    success : failure at most --max_ratio : 1. The names of the episode in episode_summary are replaced by
    their role (the subject / reference / answer entity, the relation, the anchor / answer date): the summary is not
    used downstream, and names in the targets teach the analyzer to write names (which then reach skills). An example
    whose prompt differs from the one the labeller saw is left out (counted in the manifest).
(b) actor replay: --replay step examples of --replay_dir (the own-trajectory replay and the corrected trajectories,
    scripts.stage2.corrected; default: as many as the analyzer examples).
Questions in --holdout_frac are left out of (a), for the analyzer evaluation, and so are the unsolvable questions of
--unsolvable (grounding ceiling: no avoidance rule to teach there). Train/val are split by question.

    python -m scripts.stage2.build_sft --dataset MultiTQ --base data/tkgqa/stage2/MultiTQ \\
        --replay_dir data/tkgqa/sft/replay/MultiTQ --replay 1579 --out_dir data/tkgqa/sft/stage2/MultiTQ
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

import pandas as pd

from agent_system.environments.env_package.tkgqa.offline import prompt_version
from scripts.common import REPO_ROOT
from scripts.common import analyzer as make_analyzer
from tkgqa.skill_filter import episode_names, episode_relations, scrub_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="MultiTQ")
    parser.add_argument("--base", default=None, help="collection directory (default: data/tkgqa/stage2/<dataset>)")
    parser.add_argument("--skills", default=None, help="default: <base>/skills.final.jsonl")
    parser.add_argument("--out_dir", default=None, help="default: data/tkgqa/sft/stage2/<dataset>")
    parser.add_argument("--max_ratio", type=float, default=1.5, help="success : failure analyzer examples at most this : 1")
    parser.add_argument("--holdout_frac", type=float, default=0.1)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replay", type=int, default=None, help="actor replay examples (default: as many as (a) train)")
    parser.add_argument("--replay_dir", required=True, help="the replay set (train.parquet / val.parquet)")
    parser.add_argument("--unsolvable", default=None, help="default: data/tkgqa/unsolvable/<dataset>.json if present")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = args.base or os.path.join(REPO_ROOT, "data", "tkgqa", "stage2", args.dataset)
    trajs = {}
    for path in glob.glob(os.path.join(base, "trajectories.shard*.jsonl")):
        for line in open(path, encoding="utf-8"):
            t = json.loads(line)
            trajs[(t["uid"], t["rollout"])] = t
    skills = [json.loads(line) for line in open(args.skills or os.path.join(base, "skills.final.jsonl"), encoding="utf-8")]
    accepted = [s for s in skills if s["accepted"]]
    versions = {trajs[(s["uid"], s["rollout"])]["prompt_version"] for s in accepted}
    if versions != {prompt_version()}:
        print(f"trajectories were collected with prompt versions {versions}, the current one is {prompt_version()}", file=sys.stderr)
        return 1

    uids = sorted({t["uid"] for t in trajs.values()})
    rng.shuffle(uids)
    n_hold = int(len(uids) * args.holdout_frac)
    holdout, rest = set(uids[:n_hold]), uids[n_hold:]
    val = set(rest[: max(1, int(len(rest) * args.val_frac))])
    split = lambda uid: "val" if uid in val else "train"

    unsolvable_path = args.unsolvable or os.path.join(REPO_ROOT, "data", "tkgqa", "unsolvable", f"{args.dataset}.json")
    unsolvable = set()
    if os.path.exists(unsolvable_path):
        with open(unsolvable_path) as f:
            u = json.load(f)
        unsolvable = set(u.get("grounding_ceiling", [])) | set(u.get("operation_ceiling", []))

    # (a) analyzer examples, success : failure balanced
    analyzer = make_analyzer()
    pool = [s for s in accepted if s["uid"] not in holdout and s["uid"] not in unsolvable]
    excluded_unsolvable = sum(s["uid"] in unsolvable for s in accepted)
    succ = [s for s in pool if s["success"] >= 1.0]
    fail = [s for s in pool if s["success"] < 1.0]
    rng.shuffle(succ)
    succ = succ[: int(args.max_ratio * len(fail))]
    rows = {"train": [], "val": []}
    prompt_mismatch = 0
    for s in succ + fail:
        t = trajs[(s["uid"], s["rollout"])]
        prompt = analyzer._build_default_episode_analysis_prompt(steps=t["analysis_steps"], candidate_step_indices=[],
                                                                 episode_success=t["success"])["messages"][0]["content"]
        labelled = (s.get("analyzer_prompt") or {}).get("messages", [{}])[0].get("content")
        if labelled is not None and labelled != prompt:
            prompt_mismatch += 1
            continue
        p = s["parsed"]
        summary = scrub_names(p["episode_summary"], episode_names(t["analysis_steps"]), question=t["question"],
                              answers=t["answers"], relations=episode_relations(t["analysis_steps"]))
        fields = {"episode_summary": summary, "failure_step": p["failure_step"], "failure_type": p["failure_type"],
                  "episode_skill": p["episode_skill"], "step_skill": p.get("step_skill") or ""}
        target = json.dumps(fields, indent=2, ensure_ascii=False)
        rows[split(s["uid"])].append({"role": "analyzer", "prompt": prompt, "response": target, "uid": s["uid"],
                                      "rollout": s["rollout"], "qtype": s["qtype"], "success": float(s["success"]),
                                      "prompt_version": t["prompt_version"]})

    # (b) actor replay
    n_replay = args.replay if args.replay is not None else sum(r["role"] == "analyzer" for r in rows["train"])
    for name, n in (("train", n_replay), ("val", max(1, n_replay * len(rows["val"]) // max(len(rows["train"]), 1)))):
        r0 = pd.read_parquet(os.path.join(args.replay_dir, f"{name}.parquet"))
        if set(r0["prompt_version"]) != {prompt_version()}:
            print(f"the replay set has prompt versions {set(r0['prompt_version'])}", file=sys.stderr)
            return 1
        r0 = r0.sample(n=min(n, len(r0)), random_state=args.seed)
        rows[name] += [{"role": "replay", "prompt": x.prompt, "response": x.response, "uid": x.uid, "rollout": -1,
                        "qtype": x.qtype, "success": 1.0, "prompt_version": x.prompt_version} for x in r0.itertuples()]

    out_dir = args.out_dir or os.path.join(REPO_ROOT, "data", "tkgqa", "sft", "stage2", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    for name, data in rows.items():
        rng.shuffle(data)
        pd.DataFrame(data).to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
    with open(os.path.join(out_dir, "holdout_uids.json"), "w") as f:
        json.dump(sorted(holdout), f)
    count = lambda role, split_name: sum(r["role"] == role for r in rows[split_name])
    manifest = {
        "prompt_version": prompt_version(), "skills": args.skills or "skills.final.jsonl", "replay_dir": args.replay_dir,
        "holdout_questions": len(holdout), "val_questions": len(val),
        "unsolvable_questions": len(unsolvable), "accepted_skills_left_out_as_unsolvable": excluded_unsolvable,
        "analyzer_examples": {"train": count("analyzer", "train"), "val": count("analyzer", "val"),
                              "success": len(succ), "failure": len(fail), "left_out_prompt_mismatch": prompt_mismatch},
        "analyzer_failure_types": dict(Counter(s["parsed"]["failure_type"] for s in fail).most_common()),
        "replay_examples": {"train": count("replay", "train"), "val": count("replay", "val")},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
