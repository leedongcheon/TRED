"""Continuation verification of the decision-level hindsight skills. For each decision-level skill that passed the
checks and runs in the symbolic executor (skills.prechecked.jsonl: step_accepted and step_runnable, failure type other
than entity_grounding; its correction proposal with the agent's result variable filled in): the failed trajectory's
own steps 0..k-1 are replayed (the replayed step-k prompt must equal the stored one), then --n continuations from the
selected step k at --temperature under three contexts, with the same sampling seeds:
  none          no skill
  prescription  this case's decision-level skill in SEED's Critical-Step block at step k
  control       the decision-level skill of a case of another question (another failure type when possible)
Accepted (fixed before the run): prescription - none >= --min_gain AND prescription - control >= --min_gain (successes
of --n). Chance: control - none >= --min_gain. Reported by failure type and by the executor status of the correction
proposal. The cases are split by question into --num_shards shards (one per GPU, questions alternating in uid order);
the control choice and the seeds depend on the shard, so use the same number of shards to reproduce a run.

    python -m scripts.stage2.verify run --model_path outputs/stage1/MultiTQ/global_step_150 \\
        --dir data/tkgqa/stage2/MultiTQ --shard 0 --num_shards 2
    python -m scripts.stage2.verify merge --dir data/tkgqa/stage2/MultiTQ --num_shards 2
run writes <dir>/val_shard<k>/prescription_validation.jsonl (+ .summary.json), prescription_continuations.jsonl and
skills.validated.jsonl (accepted, step_skill normalised); merge concatenates the shards into <dir>.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import zlib
from collections import Counter, defaultdict

from scripts.common import REPO_ROOT

STEP_RE = re.compile(r"^\s*At this decision, write (.+?) instead of (.+?)\.?\s*$", re.S)
CONTEXTS = ("none", "prescription", "control")
OUTPUTS = ("skills.validated.jsonl", "prescription_validation.jsonl", "prescription_continuations.jsonl")


def shard_rows(rows, shard: int, num_shards: int):
    """The validation cases of one shard, in file order."""
    keep = [r for r in rows if r.get("step_accepted") and r.get("step_runnable")
            and (r.get("parsed") or {}).get("failure_type") != "entity_grounding"]
    part = {u: i % num_shards for i, u in enumerate(sorted({r["uid"] for r in keep}))}
    return [r for r in keep if part[r["uid"]] == shard]


def merge(args) -> int:
    for name in OUTPUTS:
        with open(os.path.join(args.dir, name), "w", encoding="utf-8") as out:
            for k in range(args.num_shards):
                out.writelines(open(os.path.join(args.dir, f"val_shard{k}", name), encoding="utf-8"))
    v = [json.loads(l) for l in open(os.path.join(args.dir, "prescription_validation.jsonl"), encoding="utf-8")]
    c = sum(1 for _ in open(os.path.join(args.dir, "prescription_continuations.jsonl"), encoding="utf-8"))
    summary = {"n": len(v), "accepted": sum(bool(r.get("accepted")) for r in v), "continuations": c}
    json.dump(summary, open(os.path.join(args.dir, "prescription_validation.summary.json"), "w"), indent=2)
    print(json.dumps(summary))
    return 0


def run(args) -> int:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from agent_system.environments.env_package.tkgqa.offline import chat_prompt_ids, env_kwargs, make_manager
    from agent_system.environments.env_package.tkgqa.rollout import load_policy
    from seed.prompting import build_augmented_observation_text
    from tkgqa.grounder import Grounder
    from tkgqa.kg import TemporalKG

    trajs = {(t["uid"], t["rollout"]): t for path in sorted(glob.glob(os.path.join(args.dir, "trajectories.shard*.jsonl")))
             for t in map(json.loads, open(path, encoding="utf-8"))}
    rows = shard_rows([json.loads(line) for line in open(os.path.join(args.dir, "skills.prechecked.jsonl"), encoding="utf-8")],
                      args.shard, args.num_shards)
    out_dir = os.path.join(args.dir, f"val_shard{args.shard}")
    os.makedirs(out_dir, exist_ok=True)
    cases = []
    for r in rows:
        m = STEP_RE.match(r["parsed"]["step_skill"].strip())
        text = f"At this decision, write {r['step_corrected_filled']} instead of {m.group(2).strip().rstrip('.')}."
        cases.append({"uid": r["uid"], "rollout": r["rollout"], "qtype": r["qtype"], "failure_type": r["parsed"]["failure_type"],
                      "k": r["parsed"]["failure_step"], "exec": r["step_exec_filled"][0], "text": text, "record": r,
                      "traj": trajs[(r["uid"], r["rollout"])]})
    rng = random.Random(0)
    for c in cases:
        pool = [o for o in cases if o["uid"] != c["uid"] and o["failure_type"] != c["failure_type"]] or [o for o in cases if o["uid"] != c["uid"]]
        c["control"] = rng.choice(pool)["text"]
    print(f"{len(cases)} prescriptions ({Counter(c['exec'] for c in cases)}) over {len({c['uid'] for c in cases})} questions", flush=True)

    kg = TemporalKG.load(args.dataset, args.data_root)
    grounder = Grounder.from_store_root(kg, args.store_root, device="cuda")
    llm, tok = load_policy(args.model_path, 0.6, 8192, args.seed)
    success = defaultdict(list)
    mismatched = set()
    continuation = {}  # case index -> [[plain prompt, response], ...] from step k: the first success under the decision-level skill
    for k in sorted({c["k"] for c in cases} & (set(args.steps) if args.steps else {c["k"] for c in cases})):
        group = [(i, c) for i, c in enumerate(cases) if c["k"] == k]
        items = [(i, c, ctx, r) for i, c in group for ctx in CONTEXTS for r in range(args.n)]
        manager = make_manager(kg, grounder, len(items))
        obs, _ = manager.reset([env_kwargs(c["traj"]) for _, c, _, _ in items])
        current = obs["text"]
        for step in range(k):
            current = manager.step([c["traj"]["steps"][step]["response"] for _, c, _, _ in items])[0]["text"]
        ok = [current[j] == c["traj"]["steps"][k]["prompt"] for j, (_, c, _, _) in enumerate(items)]
        mismatched |= {i for j, (i, _, _, _) in enumerate(items) if not ok[j]}
        sessions, step = manager.envs.sessions, k
        log = [[] for _ in items]  # (the plain prompt, the response) of every step from k on (no skill block in the prompt)
        while True:
            active = [j for j, s in enumerate(sessions) if not s.done]
            if not active:
                break
            reqs, params = [], []
            for j in active:
                i, c, ctx, r = items[j]
                prompt = current[j]
                log[j].append([prompt, None])
                if step == k and ctx != "none":
                    prompt = build_augmented_observation_text(observation=prompt, step_skill=c["text"] if ctx == "prescription" else c["control"])
                reqs.append(TokensPrompt(prompt_token_ids=chat_prompt_ids(tok, prompt)))
                params.append(SamplingParams(temperature=args.temperature, max_tokens=256, stop=["</action>"], include_stop_str_in_output=True,
                                             seed=zlib.crc32(f"{args.seed}:{c['uid']}:{i}:{r}:{step}".encode())))
            outs = llm.generate(reqs, params, use_tqdm=False)
            texts = [""] * len(items)
            for j, o in zip(active, outs):
                texts[j] = o.outputs[0].text
                log[j][-1][1] = texts[j]
            current = manager.step(texts)[0]["text"]
            step += 1
        for j, (i, c, ctx, r) in enumerate(items):
            if ok[j]:
                success[(i, ctx)].append(bool(manager.envs.scores[j]["success"]))
                # the first successful continuation under the decision-level skill (the final recipe's corrected trajectories)
                if ctx == "prescription" and manager.envs.scores[j]["success"] and i not in continuation:
                    continuation[i] = log[j]
        print(f"  k={k}: {len(group)} prescriptions done", flush=True)

    out_rows = []
    for i, c in enumerate(cases):
        if i in mismatched or not success[(i, "none")]:
            continue
        cnt = {ctx: sum(success[(i, ctx)]) for ctx in CONTEXTS}
        acc = cnt["prescription"] - cnt["none"] >= args.min_gain and cnt["prescription"] - cnt["control"] >= args.min_gain
        # the text goes under its own key: "decision-level skill" is also a context name, so **cnt would overwrite it with a count
        out_rows.append({"uid": c["uid"], "rollout": c["rollout"], "qtype": c["qtype"], "failure_type": c["failure_type"], "k": c["k"],
                         "exec": c["exec"], "prescription_text": c["text"], **cnt, "accepted": acc,
                         "chance": cnt["control"] - cnt["none"] >= args.min_gain})
    n = len(out_rows)
    def agg(rs):
        m = len(rs)
        return {"n": m, "accept_rate": sum(r["accepted"] for r in rs) / m, "chance_rate": sum(r["chance"] for r in rs) / m,
                "mean_gain_vs_none": sum(r["prescription"] - r["none"] for r in rs) / m / args.n,
                "mean_gain_vs_control": sum(r["prescription"] - r["control"] for r in rs) / m / args.n,
                "success_none": sum(r["none"] for r in rs) / m / args.n, "success_prescription": sum(r["prescription"] for r in rs) / m / args.n,
                "success_control": sum(r["control"] for r in rs) / m / args.n} if m else {"n": 0}
    summary = {"model_path": args.model_path, "n_per_context": args.n, "temperature": args.temperature,
               "rule": f"prescription - none >= {args.min_gain} and prescription - control >= {args.min_gain} (of {args.n})",
               "replay_mismatch": len(mismatched), "all": agg(out_rows)}
    for key in ("exec", "failure_type"):
        g = defaultdict(list)
        for r in out_rows:
            g[r[key]].append(r)
        summary[f"by_{key}"] = {v: agg(rs) for v, rs in sorted(g.items(), key=lambda kv: -len(kv[1]))}
    json.dump(summary, open(os.path.join(out_dir, "prescription_validation.summary.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "prescription_validation.jsonl"), "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # successful continuations of the accepted decision-level skills (prefix = the agent's own steps 0..k-1 of the trajectory)
    acc_keys = {(r["uid"], r["rollout"]) for r in out_rows if r["accepted"]}
    with open(os.path.join(out_dir, "prescription_continuations.jsonl"), "w", encoding="utf-8") as f:
        for i, c in enumerate(cases):
            if (c["uid"], c["rollout"]) in acc_keys and i in continuation and i not in mismatched:
                prefix = [[s["prompt"], s["response"]] for s in c["traj"]["steps"][: c["k"]]]
                f.write(json.dumps({"uid": c["uid"], "rollout": c["rollout"], "qtype": c["qtype"], "failure_type": c["failure_type"],
                                    "k": c["k"], "prescription_text": c["text"], "prompt_version": c["traj"].get("prompt_version"),
                                    "prefix": prefix, "continuation": continuation[i]}, ensure_ascii=False) + "\n")
    accepted = {(r["uid"], r["rollout"]): r["prescription_text"] for r in out_rows if r["accepted"]}
    with open(os.path.join(out_dir, "skills.validated.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            key = (r["uid"], r["rollout"])
            if key in accepted:
                r = dict(r, parsed=dict(r["parsed"], step_skill=accepted[key]), step_validated=True)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model_path", required=True)
    r.add_argument("--dir", required=True)
    r.add_argument("--dataset", default="MultiTQ")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--num_shards", type=int, default=2)
    r.add_argument("--n", type=int, default=16)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--min_gain", type=int, default=2)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--steps", type=int, nargs="*", default=None,
                   help="run only the cases whose failure step is one of these (a quick check; cases, controls and seeds unchanged)")
    r.add_argument("--data_root", default=os.path.join(REPO_ROOT, "datasets"))
    r.add_argument("--store_root", default=os.path.join(REPO_ROOT, "stores"))
    m = sub.add_parser("merge")
    m.add_argument("--dir", required=True)
    m.add_argument("--num_shards", type=int, default=2)
    args = p.parse_args()
    return run(args) if args.cmd == "run" else merge(args)


if __name__ == "__main__":
    sys.exit(main())
