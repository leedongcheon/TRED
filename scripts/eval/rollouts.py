"""Roll out a checkpoint on a question split (vLLM + the symbolic executor) and score it.

Used for the Stage 1 checkpoint choice (by rollouts, not loss; scripts.stage1.select) and for the test evaluation.
Each question gets ``--n`` rollouts at ``--temperature`` in the TKGQA environment (TKGQAEnvironmentManager), so every
step uses the same actor prompt and chat template as SFT and RL.

    python -m scripts.eval.rollouts --model_path outputs/stage1/MultiTQ/global_step_150 \\
        --dataset MultiTQ --n 8 --temperature 0.4                      # heldout (default questions)
    python -m scripts.eval.rollouts --model_path <HF dir> --dataset MultiTQ \\
        --questions data/tkgqa/splits/MultiTQ/test.jsonl --n 1 --temperature 0.0 --shard 0 --num_shards 2
    python -m scripts.eval.rollouts --model_path <HF dir> --dataset MultiTQ \\
        --questions data/tkgqa/splits/MultiTQ/test.jsonl --merge 2   # the shards into one evaluation

Reported: action validity (a failed FIND counts as valid), success = F1 of 1,
mean F1, Hits@1, mixed groups (reward variance > 0 within a question's
rollouts; also with std >= --tie_std), step-limit rate, per question type, and
decision-slot diversity (tkgqa.diversity): per slot (PICK direction, FILTER
relation and gran, TIME first/last, GET gran) the share of groups whose
rollouts chose different values, and the probability put on the chosen value
vs its alternatives at that token.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List

from agent_system.environments.env_package.tkgqa.offline import max_steps_for, prompt_version
from agent_system.environments.env_package.tkgqa.rollout import load_policy, run_rollouts
from tkgqa.diversity import group_divergence, slot_values, summarize_decisions
from tkgqa.grounder import Grounder
from tkgqa.kg import TemporalKG
from tkgqa.kg_vocab import resolve_dataset_name
from scripts.common import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", default=None, help="default: data/tkgqa/splits/<dataset>/heldout.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=None, help="default: the dataset's limit (offline.max_steps_for)")
    parser.add_argument("--tie_std", type=float, default=0.05, help="groups below this reward std count as ties")
    parser.add_argument("--logprobs", type=int, default=5, help="top-k logprobs per token for decision probabilities")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "datasets"))
    parser.add_argument("--store_root", default=os.path.join(REPO_ROOT, "stores"))
    parser.add_argument("--device", default="cuda", help="device for the bge-m3 grounder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0, help="evaluate questions[shard::num_shards] (merge them with --merge)")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--out", default=None, help="default: <model_path>/eval_<dataset>_<questions file>[.shardKofN].json")
    parser.add_argument("--force", action="store_true", help="re-run even if --out already holds the same evaluation")
    parser.add_argument("--merge", type=int, default=0, metavar="N",
                        help="no model: merge the N shard evaluations of --questions into one (as --num_shards 1 writes it)")
    return parser.parse_args()


def find_recovery(per_rollout: List[Dict]) -> Dict[str, float]:
    """Failed FINDs and the share recovered: a later FIND of the same rollout assigns that variable."""
    failed = recovered = 0
    for r in per_rollout:
        for i, step in enumerate(r["steps"]):
            if step["status"] != "find_failed":
                continue
            failed += 1
            name = (step["action"] or "").split("=", 1)[0].strip()
            recovered += any(later["status"] == "ok" and (later["action"] or "").split("=", 1)[0].strip() == name
                             and "FIND(" in (later["action"] or "") for later in r["steps"][i + 1:])
    return {"find_failed_steps": float(failed), "find_failed_recovery": recovered / failed if failed else 0.0}


def summarize(per_rollout: List[Dict], *, model_path: str, dataset: str, max_steps: int, temperature: float,
              n_questions: int, tie_std: float) -> Dict:
    """The evaluation summary of a list of rollout records (also used to merge shards)."""
    groups = defaultdict(list)
    for r in per_rollout:
        groups[r["uid"]].append(r)
    steps = [s for r in per_rollout for s in r["steps"]]
    rewards_by_q = {uid: [r["score"]["reward"] for r in rs] for uid, rs in groups.items()}
    std = {uid: statistics.pstdev(v) for uid, v in rewards_by_q.items()}
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    summary = {
        "model_path": model_path,
        "dataset": dataset,
        "prompt_version": prompt_version(),
        "max_steps": max_steps,
        "temperature": temperature,
        "questions": n_questions,
        "rollouts": len(per_rollout),
        "action_validity": mean([s["status"] != "invalid" for s in steps]),
        "invalid_rate": mean([s["status"] == "invalid" for s in steps]),
        **find_recovery(per_rollout),
        "success": mean([r["score"]["success"] for r in per_rollout]),
        "f1": mean([r["score"]["f1"] for r in per_rollout]),
        "hit1": mean([r["score"]["hit1"] for r in per_rollout]),
        "mixed_groups": mean([s > 0 for s in std.values()]),
        f"mixed_groups_std>={tie_std}": mean([s >= tie_std for s in std.values()]),
        "all_same_reward_groups": mean([s == 0 for s in std.values()]),
        "step_limit_rate": mean([r["truncated"] for r in per_rollout]),
        "find_failed_rate": mean([s["status"] == "find_failed" for s in steps]),
        "mean_steps": mean([len(r["steps"]) for r in per_rollout]),
        "per_qtype": {},
        "failure_type": {t: n / len(per_rollout) for t, n in Counter(r["failure_type"] for r in per_rollout).most_common()},
    }
    for qtype in sorted({r["qtype"] for r in per_rollout}):
        rs = [r for r in per_rollout if r["qtype"] == qtype]
        summary["per_qtype"][qtype] = {
            "rollouts": len(rs),
            "success": mean([r["score"]["success"] for r in rs]),
            "f1": mean([r["score"]["f1"] for r in rs]),
            "hit1": mean([r["score"]["hit1"] for r in rs]),
            "mean_steps": mean([len(r["steps"]) for r in rs]),
        }
    # slot values are tuples; records read back from JSON hold lists
    slots = lambda r: {k: tuple(v) for k, v in r["slots"].items()}
    summary["slot_divergence"] = group_divergence({uid: [slots(r) for r in rs] for uid, rs in groups.items()})
    summary["decision_probability"] = summarize_decisions(tuple(d) for r in per_rollout for d in r["decisions"])
    summary["no_answer_rate"] = mean([r["failure_type"] == "no_answer" or bool(r["truncated"]) for r in per_rollout])
    return summary


def merge_shards(args, dataset: str, questions_path: str) -> int:
    base = os.path.join(args.model_path, f"eval_{dataset}_{os.path.splitext(os.path.basename(questions_path))[0]}")
    heads = [json.load(open(f"{base}.shard{k}of{args.merge}.json")) for k in range(args.merge)]
    rows = [json.loads(l) for k in range(args.merge) for l in open(f"{base}.shard{k}of{args.merge}.rollouts.jsonl")]
    summary = summarize(rows, model_path=args.model_path, dataset=dataset, max_steps=heads[0]["max_steps"],
                        temperature=heads[0]["temperature"], n_questions=sum(h["questions"] for h in heads), tie_std=args.tie_std)
    summary["merged_from"] = [f"shard{k}of{args.merge}" for k in range(args.merge)]
    json.dump(summary, open(f"{base}.json", "w"), indent=2)
    with open(f"{base}.rollouts.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{base}.json: {len(rows)} rollouts, success {summary['success']:.4f}")
    return 0


def main() -> int:
    args = parse_args()

    dataset = resolve_dataset_name(args.dataset)
    args.max_steps = max_steps_for(dataset) if args.max_steps is None else args.max_steps
    questions_path = args.questions or os.path.join(REPO_ROOT, "data", "tkgqa", "splits", dataset, "heldout.jsonl")
    if args.merge:
        return merge_shards(args, dataset, questions_path)
    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    if args.limit:
        questions = questions[: args.limit]
    shard_tag = ""
    if args.num_shards > 1:
        questions = questions[args.shard :: args.num_shards]
        shard_tag = f".shard{args.shard}of{args.num_shards}"
    out = args.out or os.path.join(
        args.model_path, f"eval_{dataset}_{os.path.splitext(os.path.basename(questions_path))[0]}{shard_tag}.json"
    )
    if os.path.exists(out) and not args.force:
        with open(out, encoding="utf-8") as f:
            done = json.load(f)
        same = (done.get("prompt_version") == prompt_version() and done.get("questions") == len(questions)
                and done.get("rollouts") == len(questions) * args.n and done.get("temperature") == args.temperature
                and done.get("max_steps") == args.max_steps)
        if same:
            print(f"{out} already holds this evaluation; skipping (use --force to re-run)")
            return 0

    t0 = time.time()
    kg = TemporalKG.load(dataset, args.data_root)
    grounder = Grounder.from_store_root(kg, args.store_root, device=args.device)
    llm, tokenizer = load_policy(args.model_path, args.gpu_memory_utilization, args.max_model_len, args.seed)
    print(f"[{dataset}] {len(questions)} questions x {args.n} rollouts (setup {time.time() - t0:.0f}s)", flush=True)
    rollouts = run_rollouts(
        llm, tokenizer, kg, grounder, questions, n=args.n, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, max_steps=args.max_steps,
        seed=args.seed, logprobs=args.logprobs, log=lambda m: print(m, flush=True),
    )

    per_rollout, groups = [], defaultdict(list)
    for ro in rollouts:
        q, session = ro["q"], ro["session"]
        record = {
            "uid": q["uid"], "qtype": q["qtype"], "rollout": ro["r"], "score": ro["score"],
            "failure_type": ro["hindsight"]["failure_type"],
            "answer": session.answer, "truncated": session.truncated, "slots": slot_values(session),
            "decisions": ro["decisions"],
            "steps": [{"action": s.action, "status": s.status, "obs": s.obs} for s in session.log],
        }
        per_rollout.append(record)
        groups[q["uid"]].append(record)

    summary = summarize(per_rollout, model_path=args.model_path, dataset=dataset, max_steps=args.max_steps,
                        temperature=args.temperature, n_questions=len(questions), tie_std=args.tie_std)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out.replace(".json", ".rollouts.jsonl"), "w", encoding="utf-8") as f:
        for r in per_rollout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_qtype", "slot_divergence", "decision_probability")}, indent=2))
    print(f"{'slot':12s} {'groups':>6s} {'value_split':>11s} {'presence_split':>14s} {'decisions':>9s} {'p_chosen':>8s} {'p_alt':>6s} {'>0.99':>6s}")
    for slot in summary["slot_divergence"]:
        d, p = summary["slot_divergence"][slot], summary["decision_probability"][slot]
        print(f"{slot:12s} {d['groups_using']:6.0f} {d['value_split']:11.2f} {d['presence_split']:14.2f} "
              f"{p['decisions']:9.0f} {p['p_chosen_mean']:8.3f} {p['p_alternatives_mean']:6.3f} {p['near_deterministic']:6.2f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
