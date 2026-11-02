"""Stage 2 collection: completed trajectories of the Stage 1 checkpoint on the stage2 questions, each with the
analyzer input the RL trainer would build for it, and the analyzer prompt rendered by SEED's own analyzer.

Per trajectory: the actor prompt, response, action, status and observation of every step; the step records of
seed.analysis.build_episode_records (observation = what the agent saw before acting: the question at step 0, then
the previous observation; response; action_valid = projection and executor; the env's Hindsight line), with the
episode's final_observation and auto_hints (the diagnostic hints, with the lines of discarded gold-bearing facts and
extra answers) on the last step; and the prompt of SEEDEpisodeAnalyzer._build_default_episode_analysis_prompt (the same prompt as in RL).

Roll out (one shard per GPU):
    python -m scripts.stage2.collect rollout --model_path outputs/stage1/MultiTQ/global_step_150 \\
        --dataset MultiTQ --shard 0 --num_shards 2 --out data/tkgqa/stage2/MultiTQ/trajectories.shard0.jsonl
Summary, distinct analyzer prompts and an API cost estimate (no API call):
    python -m scripts.stage2.collect summary data/tkgqa/stage2/MultiTQ/trajectories.shard*.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

from scripts.common import REPO_ROOT, analyzer


def analysis_steps(question: str, prompts, responses, session, step_lines, auto_hints):
    """The step records the RL trainer builds for this trajectory (seed.analysis.build_episode_records layout)."""
    from agent_system.environments.env_package.tkgqa.projection import tkgqa_projection
    from seed.analysis import _extract_task_description_from_text

    steps = []
    for k, (prompt, response, step, line) in enumerate(zip(prompts, responses, session.log, step_lines)):
        projected_valid = bool(tkgqa_projection([response])[1][0])
        record = {
            "step_index": k,
            "observation": question if k == 0 else session.log[k - 1].obs,
            "observation_prompt": prompt,
            "response": response,
            "action_valid": projected_valid and step.status != "invalid",
            "hindsight": line,
        }
        task = _extract_task_description_from_text(prompt)
        if task:
            record["task_description"] = task
        steps.append(record)
    if steps:
        steps[-1]["final_observation"] = session.log[-1].obs
        steps[-1]["auto_hints"] = auto_hints or ""
    return steps


def rollout(args) -> int:
    from agent_system.environments.env_package.tkgqa.offline import prompt_version
    from agent_system.environments.env_package.tkgqa.rollout import load_policy, run_rollouts
    from tkgqa.grounder import Grounder
    from tkgqa.kg import TemporalKG

    path = args.questions or os.path.join(REPO_ROOT, "data", "tkgqa", "splits", args.dataset, "stage2.jsonl")
    with open(path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f][args.shard :: args.num_shards]
    kg = TemporalKG.load(args.dataset, os.path.join(REPO_ROOT, "datasets"))
    grounder = Grounder.from_store_root(kg, os.path.join(REPO_ROOT, "stores"), device="cpu")
    llm, tokenizer = load_policy(args.model_path, 0.6, 8192, args.seed)
    print(f"[{args.dataset}] shard {args.shard}/{args.num_shards}: {len(questions)} questions x {args.n} (T {args.temperature})", flush=True)
    rollouts = run_rollouts(llm, tokenizer, kg, grounder, questions, n=args.n, temperature=args.temperature,
                            seed=args.seed, keep_prompts=True, log=lambda m: print(m, flush=True))
    a = analyzer()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    version = prompt_version()
    with open(args.out, "w", encoding="utf-8") as out:
        for ro in rollouts:
            q, session, score = ro["q"], ro["session"], ro["score"]
            steps = analysis_steps(q["question"], ro["prompts"], ro["responses"], session, ro["step_hindsights"], ro["auto_hints"])
            success = float(score["success"])
            prompt = a._build_default_episode_analysis_prompt(steps=steps, candidate_step_indices=[], episode_success=success)
            out.write(json.dumps({
                "uid": q["uid"], "dataset": q["dataset"], "qtype": q["qtype"], "question": q["question"],
                "answers": q["answers"], "kind": q["kind"], "rollout": ro["r"], "model": args.model_path,
                "temperature": args.temperature, "prompt_version": version,
                "score": score, "success": success, "failure_type": ro["hindsight"]["failure_type"],
                "truncated": session.truncated, "actions": [s.action for s in session.log],
                "steps": [{"prompt": p, "response": r, "action": s.action, "status": s.status, "obs": s.obs}
                          for p, r, s in zip(ro["prompts"], ro["responses"], session.log)],
                "analysis_steps": steps,
                "analyzer_prompt": prompt["messages"][0]["content"] if "messages" in prompt else prompt,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(rollouts)} trajectories -> {args.out}", flush=True)
    return 0


def summary(args) -> int:
    rows = [json.loads(line) for path in args.inputs for line in open(path, encoding="utf-8")]
    distinct = {}
    for r in rows:  # the RL trainer analyses each distinct action sequence of a group once
        distinct.setdefault((r["uid"], tuple(a or "" for a in r["actions"])), r)
    groups = defaultdict(list)
    for r in rows:
        groups[r["uid"]].append(r["success"])
    mixed = sum(0 < sum(v) < len(v) for v in groups.values())
    chars = [len(r["analyzer_prompt"]) for r in distinct.values()]
    tok_in = sum(chars) / 4.0  # ~4 characters per GPT token for this text
    cost = tok_in * args.price_in / 1e6 + len(distinct) * args.out_tokens * args.price_out / 1e6
    print(f"{len(rows)} trajectories over {len(groups)} questions; success {sum(r['success'] for r in rows) / len(rows):.1%}; "
          f"mixed groups {mixed}/{len(groups)}; step limit {sum(r['truncated'] for r in rows) / len(rows):.1%}")
    print("per qtype success:", {t: f"{sum(r['success'] for r in rows if r['qtype'] == t) / max(sum(r['qtype'] == t for r in rows), 1):.2f}"
                                 for t in sorted({r['qtype'] for r in rows})})
    print("failure types:", dict(Counter(r["failure_type"] for r in rows).most_common()))
    print(f"distinct trajectories (analyzer prompts): {len(distinct)} "
          f"(success {sum(r['success'] for r in distinct.values())}, failure {sum(not r['success'] for r in distinct.values())})")
    print(f"prompt length: mean {sum(chars) / len(chars):.0f} chars, max {max(chars)} -> ~{tok_in / 1e6:.2f}M input tokens")
    print(f"estimated labelling cost for one call each: ${cost:.2f} (input ${args.price_in}/M, ~{args.out_tokens} output tokens at ${args.price_out}/M)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollout")
    r.add_argument("--model_path", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--questions", default=None, help="default: data/tkgqa/splits/<dataset>/stage2.jsonl")
    r.add_argument("--n", type=int, default=8)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--num_shards", type=int, default=1)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--price_in", type=float, default=0.4, help="USD per 1M prompt tokens (gpt-4.1-mini)")
    s.add_argument("--price_out", type=float, default=1.6, help="USD per 1M completion tokens (gpt-4.1-mini)")
    s.add_argument("--out_tokens", type=int, default=250)
    args = parser.parse_args()
    return rollout(args) if args.cmd == "rollout" else summary(args)


if __name__ == "__main__":
    sys.exit(main())
