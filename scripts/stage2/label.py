"""Stage 2 labels: an external LLM (GPT-4.1-mini) writes the hindsight of each distinct collected trajectory.

The analyzer is SEED's own (seed.analysis.SEEDEpisodeAnalyzer: the prompt of _build_default_episode_analysis_prompt,
the JSON retry prompt of _build_json_retry_prompt for SEED_ANALYSIS_PARSE_RETRIES attempts, _parse_analysis_response
and the required-skill check), called here directly so the token usage is known (cost cap). Each parsed episode-level skill
then goes through the TKGQA skill filter (tkgqa.skill_filter.validate_tkgqa_skill) and each failure's decision-level skill
through tkgqa.skill_filter.validate_step_prescription. Costs API money: check the estimate with --dry_run first.

    python -m scripts.stage2.label data/tkgqa/stage2/MultiTQ/analyze_order.jsonl --dry_run
    python -m scripts.stage2.label data/tkgqa/stage2/MultiTQ/analyze_order.jsonl --max_cost 7 \\
        --out data/tkgqa/stage2/MultiTQ/skills.jsonl
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.common import REPO_ROOT, load_env_file
from scripts.common import analyzer as make_analyzer
from tkgqa.skill_filter import suspect_steps, validate_step_prescription, validate_tkgqa_skill
from utils.openai_api import chat_completion_with_retry, create_openai_client, extract_message_text


def distinct_trajectories(paths):
    rows, seen = [], set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r["uid"], tuple(a or "" for a in r["actions"]))
                if key not in seen:  # the RL trainer analyses each distinct action sequence of a group once
                    seen.add(key)
                    rows.append(r)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default=None, help="default: data/tkgqa/stage2/<dataset>/skills.jsonl")
    parser.add_argument("--model", default=None, help="default: $OPENAI_MODEL or gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_cost", type=float, default=None, help="USD cap (resumed runs count earlier usage)")
    parser.add_argument("--price_in", type=float, default=0.4, help="USD per 1M uncached prompt tokens (gpt-4.1-mini)")
    parser.add_argument("--price_cached", type=float, default=0.1, help="USD per 1M cached prompt tokens (gpt-4.1-mini)")
    parser.add_argument("--price_out", type=float, default=1.6, help="USD per 1M completion tokens (gpt-4.1-mini)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="count prompts and estimate the cost; no API call")
    args = parser.parse_args()

    rows = distinct_trajectories(args.inputs)
    analyzer = make_analyzer()
    for r in rows:  # prompts are rendered by the current analyzer
        prompt = analyzer._build_default_episode_analysis_prompt(steps=r["analysis_steps"], candidate_step_indices=[],
                                                                 episode_success=r["success"])
        r["analyzer_prompt"] = prompt["messages"][0]["content"]
    est_in = sum(len(r["analyzer_prompt"]) for r in rows) / 4.0
    estimate = est_in * args.price_in / 1e6 + len(rows) * 250 * args.price_out / 1e6
    print(f"{len(rows)} distinct trajectories (success {sum(r['success'] for r in rows)}, failure "
          f"{sum(not r['success'] for r in rows)}); ~{est_in / 1e6:.2f}M input tokens; estimated ${estimate:.2f} "
          f"for one call each (JSON retries add a little)")
    if args.dry_run:
        return 0

    out_path = args.out or os.path.join(REPO_ROOT, "data", "tkgqa", "stage2", rows[0]["dataset"], "skills.jsonl")
    done, spent = set(), 0.0
    cost_of = lambda u: ((u["prompt_tokens"] - u["cached_tokens"]) * args.price_in + u["cached_tokens"] * args.price_cached
                         + u["completion_tokens"] * args.price_out) / 1e6
    if os.path.exists(out_path):
        if not args.resume:
            print(f"{out_path} exists; pass --resume to continue it", file=sys.stderr)
            return 1
        with open(out_path, encoding="utf-8") as f:
            for r in map(json.loads, f):
                spent += cost_of(r["usage"])
                if not r.get("error"):
                    done.add((r["uid"], r["rollout"]))
    todo = [r for r in rows if (r["uid"], r["rollout"]) not in done]
    load_env_file(os.path.join(REPO_ROOT, ".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (environment or .env).", file=sys.stderr)
        return 1
    model = args.model or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
    client = create_openai_client(timeout=180)
    lock = threading.Lock()
    state = {"spent": spent, "done": 0, "accepted": 0, "errors": 0}

    def analyze(r):
        steps, success = r["analysis_steps"], r["success"]
        prompt = analyzer._build_default_episode_analysis_prompt(steps=steps, candidate_step_indices=[], episode_success=success)
        usage = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        content, parsed, last_error, error = "", None, None, None
        for attempt in range(analyzer._get_parse_attempts()):
            with lock:
                if args.max_cost is not None and state["spent"] >= args.max_cost:
                    error = "budget_exhausted"
                    break
            prompt_now = prompt if attempt == 0 else analyzer._build_json_retry_prompt(
                original_prompt=prompt, invalid_response=content, error=last_error)
            try:
                response = chat_completion_with_retry(
                    client=client, model=model, prompt=prompt_now,
                    temperature=args.temperature, max_completion_tokens=args.max_tokens, return_response=True,
                    retries=max(1, int(os.environ.get("OPENAI_API_RETRIES", "5"))))
            except Exception as exc:
                error = repr(exc)
                break
            content = extract_message_text(response)
            u = response.usage
            call = {"prompt_tokens": int(u.prompt_tokens or 0), "completion_tokens": int(u.completion_tokens or 0),
                    "cached_tokens": int(getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0)}
            usage["calls"] += 1
            for key, value in call.items():
                usage[key] += value
            with lock:
                state["spent"] += cost_of(call)
            try:
                parsed = analyzer._parse_analysis_response(content)
            except ValueError as exc:
                last_error, parsed = exc, None
                continue
            if not analyzer._analysis_has_required_skill(parsed):
                last_error, parsed = ValueError("SEED analyzer response missing required field: episode_skill"), None
                continue
            break
        violations = (validate_tkgqa_skill(parsed["episode_skill"], steps, success >= 1.0, parsed.get("failure_step"),
                                           parsed.get("failure_type")) if parsed else ["json" if not error else "api_error"])
        suspects = suspect_steps(steps)
        step_viol = validate_step_prescription(parsed, steps, success, r["answers"]) if parsed else ["json"]
        return {
            "uid": r["uid"], "rollout": r["rollout"], "dataset": r["dataset"], "qtype": r["qtype"],
            "question": r["question"], "answers": r["answers"], "success": success, "failure_type": r["failure_type"],
            "model": model, "temperature": args.temperature, "prompt_version": r["prompt_version"],
            "analyzer_prompt": prompt, "raw_output": content,
            "parsed": {k: parsed.get(k) for k in ("episode_summary", "failure_step", "failure_type", "episode_skill", "step_skill")} if parsed else None,
            "suspects": suspects,
            "failure_step_in_suspects": (None if not parsed or parsed.get("failure_step") is None or suspects is None
                                         else parsed["failure_step"] in suspects),
            "violations": violations, "accepted": not violations, "usage": usage, "error": error,
            "step_violations": step_viol, "step_accepted": bool(parsed) and success < 1.0 and not step_viol,
            "parse_error": None if parsed or error else str(last_error),
        }

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(analyze, r) for r in todo]):
            rec = future.result()
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                state["done"] += 1
                state["accepted"] += rec["accepted"]
                state["errors"] += rec["error"] is not None
                if state["done"] % 20 == 0 or state["done"] == len(todo):
                    print(f"  {state['done']}/{len(todo)} | accepted {state['accepted'] / state['done']:.1%} | errors "
                          f"{state['errors']} | ${state['spent']:.2f}{'' if args.max_cost is None else f' of ${args.max_cost:.2f}'} "
                          f"| {time.time() - t0:.0f}s", flush=True)
    recs = [json.loads(line) for line in open(out_path, encoding="utf-8")]
    print("violations:", dict(Counter(v for r in recs for v in r["violations"]).most_common()))
    print(f"done: ${state['spent']:.2f} spent; accepted {sum(r['accepted'] for r in recs)}/{len(recs)} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
