"""Stage 1 (Temporal-Operation SFT) data: a teacher LLM solves questions step by step against the symbolic executor.

The teacher plays in the RL environment itself (TKGQAEnvironmentManager, env_package/tkgqa/offline.py):
every step sends the solved toy examples (tkgqa.fewshot) plus the unchanged actor prompt (the Search-style template
with the full history) to the teacher, which answers in the agent's own format (one <action> block), and the manager
projects the response, runs it in the executor and builds the next prompt.
Attempts of a question run in order: attempt k > 0 only runs when the earlier ones did not pass, at the
next temperature, and when the previous answer was wrong the teacher message carries TEACHER_RETRY_NOTE
with that answer (never the gold one). Each attempt is scored with the single scoring function (success =
F1 of 1) and filtered: canonical form, except that invalid actions and failed FINDs in between are kept
(error recovery) unless the same invalid action error occurs twice. Actor prompts are stored for passing
attempts so the SFT set can be built without re-running the grounder.

Run from the repo root (OPENAI_API_KEY in the environment or in .env); scripts/run_tred.sh runs a pilot of 10
questions and then the main run:

    python -m scripts.stage1.teacher --dataset MultiTQ --model gpt-4.1-mini --limit 10 \\
        --out data/tkgqa/stage1/MultiTQ/pilot.jsonl
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
import threading
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from agent_system.environments.env_package.tkgqa.offline import env_kwargs, make_manager, max_steps_for, prompt_version
from scripts.common import REPO_ROOT, load_env_file
from tkgqa.analysis import Analyzer
from tkgqa.answers import infer_gran, score_episode
from tkgqa.canonical import check_canonical
from tkgqa.executor import Session
from tkgqa.fewshot import fewshot_version, render_fewshot, teacher_message
from tkgqa.grounder import Grounder
from tkgqa.kg import TemporalKG
from tkgqa.kg_vocab import resolve_dataset_name
from utils.openai_api import chat_completion_with_retry, create_openai_client, extract_message_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", default=None, help="default: data/tkgqa/splits/<dataset>/stage1.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="number of questions; 0 = all")
    parser.add_argument("--exclude", nargs="*", default=[], help="skip questions that appear in these runs (jsonl)")
    parser.add_argument("--budget_from", default=None,
                        help="earlier run (jsonl); questions per qtype are allotted in inverse proportion to its pass rate")
    parser.add_argument("--min_pass_rate", type=float, default=0.1, help="floor on pass rates used by --budget_from")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--temperatures", default="0.0,0.7", help="attempt i uses the i-th value (last one repeats)")
    parser.add_argument("--model", default=None, help="default: $OPENAI_MODEL or gpt-4.1-mini")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_cost", type=float, default=None,
                        help="USD cap for this run (resumed runs count their earlier usage); no new API call starts beyond it")
    parser.add_argument("--price_in", type=float, default=0.4, help="USD per 1M uncached prompt tokens (gpt-4.1-mini)")
    parser.add_argument("--price_cached", type=float, default=0.1, help="USD per 1M cached prompt tokens (gpt-4.1-mini)")
    parser.add_argument("--price_out", type=float, default=1.6, help="USD per 1M completion tokens (gpt-4.1-mini)")
    parser.add_argument("--api_retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache_shards", type=int, default=16,
                        help="prompt_cache_key shards; all steps of a question share one key (0 disables)")
    parser.add_argument("--max_steps", type=int, default=None, help="default: the dataset's limit (offline.max_steps_for)")
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "datasets"))
    parser.add_argument("--store_root", default=os.path.join(REPO_ROOT, "stores"))
    parser.add_argument("--device", default="cpu", help="device for the bge-m3 grounder (cuda: set CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true", help="continue the questions of --out (attempts already there are kept)")
    parser.add_argument("--per_type_cap", type=int, default=None,
                        help="stop generating a question type once it has this many accepted questions (balance by accepted samples)")
    parser.add_argument("--count_from", nargs="*", default=[],
                        help="earlier runs whose accepted questions count towards --per_type_cap (e.g. the pilot)")
    return parser.parse_args()


def type_weights(path: str, floor: float) -> Dict[str, float]:
    """1 / (question-level pass rate) per qtype from an earlier run."""
    seen, passed = defaultdict(set), defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            seen[r["qtype"]].add(r["uid"])
            if r["passed"]:
                passed[r["qtype"]].add(r["uid"])
    return {t: 1.0 / max(len(passed[t]) / len(uids), floor) for t, uids in seen.items()}


def select_questions(path: str, limit: int, weights: Optional[Dict[str, float]] = None,
                     exclude: Optional[set] = None) -> List[Dict]:
    """Round-robin over question types (or quotas from --budget_from weights) in split-file order."""
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if exclude:
        records = [r for r in records if r["uid"] not in exclude]
    if not limit or limit >= len(records):
        return records
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:  # split files are already shuffled
        groups[r["qtype"]].append(r)
    if weights:
        total = sum(weights.get(t, 1.0) for t in groups)
        quota = {t: round(limit * weights.get(t, 1.0) / total) for t in groups}
        return [r for t, g in sorted(groups.items()) for r in g[: quota[t]]]
    picked, queues = [], [list(g) for _, g in sorted(groups.items())]
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def interleave_types(questions: List[Dict]) -> List[Dict]:
    """Interleave question types in proportion to their counts (order inside a type kept), so a run stopped by
    --max_cost at any point keeps the type mix of the selection (e.g. the --budget_from quotas)."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for q in questions:
        groups[q["qtype"]].append(q)
    keyed = [((i + 0.5) / len(g), t, i, q) for t, g in groups.items() for i, q in enumerate(g)]
    return [q for *_, q in sorted(keyed, key=lambda x: x[:3])]


class Context:
    def __init__(self, args, kg, grounder, client, fewshot, model, temperatures):
        self.args, self.kg, self.grounder, self.client = args, kg, grounder, client
        self.fewshot, self.model, self.temperatures = fewshot, model, temperatures
        self.prompt_version = prompt_version()
        self.spent = 0.0
        self.cost_lock = threading.Lock()
        self.accepted: Dict[str, int] = defaultdict(int)  # accepted questions per type (for --per_type_cap)
        self.fewshot_version = fewshot_version(fewshot)
        self.analyzer = Analyzer(kg)


def usage_cost(args, usage: Dict) -> float:
    cached = usage.get("cached_tokens", 0)
    return ((usage.get("prompt_tokens", 0) - cached) * args.price_in + cached * args.price_cached
            + usage.get("completion_tokens", 0) * args.price_out) / 1e6


def over_budget(ctx: Context) -> bool:
    return ctx.args.max_cost is not None and ctx.spent >= ctx.args.max_cost


def type_full(ctx: Context, qtype: str) -> bool:
    return ctx.args.per_type_cap is not None and ctx.accepted[qtype] >= ctx.args.per_type_cap


def run_question(ctx: Context, q: Dict, done: List[Dict]) -> List[Dict]:
    """The remaining attempts of a question, given its attempts already in the run (``done``)."""
    records = sorted(done, key=lambda r: r["attempt"])
    new = []
    while (len(records) < ctx.args.attempts and not any(r["passed"] for r in records) and not over_budget(ctx)
           and not type_full(ctx, q["qtype"])):
        prev = records[-1] if records else None
        prev_answer = None
        if prev is not None and not prev["score"]["success"] and prev["error"] is None:
            prev_answer = prev["answer_text"]
        record = run_attempt(ctx, q, len(records), prev_answer)
        records.append(record)
        new.append(record)
        if record["passed"]:
            with ctx.cost_lock:
                ctx.accepted[q["qtype"]] += 1
        if record["error"] is not None:  # API failure: leave the question for --resume
            break
    return new


def run_attempt(ctx: Context, q: Dict, attempt: int, prev_answer: Optional[str] = None) -> Dict:
    temperature = ctx.temperatures[min(attempt, len(ctx.temperatures) - 1)]
    manager = make_manager(ctx.kg, ctx.grounder, 1, max_steps=ctx.args.max_steps)
    obs, _ = manager.reset([env_kwargs(q)])
    session = manager.envs.sessions[0]
    prompt = obs["text"][0]
    steps, prompts = [], []
    usage = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
    error = None
    extra = {}
    if ctx.args.cache_shards > 0:  # same key for every step of a question -> its prefix stays cached
        shard = zlib.crc32(q["uid"].encode()) % ctx.args.cache_shards
        extra["prompt_cache_key"] = f"tkgqa-s0-{ctx.prompt_version}-{shard}"
    while not session.done:
        if over_budget(ctx):  # recorded as an error: dropped from the SFT set, picked up again by --resume
            error = "budget_exhausted"
            break
        try:
            response = chat_completion_with_retry(
                client=ctx.client,
                model=ctx.model,
                messages=teacher_message(ctx.fewshot, prompt, prev_answer),
                temperature=temperature,
                max_completion_tokens=ctx.args.max_tokens,
                retries=ctx.args.api_retries,
                return_response=True,
                **extra,
            )
        except Exception as exc:  # recorded; the attempt is dropped from the SFT set
            error = repr(exc)
            break
        text = extract_message_text(response)
        if getattr(response, "usage", None) is not None:
            details = getattr(response.usage, "prompt_tokens_details", None)
            call = {"prompt_tokens": int(response.usage.prompt_tokens or 0),
                    "completion_tokens": int(response.usage.completion_tokens or 0),
                    "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0)}
            for key, value in call.items():
                usage[key] += value
            with ctx.cost_lock:
                ctx.spent += usage_cost(ctx.args, call)
        usage["calls"] += 1
        next_obs, _, _, infos = manager.step([text])
        last = session.log[-1]
        steps.append({"response": text, "action": last.action, "status": last.status, "obs": last.obs,
                      "format_ok": bool(infos[0]["is_action_valid"]) or not infos[0]["executor_valid"]})
        prompts.append(prompt)
        prompt = next_obs["text"][0]

    record = finish_record(ctx, q, attempt, temperature, session, steps, prompts, usage, error)
    record["retry_note"] = prev_answer is not None
    return record


def stage1_filter(session: Session, steps: List[Dict], score: Dict, error) -> Tuple[List[str], List[str]]:
    """(reject codes, diagnostic codes) of one attempt. Accepted = correct final answer (F1 = 1), a response format
    the SFT target can be read from (one <action> block), and no invalid action error repeated. Invalid actions,
    failed FINDs and skipped variable numbers in between are kept (error recovery); the other canonical rules
    (tkgqa.canonical) are recorded as diagnostics only."""
    if error:
        return ["api_error"], []
    diagnostics = [v for v in check_canonical(session) if v not in ("invalid_step", "numbering")]
    reject = []
    if not score["success"]:
        reject.append("wrong_answer" if session.answer is not None else "no_answer")
    if "format" in diagnostics or not all(step.get("format_ok", True) for step in steps):
        reject.append("format")  # not one <action> block (check_canonical) or rejected by the projection
    invalid_errors = Counter(step["obs"] for step in steps if step["status"] == "invalid")
    if any(n >= 2 for n in invalid_errors.values()):
        reject.append("repeated_invalid")  # the same invalid action error twice
    return reject, [v for v in diagnostics if v != "format"]


FILTER_VERSION = hashlib.sha1((inspect.getsource(stage1_filter) + inspect.getsource(check_canonical)).encode()).hexdigest()[:8]


def answer_text(session: Session) -> str:
    """The submitted answer as the executor showed it ("Answer submitted: ..."), or "(no answer)"."""
    if session.answer is None:
        return "(no answer)"
    obs = session.log[-1].obs
    return obs[len("Answer submitted: "):] if obs.startswith("Answer submitted: ") else ", ".join(map(str, session.answer))


def finish_record(ctx: Context, q: Dict, attempt: int, temperature: float, session: Session,
                  steps: List[Dict], prompts: List[str], usage: Dict, error) -> Dict:
    """Score, analyze and canonical-check one attempt."""
    gold, kind = q["answers"], q["kind"]
    score = score_episode(session.answer, gold, kind)
    analysis = ctx.analyzer.analyze(session, gold, kind, infer_gran(gold) if kind in ("time", "span") else None)
    reject, diagnostics = stage1_filter(session, steps, score, error)
    passed = not reject
    record = {
        "uid": q["uid"],
        "dataset": q["dataset"],
        "qtype": q["qtype"],
        "answer_type": q["answer_type"],
        "question": q["question"],
        "answers": gold,
        "kind": kind,
        "attempt": attempt,
        "temperature": temperature,
        "model": ctx.model,
        "prompt_version": ctx.prompt_version,
        "fewshot_version": ctx.fewshot_version,
        "max_steps": ctx.args.max_steps,
        "steps": steps,
        "answer": session.answer,
        "answer_text": answer_text(session),
        "score": score,
        "failure_type": analysis["failure_type"],
        "suspect_step": analysis["suspect_step"],
        "coverage": analysis["coverage"],
        "filter_version": FILTER_VERSION,
        "reject_codes": reject,
        "canonical_violations": diagnostics,  # recorded, not used for rejection (see stage1_filter)
        "passed": passed,
        "find_retry": any(s["status"] == "find_failed" for s in steps),
        "invalid_steps": sum(s["status"] == "invalid" for s in steps),
        "num_steps": len(steps),
        "truncated": session.truncated,
        "usage": usage,
        "error": error,
    }
    if passed:
        record["prompts"] = prompts
    return record


def main() -> int:
    args = parse_args()
    load_env_file(os.path.join(REPO_ROOT, ".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (environment or .env).", file=sys.stderr)
        return 1
    dataset = resolve_dataset_name(args.dataset)
    args.max_steps = max_steps_for(dataset) if args.max_steps is None else args.max_steps
    model = args.model or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
    temperatures = [float(t) for t in args.temperatures.split(",")]
    questions_path = args.questions or os.path.join(REPO_ROOT, "data", "tkgqa", "splits", dataset, "stage1.jsonl")
    weights = type_weights(args.budget_from, args.min_pass_rate) if args.budget_from else None
    exclude = set()
    for path in args.exclude:
        with open(path, encoding="utf-8") as f:
            exclude |= {json.loads(line)["uid"] for line in f}
    questions = interleave_types(select_questions(questions_path, args.limit, weights, exclude))
    if weights:
        print("questions per qtype:", dict(sorted(((t, sum(q["qtype"] == t for q in questions)) for t in weights))))

    done: Dict[str, List[Dict]] = defaultdict(list)
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("error"):
                    done[r["uid"]].append(r)
    elif os.path.exists(args.out):
        print(f"{args.out} exists; pass --resume to continue it or choose another --out", file=sys.stderr)
        return 1
    tasks = [q for q in questions if len(done[q["uid"]]) < args.attempts and not any(r["passed"] for r in done[q["uid"]])]
    accepted_uids: Dict[str, set] = defaultdict(set)
    for path in list(args.count_from) + ([args.out] if args.resume and os.path.exists(args.out) else []):
        with open(path, encoding="utf-8") as f:
            for r in map(json.loads, f):
                if r["passed"]:
                    accepted_uids[r["qtype"]].add(r["uid"])
    earlier_cost = 0.0
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            earlier_cost = sum(usage_cost(args, json.loads(line).get("usage") or {}) for line in f)

    t0 = time.time()
    kg = TemporalKG.load(dataset, args.data_root)
    grounder = Grounder.from_store_root(kg, args.store_root, device=args.device)
    fewshot = render_fewshot()
    client = create_openai_client(timeout=120)
    ctx = Context(args, kg, grounder, client, fewshot, model, temperatures)
    ctx.spent = earlier_cost
    for qtype, uids in accepted_uids.items():
        ctx.accepted[qtype] = len(uids)
    if args.per_type_cap is not None:
        print("accepted per type at start:", dict(sorted(ctx.accepted.items())), f"(cap {args.per_type_cap})", flush=True)
    print(f"[{dataset}] {len(questions)} questions, up to {args.attempts} attempts each -> {len(tasks)} questions to run "
          f"(model={model}, temperatures={temperatures}, setup {time.time() - t0:.1f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "model": model, "questions_path": questions_path, "num_questions": len(questions),
                   "prompt_version": ctx.prompt_version, "fewshot_version": ctx.fewshot_version}, f, indent=2)

    lock = threading.Lock()
    stats = {"done": 0, "attempts": 0, "passed": 0, "retry_passed": 0, "errors": 0}
    with open(args.out, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_question, ctx, q, done[q["uid"]]) for q in tasks]
        for future in as_completed(futures):
            records = future.result()
            with lock:
                for record in records:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                stats["done"] += 1
                stats["attempts"] += len(records)
                stats["passed"] += int(any(r["passed"] for r in records))
                stats["retry_passed"] += int(any(r["passed"] and r["attempt"] > 0 for r in records))
                stats["errors"] += sum(r["error"] is not None for r in records)
                if stats["done"] % 10 == 0 or stats["done"] == len(tasks):
                    n = stats["done"]
                    print(f"  {n}/{len(tasks)} questions | {stats['attempts']} attempts | passed {stats['passed'] / n:.3f} "
                          f"(on a retry {stats['retry_passed'] / n:.3f}) | errors {stats['errors']} | "
                          f"${ctx.spent:.2f}{'' if args.max_cost is None else f' of ${args.max_cost:.2f}'} | {time.time() - t0:.0f}s",
                          flush=True)
    if args.per_type_cap is not None:
        print("accepted per type:", dict(sorted(ctx.accepted.items())), flush=True)
    print(f"done: ${ctx.spent:.2f} spent (estimate from usage at the --price_* rates)"
          + (" -- budget reached, resume with a higher --max_cost to continue" if over_budget(ctx) else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
