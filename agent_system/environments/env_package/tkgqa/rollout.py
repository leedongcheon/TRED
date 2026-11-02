"""Batched policy rollouts in the TKGQA environment (vLLM), shared by evaluation and Stage 2 collection.

All rollouts run in one TKGQAEnvironmentManager, so every step uses the RL prompt and the same token ids
as SFT/RL (offline.chat_prompt_ids); generation stops at </action>, where the projection cuts anyway.
Each request has its own seed, so a rollout is reproducible given (seed, question uid, rollout index, step).
"""

import os
import zlib
from typing import Callable, Dict, List, Optional, Sequence

from agent_system.environments.env_package.tkgqa.offline import chat_prompt_ids, env_kwargs, make_manager
from tkgqa.diversity import decision_probs


def load_policy(model_path: str, gpu_memory_utilization: float = 0.6, max_model_len: int = 8192, seed: int = 0):
    # vLLM's engine process must be spawned: with the grounder on CPU nothing initializes CUDA before vLLM picks its
    # start method, and vLLM then initializes CUDA itself before forking (the child fails).
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
        dtype="bfloat16",  # SFT checkpoints may record float32 in their config
    )
    return llm, llm.get_tokenizer()


def run_rollouts(
    llm,
    tokenizer,
    kg,
    grounder,
    questions: Sequence[Dict],
    n: int,
    temperature: float,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    max_steps: Optional[int] = None,  # None: the dataset's limit (offline.max_steps_for)
    seed: int = 0,
    logprobs: int = 0,
    keep_prompts: bool = False,
    log: Optional[Callable[[str], None]] = print,
    prompt_fn: Optional[Callable[[Dict, str], str]] = None,
) -> List[Dict]:
    """Roll out ``n`` episodes per question; returns [{"q", "r", "session", "score", "hindsight", "step_hindsights",
    "auto_hints", "decisions", "prompts", "responses"}] (step_hindsights / auto_hints: the env's analyzer-only lines). ``prompt_fn(q, prompt)`` may rewrite each actor prompt (e.g. insert a skill)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    items = [(q, r) for q in questions for r in range(n)]
    manager = make_manager(kg, grounder, len(items), max_steps=max_steps)
    obs, _ = manager.reset([env_kwargs(q) for q, _ in items])
    current = obs["text"]
    sessions = manager.envs.sessions
    decisions: List[list] = [[] for _ in items]
    prompts: List[list] = [[] for _ in items]
    responses: List[list] = [[] for _ in items]
    step = 0
    while True:
        active = [i for i, session in enumerate(sessions) if not session.done]
        if not active:
            break
        step += 1
        requests, params = [], []
        for i in active:
            q, r = items[i]
            prompt = current[i] if prompt_fn is None else prompt_fn(q, current[i])
            if keep_prompts:
                prompts[i].append(prompt)
            requests.append(TokensPrompt(prompt_token_ids=chat_prompt_ids(tokenizer, prompt)))
            params.append(SamplingParams(
                temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=["</action>"],
                include_stop_str_in_output=True, seed=zlib.crc32(f"{seed}:{q['uid']}:{r}:{step}".encode()),
                logprobs=logprobs or None,
            ))
        outputs = llm.generate(requests, params, use_tqdm=False)
        texts = [""] * len(items)  # finished environments ignore their action
        for i, output in zip(active, outputs):
            completion = output.outputs[0]
            texts[i] = completion.text
            responses[i].append(completion.text)
            if completion.logprobs:
                pieces = []
                for token_id, top in zip(completion.token_ids, completion.logprobs):
                    chosen = top[token_id]
                    alternatives = {lp.decoded_token: lp.logprob for tid, lp in top.items() if tid != token_id and lp.decoded_token}
                    pieces.append((chosen.decoded_token or "", chosen.logprob, alternatives))
                decisions[i] += decision_probs(pieces)
        current = manager.step(texts)[0]["text"]
        if log:
            log(f"  step {step}: {len(active)} active")
    return [
        {"q": q, "r": r, "session": sessions[i], "score": manager.envs.scores[i], "hindsight": manager.envs.hindsights[i],
         "step_hindsights": list(manager.envs.step_lines[i]), "auto_hints": manager.envs.hints[i],
         "decisions": decisions[i], "prompts": prompts[i], "responses": responses[i]}
        for i, (q, r) in enumerate(items)
    ]
