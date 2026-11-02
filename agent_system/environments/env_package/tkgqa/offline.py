"""The TKGQA environment outside the trainer: Stage 1 teacher runs, Stage 2 collection, evaluation.

Every actor prompt comes from TKGQAEnvironmentManager (template + TKGQAMemory history), the same object RL
uses, so SFT, RL and evaluation prompts are token-identical. ``prompt_version`` hashes the templates, the operation
document and the observation format; it is recorded with generated data so versions never mix.
"""

import hashlib
import os
import re
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from agent_system.environments.env_manager import TKGQAEnvironmentManager
from agent_system.environments.env_package.tkgqa.envs import TKGQAEnvs
from agent_system.environments.env_package.tkgqa.projection import tkgqa_projection
from agent_system.environments.prompts.tkgqa import TKGQA_OPERATIONS_SPEC, TKGQA_TEMPLATE, TKGQA_TEMPLATE_NO_HIS
from tkgqa.executor import OBSERVATION_VERSION

MAX_STEPS = 10  # the history always keeps every step
MAX_STEPS_BY_DATASET = {"TimelineCronQR": 15}  # its multi-hop questions need more steps (scripts/env.sh: the same table)


def max_steps_for(dataset: str) -> int:
    """Step limit of a dataset (canonical name, as TemporalKG.dataset and resolve_dataset_name give it);
    TKGQA_MAX_STEPS_<dataset> overrides it for one dataset."""
    override = os.environ.get(f"TKGQA_MAX_STEPS_{dataset}")
    return int(override) if override else MAX_STEPS_BY_DATASET.get(dataset, MAX_STEPS)


_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)


def prompt_version() -> str:
    """Hash of the actor prompt and the observation format."""
    parts = [TKGQA_TEMPLATE_NO_HIS, TKGQA_TEMPLATE, TKGQA_OPERATIONS_SPEC, OBSERVATION_VERSION, "response=action_only"]
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def make_manager(kg, grounder, n_envs: int, max_steps: Optional[int] = None,
                 history_length: Optional[int] = None) -> TKGQAEnvironmentManager:
    """A manager over ``n_envs`` environments sharing ``kg`` and ``grounder`` (as make_envs builds it).
    max_steps defaults to the dataset's limit (max_steps_for), history_length to max_steps (the full history)."""
    max_steps = max_steps_for(kg.dataset) if max_steps is None else max_steps
    history_length = max_steps if history_length is None else history_length
    config = OmegaConf.create({"env": {"max_steps": max_steps, "history_length": history_length, "tkgqa": {}}})
    envs = TKGQAEnvs(env_num=n_envs, group_n=1, env_config=config.env, kg=kg, grounder=grounder)
    return TKGQAEnvironmentManager(envs, tkgqa_projection, config)


def env_kwargs(q: Dict[str, Any]) -> Dict[str, Any]:
    """The env_kwargs of a split record (as in the RL parquet)."""
    return {"question": q["question"], "answers": list(q["answers"]), "kind": q["kind"], "uid": q["uid"],
            "qtype": q.get("qtype"), "data_source": q.get("dataset", "tkgqa")}


def chat_prompt_ids(tokenizer, prompt: str) -> List[int]:
    """Token ids of a prompt as SFT (verl SFTDataset) and the RL rollout loop build them: one user turn,
    chat template with the generation prompt, no extra special tokens."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def normalize_response(text: str) -> str:
    """SFT target in the format the actor prompt asks for: ``<action> OP </action>``."""
    return f"<action> {_ACTION_RE.search(text).group(1).strip()} </action>"
