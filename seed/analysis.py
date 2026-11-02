import ast
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gigpo import core_gigpo
from seed.prompting import validate_skill_mode
from tkgqa.hindsight import DROPPED_GOLD_INTRO, DROPPED_GOLD_MARK
from tkgqa.operations import OPERATIONS_LINE
from utils import build_prompt_dict, chat_completion_with_retry, create_openai_client


logger = logging.getLogger(__name__)


_TASK_DESCRIPTION_PATTERNS = (
    re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your task is:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your current task is:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your question:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your goal is to\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"You are [^\n]*?\bhelping to\s*(.+?)(?:\n|$)", re.IGNORECASE),
)

_ANALYSIS_PROMPT_VERSIONS = {"seed"}

# A fix of several steps is prescribed by its first operation only (the policy writes the rest), so a decision-level skill never
# packs two operations into one (e.g. a FILTER whose reference is a fact set instead of a TIME of it).
FIRST_ACTION_RULE = ("If the fix takes more than one step, write only the first operation to take at this decision "
                     "(e.g. \"t1 = TIME(f2, first)\"); the agent continues the following steps itself. ")
# A decision-level skill uses only what the agent could see: without this rule, decision-level skills write anchor dates known only from
# the hindsight and FILTERs finer than the question's time. Checked as well (tkgqa.skill_filter.context_violations).
CONTEXT_RULE = ("The corrected operation uses only values the agent could see: names, dates and numbers from the question "
                "or the earlier observations, never ones known only from the Hindsight lines. If the decision needs the time "
                "of another event, write the operation that retrieves it (a FIND of that event, or a TIME), not its date. "
                "A FILTER on a time the question names uses the granularity of that time (a year -> gran=year). ")


def validate_analysis_prompt_version(version: object) -> str:
    value = str(version or "seed").strip()
    if value not in _ANALYSIS_PROMPT_VERSIONS:
        raise ValueError(
            f"Unsupported SEED analysis_prompt_version: {value!r}. "
            f"Expected one of {sorted(_ANALYSIS_PROMPT_VERSIONS)}."
        )
    return value


def _clean_task_description(task_description: object) -> str:
    task = " ".join(str(task_description or "").split())
    return task.strip()


def _extract_task_description_from_text(text: object) -> str:
    text = str(text or "")
    for pattern in _TASK_DESCRIPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            task_description = _clean_task_description(match.group(1))
            if task_description:
                return task_description
    return ""


def build_traj_step_indices(traj_index: Sequence) -> np.ndarray:
    """Return per-sample step indices in trajectory order."""
    counters: Dict[object, int] = defaultdict(int)
    step_indices = []
    for traj_uid in traj_index:
        step_indices.append(counters[traj_uid])
        counters[traj_uid] += 1
    return np.asarray(step_indices, dtype=np.int64)


def select_critical_steps_by_stats(
    step_rewards: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    min_group_size: int = 2,
    var_quantile: float = 0.75,
    topk_per_traj: int = 1,
    below_group_mean_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Select critical steps using reward statistics of each observation group.

    A step becomes a candidate when its observation group has sufficiently high
    return variance. Candidate steps are ranked by how much they underperform
    the group mean (or deviate from it), and we keep the top-k per trajectory.
    """
    if topk_per_traj <= 0 or len(step_rewards) == 0:
        return np.zeros(len(step_rewards), dtype=bool), {
            "num_groups": 0.0,
            "eligible_groups": 0.0,
            "selected_steps": 0.0,
        }

    step_group_uids = core_gigpo.build_step_group(
        anchor_obs=anchor_obs,
        index=index,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    step_rewards_np = step_rewards.detach().cpu().numpy().astype(np.float32)
    step_indices = build_traj_step_indices(traj_index)

    group_to_indices: Dict[object, List[int]] = defaultdict(list)
    for sample_idx, group_uid in enumerate(step_group_uids):
        group_to_indices[group_uid].append(sample_idx)

    group_stats = {}
    eligible_group_vars: List[float] = []
    for group_uid, sample_indices in group_to_indices.items():
        rewards = step_rewards_np[sample_indices]
        group_mean = float(np.mean(rewards))
        group_var = float(np.var(rewards))
        group_stats[group_uid] = {
            "mean": group_mean,
            "var": group_var,
            "size": float(len(sample_indices)),
        }
        if len(sample_indices) >= min_group_size:
            eligible_group_vars.append(group_var)

    if eligible_group_vars:
        clipped_quantile = min(max(var_quantile, 0.0), 1.0)
        var_cutoff = float(np.quantile(np.asarray(eligible_group_vars, dtype=np.float32), clipped_quantile))
    else:
        var_cutoff = float("inf")

    sample_scores = np.zeros(len(step_rewards_np), dtype=np.float32)
    candidate_mask = np.zeros(len(step_rewards_np), dtype=bool)
    for sample_idx, group_uid in enumerate(step_group_uids):
        group_stat = group_stats[group_uid]
        group_mean = group_stat["mean"]
        group_var = group_stat["var"]
        group_size = int(group_stat["size"])
        if group_size < min_group_size or group_var < var_cutoff or group_var <= 0.0:
            continue

        reward = step_rewards_np[sample_idx]
        if below_group_mean_only:
            deviation = max(group_mean - reward, 0.0)
        else:
            deviation = abs(reward - group_mean)
        if deviation <= 0.0 and below_group_mean_only:
            continue

        candidate_mask[sample_idx] = True
        sample_scores[sample_idx] = group_var * (deviation + 1e-6)

    critical_mask = np.zeros(len(step_rewards_np), dtype=bool)
    traj_to_candidates: Dict[object, List[int]] = defaultdict(list)
    for sample_idx, is_candidate in enumerate(candidate_mask):
        if is_candidate:
            traj_to_candidates[traj_index[sample_idx]].append(sample_idx)

    for traj_uid, sample_indices in traj_to_candidates.items():
        ranked = sorted(
            sample_indices,
            key=lambda i: (sample_scores[i], step_indices[i]),
            reverse=True,
        )
        for sample_idx in ranked[:topk_per_traj]:
            critical_mask[sample_idx] = True

    metrics = {
        "num_groups": float(len(group_to_indices)),
        "eligible_groups": float(sum(
            1
            for group_stat in group_stats.values()
            if int(group_stat["size"]) >= min_group_size and group_stat["var"] >= var_cutoff and group_stat["var"] > 0.0
        )),
        "selected_steps": float(critical_mask.sum()),
        "variance_cutoff": 0.0 if not np.isfinite(var_cutoff) else float(var_cutoff),
    }
    return critical_mask, metrics


def build_episode_records(
    tokenizer,
    obs_texts: Sequence,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    traj_index: Sequence,
    step_indices: Sequence,
    step_rewards: Optional[torch.Tensor] = None,
    obs_raws: Optional[Sequence] = None,
    obs_images: Optional[Sequence] = None,
    action_valids: Optional[Sequence] = None,
    hindsights: Optional[Sequence] = None,
    episode_extras: Optional[Dict[object, Dict[str, str]]] = None,
) -> Dict[object, List[Dict[str, object]]]:
    """Decode trajectories into step records for analysis.

    `observation` prefers raw text feedback from the environment when
    available (for example `anchor_obs` in rollout batches). The prompt-form
    observation is preserved in `observation_prompt` for debugging.
    TKGQA: `hindsights` is the env's per-step Hindsight line of each sample; `episode_extras` maps traj_uid to
    the env's end-of-episode fields (final_observation, auto_hints), set on the last step of the trajectory.
    """
    if step_rewards is not None:
        reward_np = step_rewards.detach().cpu().numpy().astype(np.float32)
    else:
        reward_np = None

    episodes: Dict[object, List[Dict[str, object]]] = defaultdict(list)
    responses_cpu = responses.detach().cpu()
    response_mask_cpu = response_mask.detach().cpu()
    for sample_idx, traj_uid in enumerate(traj_index):
        valid_len = int(response_mask_cpu[sample_idx].sum().item())
        response_text = tokenizer.decode(
            responses_cpu[sample_idx][:valid_len],
            skip_special_tokens=True,
        )
        prompt_observation = str(obs_texts[sample_idx])
        raw_observation = prompt_observation
        if obs_raws is not None:
            candidate_raw_observation = obs_raws[sample_idx]
            if isinstance(candidate_raw_observation, (str, np.str_)):
                raw_observation = str(candidate_raw_observation)
        step_record = {
            "step_index": int(step_indices[sample_idx]),
            "observation": raw_observation,
            "observation_prompt": prompt_observation,
            "response": response_text,
        }
        if obs_images is not None:
            try:
                observation_image = obs_images[sample_idx]
            except Exception:
                observation_image = None
            if observation_image is not None:
                step_record["observation_image"] = observation_image
                step_record["has_observation_image"] = True
        task_description = (
            _extract_task_description_from_text(prompt_observation)
            or _extract_task_description_from_text(raw_observation)
        )
        if task_description:
            step_record["task_description"] = task_description
        if reward_np is not None:
            step_record["step_reward"] = float(reward_np[sample_idx])
        if action_valids is not None:
            step_record["action_valid"] = bool(action_valids[sample_idx])
        if hindsights is not None:
            step_record["hindsight"] = str(hindsights[sample_idx])
        episodes[traj_uid].append(step_record)

    for traj_uid in episodes:
        episodes[traj_uid].sort(key=lambda step: step["step_index"])
    for traj_uid, steps in episodes.items():
        if episode_extras is not None and traj_uid in episode_extras:
            steps[-1]["final_observation"] = episode_extras[traj_uid].get("final_observation", "")
            steps[-1]["auto_hints"] = episode_extras[traj_uid].get("auto_hints", "")
    return episodes


class SEEDEpisodeAnalyzer:
    """
    Analyze trajectories for reusable episode-level and critical-step skills.

    The analyzer uses an OpenAI-compatible JSON analysis backend or the
    on-policy rollout model when the trainer selects ``policy_vllm``.
    ``azure`` remains a legacy alias for ``openai``.
    """

    def __init__(
        self,
        backend: str = "openai",
        max_completion_tokens: int = 4096,
        max_step_skills_per_traj: int = 1,
        skill_mode: str = "episode_step",
        analysis_prompt_version: str = "seed",
        include_episode_summary: bool = True,
        render_response: str = "action_only",
    ):
        self.requested_backend = backend
        if backend == "azure":
            logger.warning("SEED backend='azure' is deprecated; using the OpenAI-compatible backend instead.")
            backend = "openai"
        elif backend not in {"openai", "policy_vllm"}:
            raise ValueError(f"Unsupported SEED backend: {backend}")

        self.backend = backend
        self.max_completion_tokens = max_completion_tokens
        self.max_step_skills_per_traj = max(int(max_step_skills_per_traj), 0)
        self.skill_mode = validate_skill_mode(skill_mode)
        self.analysis_prompt_version = validate_analysis_prompt_version(analysis_prompt_version)
        self.include_episode_summary = bool(include_episode_summary)
        if render_response not in ("action_only", "full"):
            raise ValueError(f"Unsupported SEED analysis render_response: {render_response}")
        self.render_response = render_response  # "action_only": the <action> block only; "full": the whole response
        self.client = None
        self.model = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash")

        logger.info(
            "Initialized SEEDEpisodeAnalyzer with requested_backend=%s, resolved_backend=%s, model=%s, max_completion_tokens=%s, max_step_skills_per_traj=%s, skill_mode=%s, analysis_prompt_version=%s, include_episode_summary=%s",
            self.requested_backend,
            self.backend,
            self.model,
            self.max_completion_tokens,
            self.max_step_skills_per_traj,
            self.skill_mode,
            self.analysis_prompt_version,
            self.include_episode_summary,
        )

    def _get_openai_client(self):
        if self.backend != "openai":
            raise ValueError(f"SEED backend {self.backend!r} does not use an OpenAI client.")
        if self.client is None:
            self.client = create_openai_client(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        return self.client

    def analyze_episode(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, object]:
        if self.backend != "openai":
            raise ValueError(f"analyze_episode is only implemented for openai, got {self.backend!r}.")
        return self._analyze_episode_with_openai(
            steps=steps,
            candidate_step_indices=candidate_step_indices,
            task_description=task_description,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
        )

    def _analysis_has_required_skill(self, parsed: Dict[str, object]) -> bool:
        if self._is_nontrivial_skill_text(parsed.get("step_skill", "")):
            return True  # a failure's decision-level skill alone (the Stage 2 targets blank an episode-level skill the TKGQA filter rejects)
        return self._is_nontrivial_skill_text(parsed.get("episode_skill", ""))

    @staticmethod
    def _is_nontrivial_skill_text(value: object) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if text in {"...", "…"} or re.fullmatch(r"[\s.。…,_-]+", text):
            return False
        return len(text.split()) >= 5

    def _analyze_episode_with_openai(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, object]:
        candidate_list = (
            [int(step["step_index"]) for step in steps]
            if candidate_step_indices is None
            else [int(idx) for idx in candidate_step_indices]
        )
        prompt = self._build_episode_analysis_prompt(
            steps=steps,
            candidate_step_indices=candidate_list,
            task_description=task_description,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
        )
        parse_attempts = self._get_parse_attempts()
        content = ""
        last_error: Optional[BaseException] = None
        for parse_attempt in range(parse_attempts):
            prompt_for_attempt = (
                prompt
                if parse_attempt == 0
                else self._build_json_retry_prompt(
                    original_prompt=prompt,
                    invalid_response=content,
                    error=last_error,
                )
            )
            content = chat_completion_with_retry(
                client=self._get_openai_client(),
                model=self.model,
                prompt=prompt_for_attempt,
                retries=max(1, int(os.environ.get("OPENAI_API_RETRIES", "5"))),
                retry_delay=float(os.environ.get("OPENAI_API_RETRY_DELAY", "1.0")),
                max_completion_tokens=self.max_completion_tokens,
            )
            try:
                parsed = self._parse_analysis_response(content)
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "SEED analyzer JSON parse failed on attempt %s/%s: %s",
                    parse_attempt + 1,
                    parse_attempts,
                    exc,
                )
                continue
            if not self._analysis_has_required_skill(parsed):
                last_error = ValueError("SEED analyzer response missing required field: episode_skill")
                logger.warning(
                    "SEED analyzer JSON parse failed on attempt %s/%s: %s",
                    parse_attempt + 1,
                    parse_attempts,
                    last_error,
                )
                continue

            candidate_step_set = set(candidate_list)
            filtered_step_skills = {
                step_idx: skill
                for step_idx, skill in parsed.get("step_skills", {}).items()
                if step_idx in candidate_step_set
            }
            parsed["step_skills"] = dict(
                list(filtered_step_skills.items())[: self.max_step_skills_per_traj]
            )
            parsed["analysis_backend_requested"] = self.requested_backend
            parsed["analysis_backend_used"] = "openai"
            parsed["analysis_error"] = None
            parsed["analysis_mode"] = analysis_mode
            parsed["analysis_prompt_version"] = self.analysis_prompt_version
            parsed["task_description"] = task_description or self._infer_task_description(steps)
            parsed["llm_prompt"] = prompt_for_attempt
            parsed["llm_raw_output"] = content
            return parsed

        logger.warning(
            "SEED analyzer failed to produce valid JSON after %s attempt(s); skipping teacher skill for this trajectory: %s",
            parse_attempts,
            last_error,
        )
        return {
            "episode_summary": "",
            "episode_skill": "",
            "step_skills": {},
            "analysis_backend_requested": self.requested_backend,
            "analysis_backend_used": "openai",
            "analysis_error": str(last_error) if last_error is not None else "unknown JSON parse error",
            "analysis_mode": analysis_mode,
            "analysis_prompt_version": self.analysis_prompt_version,
            "task_description": task_description or self._infer_task_description(steps),
            "llm_prompt": prompt,
            "llm_raw_output": content,
        }

    def _get_parse_attempts(self) -> int:
        raw_attempts = os.environ.get("SEED_ANALYSIS_PARSE_RETRIES", "2")
        try:
            return max(1, int(raw_attempts))
        except (TypeError, ValueError):
            logger.warning("Invalid SEED_ANALYSIS_PARSE_RETRIES=%r; using 2.", raw_attempts)
            return 2

    def _build_json_retry_prompt(
        self,
        original_prompt: Dict[str, Any],
        invalid_response: str,
        error: Optional[BaseException],
    ) -> Dict[str, Any]:
        messages = original_prompt.get("messages", []) if isinstance(original_prompt, dict) else []
        original_user_prompt = ""
        for message in messages:
            if message.get("role") == "user":
                original_user_prompt = str(message.get("content", ""))
                break
        invalid_preview = str(invalid_response or "")[-2000:]
        if self.skill_mode == "episode_only":
            schema_text = """{
  "episode_summary": "string",
  "episode_skill": "string"
}"""
        else:
            schema_text = """{
  "episode_summary": "string",
  "episode_skill": "string",
  "step_skills": {}
}"""
        retry_prompt = f"""{original_user_prompt}

The previous answer could not be parsed as the required JSON object.
Parse error: {error}

Previous answer:
{invalid_preview}

Return ONLY one valid JSON object with this exact shape:
{schema_text}
"""
        return build_prompt_dict(user_prompt=retry_prompt)

    def _infer_task_description(self, steps: List[Dict[str, object]]) -> str:
        for step in steps:
            task_description = _clean_task_description(step.get("task_description", ""))
            if task_description:
                return task_description

        for step in steps:
            for field_name in ("observation_prompt", "observation"):
                task_description = _extract_task_description_from_text(step.get(field_name, ""))
                if task_description:
                    return task_description
        return ""

    def _build_default_episode_analysis_prompt(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Sequence[int] = (),
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The TKGQA analyzer prompt: five fields (episode_summary, failure_step, failure_type, episode_skill, step_skill)
        over the trajectory with its execution hindsight (Hindsight lines, diagnostic hints, final observation). The same prompt
        serves the Stage 2 labeller (skill_mode episode_only) and the policy analyzer in RL (episode_step), where the
        step_skill becomes the step skill of failure_step (_parse_analysis_response)."""
        task_description = _clean_task_description(task_description) or self._infer_task_description(steps)

        outcome_label = "unknown"
        if episode_success is not None:
            try:
                outcome_label = "success" if float(episode_success) >= 1.0 else "failure"
            except (TypeError, ValueError):
                outcome_label = "unknown"

        if outcome_label == "success":
            episode_skill_instruction = (
                "Write one episode_skill that extracts the successful trajectory into a workflow: "
                "\"When the question <pattern>, <op + key argument>, then ..., then GET <side/gran>.\" "
                "Name the DSL operations and the arguments that made it work (first/last, start/end, "
                "the FILTER relation, gran, PICK, subj/obj). If the episode has invalid or failed steps that "
                "were recovered, you may append one sentence \"If <error observation>, <fix>.\""
            )
        elif outcome_label == "failure":
            episode_skill_instruction = (
                "Write one episode_skill that extracts the failed trajectory into an avoidance rule. "
                "Your episode_skill MUST contain the words \"do not\" and \"instead\": "
                "\"When <signal visible in the question or an observation>, do not <mistake>; instead <fix>.\" "
                "The cause may be an earlier step than the one where recall drops, for example the TIME step "
                "that produced the anchor a later FILTER consumed. "
                "By failure_type: "
                "entity_grounding / wrong_relation -> say which observation to read (FIND candidates, other relations, "
                "relation fallback) and how to choose among them, never the name itself; "
                "find_slot_direction -> say which slot gets ? and on which side the named entity goes; "
                "anchor_from_wrong_set -> say which FIND the TIME must come from (the reference event's, not the candidates')."
            )
        else:
            episode_skill_instruction = (
                "Write one episode_skill distilled from the trajectory. If it succeeded, extract "
                "the successful workflow; if it failed, extract the avoidance rules. "
            )
        summary_instruction = "Write a concise episode_summary."

        # The environment's line of discarded gold-bearing facts (tkgqa.hindsight) goes to its step's Hindsight line.
        auto_hints = steps[-1].get("auto_hints", "") if steps else ""
        hindsight_extra = {}
        if DROPPED_GOLD_MARK in auto_hints:
            kept_lines = []
            for line in auto_hints.split("\n"):
                m = re.match(rf"^{re.escape(DROPPED_GOLD_MARK)}(\d+):(.*)$", line)
                if m:
                    hindsight_extra[int(m.group(1))] = m.group(2)
                else:
                    kept_lines.append(line)
            auto_hints = "\n".join(kept_lines)
        # step_skill: the decision-level skill for failure_step, in the form the policy reads at that decision during RL.
        step_field = (
            "5. step_skill: for a failure, the prescription for failure_step in exactly this form: "
            "\"At this decision, write <corrected DSL operation> instead of <the operation the agent wrote at failure_step>.\" "
            "The corrected operation is one valid DSL operation that keeps the agent's variable names and changes the "
            "decisive choice (the slot that gets ?, the fact set TIME reads, first/last, the FILTER relation, gran, "
            "subj/obj, ...). Write the whole operation with its result variable, of the kind the operation returns "
            "(FIND/FILTER/PICK/EXCLUDE/UNION/INTERSECT -> fN, TIME -> tN, GET/AGG/RANK -> sN), e.g. "
            "\"s1 = GET(f2, obj)\", not \"GET(f2, obj)\". "
            + FIRST_ACTION_RULE
            + CONTEXT_RULE
            + "For a success, an empty string.\n")
        step_rule = ("- step_skill may copy entity and relation names from the question or earlier observations, never the\n"
                     "  gold answer; at most 60 words.\n")
        evidence_intro = (
            "Each step shows the Observation seen before acting, the agent's Response, whether the action was\n"
            "valid, and a Hindsight line for the result of that action (not visible to the agent): recall = fraction\n"
            "of gold answers still reachable from that value; \"gold on subj|obj\" = which side of the facts holds the\n"
            f"gold; \"aux from fN\" = a time value computed from fN, recall does not apply.{DROPPED_GOLD_INTRO}\n"
            "Auto hints below were computed by rules and may be wrong; verify them against the trajectory.\n")
        prompt_text = f"""Analyze the following temporal KGQA agent episode and return ONLY valid JSON.

The agent answers a question by executing one DSL operation per step:
{OPERATIONS_LINE}

{evidence_intro}
You need to complete five fields:
1. {summary_instruction}
2. failure_step: the 0-based step of the decisive mistake, or null for a success.
3. failure_type: one of entity_grounding, wrong_relation, find_slot_direction, anchor_from_wrong_set,
   time_mode, time_shift, filter_rel, gran_mismatch, pick_wrong, over_filter, under_filter, get_field,
   get_time_mode, answer_type, agg_wrong, rank_args, premature_answer, wasted_loop, none.
4. {episode_skill_instruction}
{step_field}
Important constraints:
- Write episode_skill as one short policy-facing skill (at most 60 words), not as retrospective
  explanation of the trajectory.
- Use only valid DSL in episode_skill: no nested operations, no literal assigned to a variable.
- Never recommend an empty answer or giving up; name the earlier choice to revisit instead.
- Never put entity names, dates, relation names, or the answer from this episode in episode_skill;
  question wording (before/after, first/last, how many) is fine.
- Never mention recall, gold, hints, step numbers, or variable numbers (f1, t2, s1) in episode_skill.
{step_rule}- Return only these top-level fields: episode_summary, failure_step, failure_type, episode_skill, step_skill.

Return format:
{{
  "episode_summary": "string",
  "failure_step": 0,
  "failure_type": "string",
  "episode_skill": "string",
  "step_skill": "string"
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Auto hints:
{auto_hints or '(none)'}
- Interaction trajectory:
{self._format_episode_steps(steps, hindsight_extra)}
"""
        return build_prompt_dict(user_prompt=prompt_text)

    def _build_episode_analysis_prompt(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Sequence[int],
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._build_default_episode_analysis_prompt(
            steps=steps,
            candidate_step_indices=candidate_step_indices,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
            task_description=task_description,
        )

    def _parse_analysis_response(self, response: str) -> Dict[str, object]:
        parsed = self._extract_json_object(response)
        step_skills_raw = parsed.get("step_skills", {}) or {}
        step_skills = {}
        if isinstance(step_skills_raw, dict):
            for step_idx, skill in step_skills_raw.items():
                try:
                    step_skills[int(step_idx)] = str(skill)
                except (TypeError, ValueError):
                    continue
        if self.skill_mode == "episode_only":
            step_skills = {}
        out = {
            "episode_summary": str(parsed.get("episode_summary", "")) if self.include_episode_summary else "",
            "episode_skill": "" if self.skill_mode == "step_only" else str(parsed.get("episode_skill", "")),
            "step_skills": step_skills,
        }
        failure_step = parsed.get("failure_step")
        out["failure_step"] = (int(failure_step) if isinstance(failure_step, (int, float))
                               and not isinstance(failure_step, bool) else None)
        out["failure_type"] = str(parsed.get("failure_type", "") or "") or "none"
        out["step_skill"] = str(parsed.get("step_skill", "") or "")
        if out["failure_step"] is not None and out["step_skill"].strip():
            out["step_skills"] = {out["failure_step"]: out["step_skill"]}  # the Critical-Step block of SEED's episode_step mode
        return out

    def _extract_json_object(self, response: str) -> Dict[str, object]:
        response_text = str(response or "")
        candidate_texts = self._candidate_json_texts(response_text)
        decoded_objects: List[Dict[str, object]] = []
        saw_object_start = False
        last_error: Optional[BaseException] = None

        for text in candidate_texts:
            if "{" in text:
                saw_object_start = True
            for variant in self._json_text_variants(text):
                objects, error = self._decode_json_objects(variant)
                decoded_objects.extend(objects)
                if error is not None:
                    last_error = error

        preferred_field = "step_skills" if self.skill_mode == "step_only" else "episode_skill"
        for obj in decoded_objects:
            if preferred_field in obj:
                return obj
        if decoded_objects:
            return decoded_objects[0]
        if not saw_object_start:
            raise ValueError("No JSON object found in SEED analyzer response.")
        raise ValueError(f"Could not parse JSON object in SEED analyzer response: {last_error}") from last_error

    def _candidate_json_texts(self, response_text: str) -> List[str]:
        fenced_blocks = re.findall(
            r"```(?:json|JSON)?\s*(.*?)```",
            response_text,
            flags=re.DOTALL,
        )
        candidates = [block.strip() for block in fenced_blocks if block.strip()]
        candidates.append(response_text.strip())
        return candidates

    def _json_text_variants(self, text: str) -> List[str]:
        repaired_trailing_commas = re.sub(r",\s*([}\]])", r"\1", text)
        if repaired_trailing_commas == text:
            return [text]
        return [text, repaired_trailing_commas]

    def _decode_json_objects(self, text: str) -> Tuple[List[Dict[str, object]], Optional[BaseException]]:
        objects: List[Dict[str, object]] = []
        last_error: Optional[BaseException] = None
        decoder = json.JSONDecoder(strict=False)

        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text, idx)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)

        if objects:
            return objects, last_error

        for json_slice in self._balanced_object_slices(text):
            try:
                parsed = ast.literal_eval(json_slice)
            except (SyntaxError, ValueError) as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
        return objects, last_error

    def _balanced_object_slices(self, text: str) -> List[str]:
        slices = []
        start: Optional[int] = None
        depth = 0
        in_string = False
        quote_char = ""
        escape = False

        for idx, char in enumerate(text):
            if escape:
                escape = False
                continue
            if in_string:
                if char == "\\":
                    escape = True
                elif char == quote_char:
                    in_string = False
                continue
            if char in {"'", '"'}:
                in_string = True
                quote_char = char
                continue
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    slices.append(text[start:idx + 1])
                    start = None
        return slices

    def _format_episode_steps(self, steps: List[Dict[str, object]], hindsight_extra: Optional[Dict[int, str]] = None) -> str:
        lines = []
        for step in steps:
            response = str(step["response"])
            if self.render_response == "action_only":
                match = re.search(r"<action>(.*?)</action>", response, re.S)
                response = f"<action>{match.group(1).strip()}</action>" if match else response[-200:]
            block = (
                f"Step {step['step_index']}\n"
                f"Observation: {step['observation']}\n"
                f"Response: {response}\n"
            )
            if "action_valid" in step:
                block += f"Action valid: {str(bool(step['action_valid'])).lower()}\n"
            if "hindsight" in step:
                extra = (hindsight_extra or {}).get(int(step["step_index"]), "") if str(step.get("step_index", "")).isdigit() else ""
                block += f"Hindsight: {step['hindsight']}{extra}\n"
            lines.append(block)
        if steps and steps[-1].get("final_observation"):
            lines.append(f"Final observation: {steps[-1]['final_observation']}\n")
        return "\n".join(lines)
