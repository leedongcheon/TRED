import re
from typing import Any, Sequence, Tuple


SKILL_MODES = ("episode_step", "step_only", "episode_only")
SKILL_TEACHER_MODES = ("step_priority", "additive")


def validate_skill_mode(mode: str) -> str:
    mode = str(mode or "episode_step")
    if mode not in SKILL_MODES:
        raise ValueError(f"Unsupported SEED skill_mode: {mode}")
    return mode


def select_skill_teacher_sources(
    *,
    step_skill: str,
    episode_skill_enabled: bool,
    step_skill_enabled: bool,
    mode: str = "step_priority",
    skill_mode: str = "episode_step",
) -> Tuple[bool, bool]:
    """Return whether to score the episode-skill and step-skill prompts."""
    skill_mode = validate_skill_mode(skill_mode)
    if mode not in SKILL_TEACHER_MODES:
        raise ValueError(f"Unsupported SEED skill_teacher_mode: {mode}")

    use_step_skill = bool(str(step_skill).strip()) and step_skill_enabled
    if skill_mode == "step_only":
        return False, use_step_skill
    if skill_mode == "episode_only":
        return episode_skill_enabled, False

    use_episode_skill = episode_skill_enabled and (mode == "additive" or not use_step_skill)
    return use_episode_skill, use_step_skill


def build_augmented_observation_text(
    *,
    observation: str,
    episode_skill: str = "",
    step_skill: str = "",
    strict: bool = True,
    episode_position: str = "task",
) -> str:
    """Insert episode-level and optional critical-step skill-augmented context into a prompt.

    With ``strict`` (default) a skill whose anchor is missing raises instead of going to another place, so the
    skill position never varies between trajectories. The observation is not stripped and only newlines are
    trimmed around the insertion point: removing the inserted block gives the observation back exactly.

    ``episode_position`` places the Episode-Level Skill block: "task" (SEED's place, right after the task line),
    "end" (right before "Now it's your turn", ahead of the Critical-Step Skill block: History -> Reminder ->
    Episode-Level Skill -> Critical-Step Skill -> Now it's your turn) or "mid" (after the operation document, before the
    history; before the reminder when there is no history yet).
    """
    if episode_position not in ("task", "end", "mid"):
        raise ValueError(f"Unsupported episode_position: {episode_position}")

    def _format_skill_section(title: str, instruction: str, body: Any) -> str:
        body_text = str(body).strip()
        if not body_text:
            return ""
        return f"**{title}**\n{instruction}:\n[{body_text}]"

    def _insert_before_anchor(prompt_text: str, sections: Sequence[str], anchors: Sequence[str], what: str) -> str:
        merged_sections = [str(section).strip() for section in sections if str(section or "").strip()]
        if not merged_sections:
            return prompt_text

        insertion_block = "\n\n".join(merged_sections)
        for anchor in anchors:
            anchor_idx = prompt_text.find(anchor)
            if anchor_idx == -1:
                continue

            prefix = prompt_text[:anchor_idx].rstrip("\n")
            suffix = prompt_text[anchor_idx:].lstrip("\n")
            if prefix and suffix:
                return f"{prefix}\n\n{insertion_block}\n\n{suffix}"
            if prefix:
                return f"{prefix}\n\n{insertion_block}"
            return f"{insertion_block}\n\n{suffix}"

        if strict:
            raise ValueError(f"{what} anchor ({anchors[0]!r}) not found in prompt")
        return f"{prompt_text}\n\n{insertion_block}" if prompt_text else insertion_block

    def _insert_after_task_description(prompt_text: str, sections: Sequence[str]) -> str:
        merged_sections = [str(section).strip() for section in sections if str(section or "").strip()]
        if not merged_sections:
            return prompt_text

        insertion_block = "\n\n".join(merged_sections)
        task_line_patterns = (
            r"^.*\bYour task is to:\s*.*(?:\n|$)",
            r"^.*\bYour current task is:\s*.*(?:\n|$)",
            r"^.*\bYour task is:\s*.*(?:\n|$)",
            r"^.*\bYour question:\s*.*(?:\n|$)",
            r"^.*\bYour goal is(?: to)?\b[:\s].*(?:\n|$)",
        )
        for pattern in task_line_patterns:
            match = re.search(pattern, prompt_text, flags=re.MULTILINE)
            if match is None:
                continue

            prefix = prompt_text[: match.end()].rstrip("\n")
            suffix = prompt_text[match.end() :].lstrip("\n")
            if prefix and suffix:
                return f"{prefix}\n\n{insertion_block}\n\n{suffix}"
            if prefix:
                return f"{prefix}\n\n{insertion_block}"
            return f"{insertion_block}\n\n{suffix}"

        if strict:
            raise ValueError("episode skill anchor ('Your question:') not found in prompt")
        return _insert_before_anchor(
            prompt_text,
            merged_sections,
            (
                "Now it's your turn to",
                "Now it's your turn",
            ),
            "episode skill",
        )

    episode_section = _format_skill_section(
        "Episode-Level Skill",
        (
            "Refer to this episode-level skill when deciding what action "
            "to take in the current episode"
        ),
        episode_skill,
    )
    step_section = _format_skill_section(
        "Critical-Step Skill",
        "Use this current-step skill for this decision only",
        step_skill,
    )
    if episode_position == "task":
        prompt_text, final_sections = _insert_after_task_description(str(observation), [episode_section]), [step_section]
    elif episode_position == "mid":
        prompt_text = _insert_before_anchor(str(observation), [episode_section],
                                            ("Prior to this step", "Reminder — your question:"), "episode skill")
        final_sections = [step_section]
    else:
        prompt_text, final_sections = str(observation), [episode_section, step_section]
    return _insert_before_anchor(
        prompt_text,
        final_sections,
        (
            "Now it's your turn to",
            "Now it's your turn",
        ),
        "step skill",
    )
