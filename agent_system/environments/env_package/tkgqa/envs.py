"""TKGQA environments: one symbolic-executor session (tkgqa.executor.Session) per environment.

Like the Search environment, the question comes from the dataset row (env_kwargs) and each step
returns the tool result as the observation: here the executor's observation string. The prompt is
built by TKGQAEnvironmentManager from the template and the history, as for the other environments.
The reward is the F1 of the answer set at the end of the episode (0 before); ``won`` is F1 = 1.
Analyzer-only annotations ride on the step infos (the actor prompt never shows them): "hindsight", the
result of that step's action (tkgqa.hindsight.step_hindsight); and when the episode ends "auto_hints"
(tkgqa.hindsight.auto_hints, plus the lines of discarded gold-bearing facts and extra answers), "final_observation" and
"episode_success". The record-level hindsight of a finished episode (compute_hindsight) is kept in ``hindsights``.
"""

from typing import Any, Dict, List, Optional

from tkgqa.analysis import Analyzer
from tkgqa.answers import infer_gran, score_episode
from tkgqa.operations import OperationError, parse
from tkgqa.executor import ACTION_RE, ExecutorConfig, Session
from tkgqa.grounder import Grounder
from tkgqa.hindsight import (DROPPED_GOLD_MARK, auto_hints, compute_hindsight, dropped_gold_suffix, extra_answers_line,
                             step_hindsight)
from tkgqa.kg import TemporalKG


class TKGQAEnvs:
    """
    - env_num  : number of groups
    - group_n  : number of environments per group
    - total_envs = env_num * group_n
    """

    def __init__(self, env_num: int = 1, group_n: int = 1, env_config=None,
                 kg: Optional[TemporalKG] = None, grounder: Optional[Grounder] = None) -> None:
        tk = env_config.tkgqa
        self.batch_size = env_num * group_n
        self.kg = kg or TemporalKG.load(tk.dataset, tk.data_root)
        self.grounder = grounder or Grounder.from_store_root(self.kg, tk.store_root, device=tk.get("device", "cpu"))
        self.analyzer = Analyzer(self.kg)
        self.exec_config = ExecutorConfig(max_steps=int(env_config.max_steps))
        self.sessions: List[Session] = []
        self.kwargs: List[Dict[str, Any]] = []
        self.scores: List[Optional[Dict[str, float]]] = []
        self.hindsights: List[Optional[Dict[str, Any]]] = []

    def reset(self, kwargs: List[Dict[str, Any]]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}")
        self.kwargs = [dict(kw) for kw in kwargs]
        self.sessions = [Session(self.kg, self.grounder, kw["question"], config=self.exec_config) for kw in self.kwargs]
        self.scores = [None] * len(self.sessions)
        self.hindsights = [None] * len(self.sessions)
        self.step_lines = [[] for _ in self.sessions]
        self.hints = [None] * len(self.sessions)
        obs = [kw["question"] for kw in self.kwargs]
        infos = [{"data_source": kw.get("data_source", "tkgqa"), "won": False} for kw in self.kwargs]
        return obs, infos

    def _prefetch(self, actions: List[str]) -> None:
        """Ground every FIND name of this step in one encoder batch per kind; the sessions then hit the grounder cache."""
        requests = []
        for session, action in zip(self.sessions, actions):
            match = None if session.done else ACTION_RE.search(action or "")
            if match is None:
                continue
            try:
                stmt = parse(match.group(1))
            except OperationError:
                continue
            if stmt.op == "FIND" and len(stmt.args) == 3:
                for slot, kind in ((0, "entity"), (1, "relation"), (2, "entity")):
                    text = stmt.args[slot].text.strip()
                    if text and text != "?":
                        requests.append((text, kind))
        if requests:
            self.grounder.prefetch(requests)

    def step(self, actions: List[str]):
        """``actions``: the projected responses (tkgqa_projection); the executor reads the first <action> block."""
        if len(actions) > len(self.sessions):
            raise ValueError(f"Got {len(actions)} actions for {len(self.sessions)} environments")
        self._prefetch(actions)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, (session, action) in enumerate(zip(self.sessions, actions)):
            if session.done:  # finished earlier; the rollout loop masks this slot out
                obs_list.append(session.current_observation)
                reward_list.append(0.0)
                done_list.append(True)
                info_list.append(self._info(i, executor_valid=True))
                continue
            session.step(action)
            last = session.log[-1]
            gold, kind, gran = self._gold(i)
            self.step_lines[i].append(step_hindsight(session, len(session.log) - 1, self.analyzer, gold, kind, gran))
            reward = 0.0
            if session.done:
                self._finish(i)
                reward = float(self.scores[i]["reward"])
            obs_list.append(last.obs)
            reward_list.append(reward)
            done_list.append(session.done)
            info_list.append(self._info(i, executor_valid=last.status != "invalid"))
        return obs_list, reward_list, done_list, info_list

    def _gold(self, i: int):
        gold, kind = list(self.kwargs[i]["answers"]), self.kwargs[i]["kind"]
        return gold, kind, (infer_gran(gold) if kind in ("time", "span") else None)

    def _finish(self, i: int) -> None:
        session = self.sessions[i]
        gold, kind, gran = self._gold(i)
        self.scores[i] = score_episode(session.answer, gold, kind)
        self.hindsights[i] = compute_hindsight(session, self.analyzer, gold, kind, gran)
        self.hints[i] = auto_hints(session, self.analyzer, gold, kind, gran, self.hindsights[i])
        k = self.hindsights[i].get("drop_step")
        if k is not None:  # moved to step k's Hindsight line by seed/analysis.py
            suffix = dropped_gold_suffix(session, self.analyzer, gold, kind, gran, k)
            if suffix:
                self.hints[i] += f"\n{DROPPED_GOLD_MARK}{k}:{suffix}"
        line = extra_answers_line(session.answer, gold, kind, session.question)
        if line:
            self.hints[i] += f"\n{line}"

    def _info(self, i: int, executor_valid: bool) -> Dict[str, Any]:
        session, score = self.sessions[i], self.scores[i] or {}
        last = session.log[-1] if session.log else None
        return {
            "data_source": self.kwargs[i].get("data_source", "tkgqa"),
            "won": bool(score.get("success", 0.0)),
            "f1": float(score.get("f1", 0.0)),
            "hit1": float(score.get("hit1", 0.0)),
            "failure_type": self.hindsights[i]["failure_type"] if self.hindsights[i] else None,
            "executor_valid": bool(executor_valid),
            "operation_action": (last.action or "") if last is not None else "",
            # analyzer-only (tkgqa.hindsight); the episode fields are empty until the episode ends
            "hindsight": self.step_lines[i][-1] if self.step_lines[i] else "",
            "auto_hints": self.hints[i] or "",
            "final_observation": (last.obs if last is not None else "") if session.done else "",
            "episode_success": float(score.get("success", 0.0)),
        }

    def close(self) -> None:
        pass


def build_tkgqa_envs(env_num: int = 1, group_n: int = 1, env_config=None, shared: Optional[Dict[str, Any]] = None):
    """``shared`` carries one KG and grounder (bge-m3) across the train and validation environments."""
    shared = shared if shared is not None else {}
    envs = TKGQAEnvs(env_num=env_num, group_n=group_n, env_config=env_config,
                     kg=shared.get("kg"), grounder=shared.get("grounder"))
    shared.setdefault("kg", envs.kg)
    shared.setdefault("grounder", envs.grounder)
    return envs
