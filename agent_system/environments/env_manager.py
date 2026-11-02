# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Tuple

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts import TKGQA_OPERATIONS_SPEC, TKGQA_TEMPLATE, TKGQA_TEMPLATE_NO_HIS
from agent_system.memory import TKGQAMemory


class TKGQAEnvironmentManager(EnvironmentManagerBase):
    """EnvironmentManager for TKGQA: the Search setting with the symbolic executor of the temporal operations as the tool."""
    def __init__(self, envs, projection_f, config):
        self.memory = TKGQAMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        self.kwargs = kwargs
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "action": [info["operation_action"] for info in infos],
            "observation": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }

        # invalid = malformed response (projection) or an operation error reported by the symbolic executor
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(bool(valids[i]) and bool(info["executor_valid"]))

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="observation",
                action_key="action"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = TKGQA_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    operations_spec=TKGQA_OPERATIONS_SPEC,
                )
            else:
                obs_i = TKGQA_TEMPLATE.format(
                    task_description=self.tasks[i],
                    operations_spec=TKGQA_OPERATIONS_SPEC,
                    step_count=len(self.memory[i]),
                    memory_context=memory_ctx[i],
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask


def make_envs(config):
    """Train and validation environment managers (one KG and grounder shared by both)."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "tkgqa" not in config.env.env_name.lower():
        raise ValueError(f"unsupported environment: {config.env.env_name!r} (this release has TKGQA only)")
    from agent_system.environments.env_package.tkgqa import build_tkgqa_envs, tkgqa_projection
    shared = {}  # one KG and grounder (bge-m3) for the train and validation environments
    _envs = build_tkgqa_envs(env_num=config.data.train_batch_size, group_n=group_n, env_config=config.env, shared=shared)
    _val_envs = build_tkgqa_envs(env_num=config.data.val_batch_size, group_n=1, env_config=config.env, shared=shared)
    envs = TKGQAEnvironmentManager(_envs, tkgqa_projection, config)
    val_envs = TKGQAEnvironmentManager(_val_envs, tkgqa_projection, config)
    return envs, val_envs
