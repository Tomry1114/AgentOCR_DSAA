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

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Tuple

import gym
import numpy as np
from omegaconf import DictConfig, ListConfig
from copy import deepcopy 

from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.search import (
    build_search_result_text,
    call_search_api_batch,
)
from agent_system.environments.env_package.search.idea_adapter import adapt_search_action


def _format_adapt_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "idea_adapter_kind": str(meta.get("kind", "")),
        "idea_adapter_adapted": bool(meta.get("adapted", False)),
        "idea_adapter_deserves_search_bonus": bool(meta.get("deserves_search_bonus", False)),
        "idea_adapter_reasons": ",".join(str(reason) for reason in meta.get("reasons", [])),
    }


class SearchMultiProcessEnv(gym.Env):
    """
    - env_num  : Number of groups (logical sharding; keep the parameter for external compatibility)
    - group_n  : Number of environments per group
    - total_envs = env_num * group_n
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: DictConfig | None = None,
    ) -> None:
        super().__init__()

        from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv

        self.env_num   = env_num
        self.group_n   = group_n
        self.batch_size = env_num * group_n
        self.is_train  = is_train
        self.max_steps = env_config.max_steps

        self._rng = np.random.RandomState(seed)

        # ---------- Key changes start ----------
        # 1) Normalize search_url into a list
        search_cfg  = env_config.search
        search_urls = search_cfg.search_url
        if not isinstance(search_urls, ListConfig):
            search_urls = [search_urls]

        n_clients = len(search_urls)
        per_url_limit = max(1, int(getattr(search_cfg, "max_concurrent_requests", 8)))
        self.batch_request_size = max(1, int(getattr(search_cfg, "batch_request_size", 32)))
        configured_max_workers = getattr(search_cfg, "max_workers", None)
        if configured_max_workers is None:
            max_workers = min(self.batch_size, per_url_limit * n_clients)
        else:
            max_workers = min(self.batch_size, max(1, int(configured_max_workers)))

        # 2) Assign a URL to each env in a round-robin manner
        self.envs = []
        for idx in range(self.batch_size):
            cfg_i = deepcopy(search_cfg)
            cfg_i.search_url = search_urls[idx % n_clients]
            self.envs.append(SearchEnv(cfg_i))

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env, kwargs):
        extras = {
            "ground_truth": kwargs["ground_truth"],
            "max_turns": self.max_steps,
            "data_source": kwargs.get("data_source", "unknown"),
            "question": kwargs["question"],
        }
        env.reset(extras)
        obs = kwargs["question"]
        info = {'data_source': kwargs.get("data_source", "unknown")}
        return obs, info
    
    def _sync_step(self, env, action: str):
        out = env.step(action)

        obs = out["observations"]
        obs = "" if len(obs) == 0 else obs[0]["content"].strip()
        reward = out["reward"]
        done = out["done"]

        info = dict(out.get("metadata", {}))
        info["postprocessed_action"] = out.get("postprocessed_action")
        info["won"] = bool(done and reward >= 1.0)
        return obs, reward, done, info

    def reset(self, kwargs: List[Dict]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {
                    "ground_truth": "",
                    "question": "",
                    "data_source": "unkown",
                }

        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, info_list

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        results: List[Tuple[str, float, bool, Dict[str, Any]]] = [("", 0.0, False, {}) for _ in range(self.batch_size)]
        pending_search_groups: Dict[str, Dict[str, Any]] = {}

        for idx, (env, act) in enumerate(zip(self.envs, padded_actions)):
            env.turns += 1
            postprocessed_action, adapt_meta = adapt_search_action(
                question=getattr(env, "question", ""),
                chat_history=env.chat_history,
                action=act,
            )
            env.chat_history.append({"role": "assistant", "content": postprocessed_action})

            if not env.done:
                done = env._is_done(postprocessed_action)
                env.done = done
            else:
                done = True

            reward = env._get_reward(postprocessed_action, done)
            if done:
                results[idx] = (
                    "",
                    reward,
                    done,
                    {
                        "data_source": getattr(env, "data_source", "unknown"),
                        "tool_calling": False,
                        "won": bool(done and reward >= 1.0),
                        "postprocessed_action": postprocessed_action,
                        **_format_adapt_meta(adapt_meta),
                    },
                )
                continue

            query = env._parse_action(postprocessed_action)
            if query and query[0]:
                if adapt_meta.get("deserves_search_bonus", True):
                    reward += env.search_reward_coef
                search_url = env.tool_group.search_url
                if search_url not in pending_search_groups:
                    pending_search_groups[search_url] = {
                        "items": [],
                        "tool_group": env.tool_group,
                    }
                pending_search_groups[search_url]["items"].append(
                    (idx, env, query[0], reward, postprocessed_action, _format_adapt_meta(adapt_meta))
                )
                continue

            results[idx] = (
                "",
                reward,
                False,
                {
                    "tool_calling": True,
                    "tool_group": "SearchToolGroup",
                    "tool_name": "search",
                    "tool_input": query,
                    "data_source": getattr(env, "data_source", "unknown"),
                    "won": False,
                    "postprocessed_action": postprocessed_action,
                    **_format_adapt_meta(adapt_meta),
                },
            )

        for search_url, group in pending_search_groups.items():
            items = group["items"]
            tool_group = group["tool_group"]
            for start_idx in range(0, len(items), self.batch_request_size):
                chunk = items[start_idx : start_idx + self.batch_request_size]
                query_list = [item[2] for item in chunk]
                try:
                    with tool_group.concurrent_semaphore:
                        api_response, error_msg = call_search_api_batch(
                            retrieval_service_url=search_url,
                            query_list=query_list,
                            topk=tool_group.topk,
                            timeout=tool_group.timeout,
                            log_requests=tool_group.log_requests,
                            session=tool_group.session,
                        )
                except Exception as exc:
                    api_response = None
                    error_msg = str(exc)

                if error_msg or api_response is None:
                    error_text = error_msg or "Search request failed or timed out after retries."
                    for idx, env, query_text, reward, action_text, adapt_meta_info in chunk:
                        new_obs = {"role": "user", "content": error_text}
                        env.chat_history.append(new_obs)
                        results[idx] = (
                            error_text,
                            reward,
                            False,
                            {
                                "tool_calling": True,
                                "tool_group": "SearchToolGroup",
                                "tool_name": "search",
                                "tool_input": [query_text],
                                "data_source": getattr(env, "data_source", "unknown"),
                                "won": False,
                                "postprocessed_action": action_text,
                                **adapt_meta_info,
                            },
                        )
                    continue

                raw_results = api_response.get("result", [])
                if len(raw_results) != len(chunk):
                    raise ValueError(
                        f"Batch search response size mismatch for {search_url}: expected {len(chunk)} results, got {len(raw_results)}"
                    )

                for (idx, env, query_text, reward, action_text, adapt_meta_info), retrieval in zip(chunk, raw_results):
                    result_text, _, formatted_result = build_search_result_text(retrieval)
                    readable_result = formatted_result or result_text
                    observation = "\n<information>\n" + readable_result + "\n</information>\n"
                    new_obs = {"role": "user", "content": observation}
                    env.chat_history.append(new_obs)
                    results[idx] = (
                        observation,
                        reward,
                        False,
                        {
                            "tool_calling": True,
                            "tool_group": "SearchToolGroup",
                            "tool_name": "search",
                            "tool_input": [query_text],
                            "data_source": getattr(env, "data_source", "unknown"),
                            "won": False,
                            "postprocessed_action": action_text,
                            **adapt_meta_info,
                        },
                    )

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


def build_search_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None,
):
    return SearchMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )
