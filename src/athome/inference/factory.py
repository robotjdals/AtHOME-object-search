"""Build the planner policy and target matcher from the robot config."""

from __future__ import annotations

import os

from athome.scene_graph.query import normalize_category
from athome.search.matching import ClipMatcher, LabelMatcher, TargetMatcher
from athome.search.policy import MinCostPolicy, Policy, Stage


def make_policy(section: dict) -> Policy:
    kind = section.get("type", "min_cost")
    if kind == "min_cost":
        return MinCostPolicy()
    if kind == "llm":
        from athome.inference.llm_policy import LLMPolicy

        llm = section["llm"]
        models = {Stage(stage): name for stage, name in llm["models"].items()}
        return LLMPolicy(
            llm["base_url"], models,
            timeout=float(llm.get("timeout", 5.0)),
            retries=int(llm.get("retries", 1)),
            # Name of the environment variable holding the key (vllm --api-key);
            # the key itself never goes into the config.
            api_key=_env_key(llm.get("api_key_env")),
        )
    raise ValueError(f"알 수 없는 planner type: {kind}")


def _env_key(name):
    if not name:
        return None
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(f"{name} 환경 변수 필요")
    return key


def make_command_parser(section: dict, categories=()):
    """instruction -> targets, or None when not configured. ``categories``:
    target category names shown as the naming convention (Vocabulary.categories)."""
    if not section:
        return None
    from functools import partial

    from athome.inference.chat import ChatJSON
    from athome.inference.command_parser import parse_command

    complete = ChatJSON(
        section["base_url"], section["model"],
        api_key_env=section.get("api_key_env"),
        timeout=float(section.get("timeout", 10.0)),
        schema_name="search_command",
        extra_body=section.get("extra_body"),
        # A target list is short; the cap stops a runaway array.
        max_tokens=int(section.get("max_tokens", 128)),
    )
    return partial(parse_command, complete=complete, categories=list(categories))


def make_matcher(section: dict, category=normalize_category, target_keys=None) -> TargetMatcher:
    """``category``: label -> target category (Vocabulary.canonical);
    ``target_keys``: target -> categories that count as it (Vocabulary.target_keys)."""
    kind = section.get("type", "label")
    label = LabelMatcher(category=category, target_keys=target_keys)
    if kind == "label":
        return label
    if kind == "clip":
        from athome.inference.text_encoder import HttpTextEncoder

        clip = section["clip"]
        return ClipMatcher(
            HttpTextEncoder(clip["base_url"], clip["model"],
                            timeout=float(clip.get("timeout", 3.0))),
            threshold=float(clip["threshold"]),
            fallback=label,
        )
    raise ValueError(f"알 수 없는 matcher type: {kind}")
