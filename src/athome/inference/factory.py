"""Build the planner policy and target matcher from the robot config."""

from __future__ import annotations

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
        )
    raise ValueError(f"알 수 없는 planner type: {kind}")


def make_command_parser(section: dict):
    """instruction -> targets, or None when not configured."""
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
    )
    return partial(parse_command, complete=complete)


def make_matcher(section: dict) -> TargetMatcher:
    kind = section.get("type", "label")
    if kind == "label":
        return LabelMatcher()
    if kind == "clip":
        from athome.inference.text_encoder import HttpTextEncoder

        clip = section["clip"]
        return ClipMatcher(
            HttpTextEncoder(clip["base_url"], clip["model"],
                            timeout=float(clip.get("timeout", 3.0))),
            threshold=float(clip["threshold"]),
        )
    raise ValueError(f"알 수 없는 matcher type: {kind}")
