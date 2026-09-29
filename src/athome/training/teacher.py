"""Teacher policy that records every planner query (proposal 6-3).

The Teacher sees exactly the planner input used by SFT and online planning
(``athome.inference.prompts``). It returns reasoning and a semantic likelihood
per candidate; travel cost is combined outside the LLM, as in object-search
planners that keep distance out of the LLM judgement (SG-Nav, LFG, ESC,
VoroNav). The label is the candidate with the largest search index

    posterior / (path_cost_m + step_cost_m)

where the posterior is the Teacher's likelihood after unsuccessful search of
the room (``after_search``: Bayesian search theory; rooms only)

i.e. the classical optimal-search ordering by detection probability over cost,
where ``step_cost_m`` is the per-step cost lambda_s of the GRPO reward
R = -D - lambda_s * N (proposal 6-4), so SFT labels and GRPO share one
objective. lambda_s is chosen on validation data (proposal); records keep the
selection for several lambda values for that comparison. The Student is
trained on the selected ID only.

``complete`` is ``(messages, json_schema) -> dict`` (e.g. ``ChatJSON``). The
offline stand-in ``MinCostTeacher`` produces the same records without an API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Sequence

from athome.inference.chat import QuotaExceeded
from athome.inference.prompts import (
    PROMPT_VERSION, TEACHER_PROMPT_VERSION, build_messages, build_teacher_messages,
    teacher_schema)
from athome.search.policy import Candidate, MinCostPolicy, PlanningContext, PolicyError, Stage

Complete = Callable[[List[dict], dict], dict]
DEFAULT_STEP_COSTS = (0.5, 1.0, 3.0, 10.0, math.inf)  # inf: likelihood only, cost breaks ties


@dataclass
class QueryRecord:
    stage: str
    target: str
    candidates: List[dict]
    context: dict
    student_messages: List[dict]
    teacher_messages: List[dict]
    aliases: Dict[str, str]
    selected_alias: Optional[str] = None
    selected_id: Optional[str] = None
    reason: str = ""
    likelihoods: Dict[str, float] = field(default_factory=dict)   # candidate_id -> p (LLM prior)
    posterior: Dict[str, float] = field(default_factory=dict)     # after unsuccessful search
    selection_by_step_cost: Dict[str, str] = field(default_factory=dict)
    single_candidate: bool = False
    error: str = ""
    usage: Optional[Dict[str, int]] = None                       # API tokens of this call
    meta: dict = field(default_factory=dict)
    # Prompt versions the messages were built with (records are rebuilt and
    # checked against them when exported).
    prompt_version: str = PROMPT_VERSION
    teacher_prompt_version: str = TEACHER_PROMPT_VERSION


def _candidate_dict(c: Candidate) -> dict:
    return {"candidate_id": c.candidate_id, "cost": c.cost, "info": c.info}


def _context_dict(ctx: PlanningContext) -> dict:
    return {
        "room_id": ctx.room_id, "room_label": ctx.room_label,
        "last_location": ctx.last_location, "last_location_label": ctx.last_location_label,
        "explored_rooms": list(ctx.explored_rooms),
        "explored_room_labels": list(ctx.explored_room_labels),
        "visited_count": ctx.visited_count,
        "current_room": ctx.current_room,
    }


def after_search(candidates: Sequence[Candidate], likelihood: Dict[str, float]) -> Dict[str, float]:
    """Room likelihood after unsuccessful search (Bayesian search theory,
    Koopman 1946 / Stone 1975): with the target equally likely anywhere on a
    room's floor and seen when in view, observing a fraction q of the room
    without seeing it scales its probability by (1 - q) (the common
    normaliser does not change the ranking). Non-room candidates are
    unchanged: searched locations are no longer candidates at all."""
    out = dict(likelihood)
    for c in candidates:
        if "observed_fraction" in c.info:        # observed floor fraction (coverage tracker)
            q = c.info["observed_fraction"]
        elif c.info.get("total_locations"):      # fallback: visited share of the room's locations
            q = c.info["searched_locations"] / c.info["total_locations"]
        else:
            continue
        out[c.candidate_id] = likelihood[c.candidate_id] * (1 - q)
    return out


def search_index_choice(candidates: Sequence[Candidate], likelihood: Dict[str, float],
                        step_cost: float) -> str:
    """argmax p / (cost + step_cost); ties by lower cost, then ID.
    step_cost = inf ranks by likelihood alone (cost only breaks ties)."""
    def key(c):
        p = likelihood[c.candidate_id]
        index = p if math.isinf(step_cost) else p / (c.cost + step_cost)
        return (-index, c.cost, c.candidate_id)
    return min(candidates, key=key).candidate_id


def parse_likelihoods(answer: dict, aliases: Dict[str, str]) -> Dict[str, float]:
    """alias-keyed answer -> candidate_id -> p. Every candidate exactly once, 0 <= p <= 1."""
    items = answer["likelihoods"]
    seen = [i["candidate"] for i in items]
    if sorted(seen) != sorted(aliases):
        raise ValueError(f"후보별 점수 누락·중복: {seen}")
    out = {}
    for item in items:
        p = float(item["likelihood"])
        if not (math.isfinite(p) and 0.0 <= p <= 1.0):
            raise ValueError(f"잘못된 가능성 값: {item}")
        out[aliases[item["candidate"]]] = p
    return out


def _cost_label(step_cost: float) -> str:
    return "inf" if math.isinf(step_cost) else f"{step_cost:g}"


class TeacherPolicy:
    """Policy that asks the Teacher LLM, selects by search index and records."""

    prompt_version = PROMPT_VERSION
    teacher_prompt_version = TEACHER_PROMPT_VERSION

    def __init__(self, complete: Complete, step_cost: float = 3.0,
                 compare_step_costs: Sequence[float] = DEFAULT_STEP_COSTS):
        if not step_cost > 0:
            raise ValueError("step_cost must be positive")
        self._complete = complete
        self.step_cost = step_cost
        self.compare_step_costs = tuple(sorted(set(compare_step_costs) | {step_cost}))
        self.records: List[QueryRecord] = []

    def _likelihoods(self, messages, aliases, stage, target, candidates, context):
        answer = self._complete(messages, teacher_schema(aliases))
        self._last_usage = getattr(self._complete, "last_usage", None)
        return parse_likelihoods(answer, aliases), str(answer.get("reasoning", ""))

    def select(self, stage: Stage, target: str, candidates: Sequence[Candidate],
               context: PlanningContext) -> str:
        student, aliases = build_messages(stage, target, candidates, context)
        teacher, _ = build_teacher_messages(stage, target, candidates, context)
        record = QueryRecord(
            stage=stage.value, target=target,
            candidates=[_candidate_dict(c) for c in candidates],
            context=_context_dict(context), student_messages=student,
            teacher_messages=teacher, aliases=aliases,
            single_candidate=len(candidates) == 1)
        self.records.append(record)
        if record.single_candidate:
            # Nothing to choose: the robot takes it without a planner call.
            record.selected_alias = next(iter(aliases))
            record.selected_id = candidates[0].candidate_id
            return record.selected_id
        self._last_usage = None
        try:
            likelihood, reason = self._likelihoods(teacher, aliases, stage, target,
                                                   candidates, context)
        except QuotaExceeded:
            record.error = "quota_exceeded"
            raise                      # no fallback: the run stops
        except Exception as e:  # noqa: BLE001 - recorded; SearchSession falls back
            record.error = f"{type(e).__name__}: {e}"
            raise PolicyError(record.error) from e
        record.likelihoods, record.reason = likelihood, reason
        record.usage = self._last_usage
        record.posterior = after_search(candidates, likelihood)
        record.selection_by_step_cost = {
            _cost_label(s): search_index_choice(candidates, record.posterior, s)
            for s in self.compare_step_costs}
        record.selected_id = record.selection_by_step_cost[_cost_label(self.step_cost)]
        record.selected_alias = next(a for a, cid in aliases.items() if cid == record.selected_id)
        return record.selected_id


class MinCostTeacher(TeacherPolicy):
    """Offline stand-in: equal likelihoods, so every rule picks the nearest."""

    def __init__(self, **kwargs):
        super().__init__(complete=None, **kwargs)

    def _likelihoods(self, messages, aliases, stage, target, candidates, context):
        return {c.candidate_id: 0.5 for c in candidates}, "uniform stand-in (no LLM)"
