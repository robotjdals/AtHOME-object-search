"""Trajectory-level GRPO rollouts in the symbolic environment (proposal 6-4).

- Reward R(tau) = -D(tau) - lambda_s N(tau): D is the travelled A* path cost
  [m], N the number of Search Location visits (search steps).
- G rollouts start from the same state (a findable start of
  scripts/export_start_states.py); the advantage of rollout i is
  A_i = (R_i - mean R) / (std R + delta).
- Only Workspace-selection outputs are trained (Workspace Selection Adapter);
  room and standalone decisions come from the fixed Room and Search Location
  adapters and are recorded but not trained.
- The reward assumes every rollout finds the target. A group with a failed
  rollout is left out (the start was findable but the session picked goals
  that do not see the target), and so is a group whose rewards are all equal
  (zero advantage, DAPO dynamic sampling).

Policies are ``select(stage, target, candidates, context) -> candidate_id``
objects; a Student sampler additionally exposes the prompt and raw output
through ``last_messages`` / ``last_output`` so they can be trained on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence

from athome.search import SearchSession
from athome.search.policy import ShuffledPolicy, Stage

TRAINED_STAGES = frozenset({Stage.WORKSPACE})


@dataclass
class Decision:
    stage: str
    candidate_ids: List[str]
    selected_id: str
    trainable: bool
    presented_order: Optional[List[str]] = None   # candidate order shown to the policy
    messages: Optional[list] = None       # Student prompt, when the policy records it
    output: Optional[str] = None          # Student raw output (the trained tokens)
    aliases: Optional[Dict[str, str]] = None   # alias -> candidate ID of the prompt
    logprob: Optional[float] = None       # log pi_old(chosen), normalized over the candidates


@dataclass
class Rollout:
    decisions: List[Decision] = field(default_factory=list)
    success: bool = False
    distance_m: float = 0.0
    visits: int = 0
    reward: float = 0.0
    advantage: Optional[float] = None


class _Recorder:
    """Routes each stage to its policy and records the decisions."""

    def __init__(self, policies: Dict[Stage, object], out: Rollout):
        self._policies, self._out = policies, out

    def select(self, stage, target, candidates, context):
        policy = self._policies[stage]
        chosen = policy.select(stage, target, candidates, context)
        self._out.decisions.append(Decision(
            stage=stage.value, candidate_ids=[c.candidate_id for c in candidates],
            selected_id=chosen, trainable=stage in TRAINED_STAGES,
            presented_order=getattr(policy, "last_order", None),
            messages=getattr(policy, "last_messages", None),
            output=getattr(policy, "last_output", None),
            aliases=getattr(policy, "last_aliases", None),
            logprob=getattr(policy, "last_logprob", None)))
        return chosen


def reward(distance_m: float, visits: int, step_cost: float) -> float:
    """R = -D - lambda_s N (proposal 6-4)."""
    return -distance_m - step_cost * visits


def rollout(graph, navigation, environment, target: str, policies: Dict[Stage, object],
            step_cost: float, max_steps: int = 10_000, coverage=None,
            shuffle_seed=None) -> Rollout:
    """One trajectory until the target is found or no location is left.

    ``coverage`` (a fresh athome.search.coverage.RoomCoverage per rollout) gives
    the room stage the observed fraction of each room, as in Teacher episodes
    and on the robot; without it the room input falls back to visit counts.
    ``shuffle_seed`` shows each query's candidates in a seeded random order
    (athome.search.policy.ShuffledPolicy), as in the SFT data."""
    out = Rollout()
    if shuffle_seed is not None:
        policies = {s: ShuffledPolicy(p, f"{shuffle_seed}:{s.value}") for s, p in policies.items()}
    session = SearchSession(graph, navigation, [target], policy=_Recorder(policies, out),
                            max_steps=max_steps, coverage=coverage)
    while True:
        decision = session.next_decision(environment.pose)
        if decision is None:
            break
        session.report(decision, environment.visit(decision))
    out.success = session.targets[0].status.value == "found"
    out.distance_m, out.visits = environment.distance_m, environment.visits
    out.reward = reward(out.distance_m, out.visits, step_cost)
    return out


def group_advantages(rollouts: Sequence[Rollout], delta: float = 1e-6) -> Optional[str]:
    """Set group-relative advantages; returns why the group is unusable, or None."""
    if any(not r.success for r in rollouts):
        return "failed_rollout"
    rewards = [r.reward for r in rollouts]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
    if std == 0:
        return "equal_rewards"
    if not any(d.trainable for r in rollouts for d in r.decisions):
        return "no_trainable_decision"
    for r in rollouts:
        r.advantage = (r.reward - mean) / (std + delta)
    return None


def clipped_objective(logpi_new, logpi_old: float, advantage: float, clip_eps: float,
                      kl=None, kl_beta: float = 0.0):
    """Per-decision GRPO term (proposal 6-4; Shao et al. 2024):
    min(rho A, clip(rho, 1-eps, 1+eps) A) - beta KL, rho = pi_theta/pi_old.
    ``logpi_new`` is a tensor with gradient; the caller averages it over the
    trained decisions of a rollout (1/M_i) and over the group (1/G) and
    maximizes it."""
    import torch
    ratio = torch.exp(logpi_new - logpi_old)
    surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage)
    return surrogate - kl_beta * kl if kl is not None else surrogate

