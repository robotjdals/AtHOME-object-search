"""Record of every planner query, for logs and rosbag replay.

Wraps a Policy without changing its behaviour: each ``select`` call yields
one JSON-serializable record with the planner input (stage, target,
candidates, context), the policy output, the selection the session applies
(minimum cost when the output is unusable, as SearchSession does), the raw
LLM text when the policy exposes it and the call latency. The prompt itself
is reproducible from the record with athome.inference.prompts.build_messages.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Sequence

from athome.inference.prompts import PROMPT_VERSION
from athome.search.policy import Candidate, MinCostPolicy, PlanningContext, PolicyError, Stage


class DecisionLog:
    def __init__(self, policy, sink: Callable[[dict], None],
                 clock: Callable[[], float] = time.monotonic):
        self.policy = policy
        self._sink = sink
        self._clock = clock

    def select(self, stage: Stage, target: str, candidates: Sequence[Candidate],
               context: PlanningContext) -> str:
        if hasattr(self.policy, "last_raw"):
            self.policy.last_raw = None      # not stale when no request is made
        start = self._clock()
        output, raised = None, None
        try:
            output = self.policy.select(stage, target, candidates, context)
        except PolicyError as e:
            raised = e
        latency = self._clock() - start
        error = str(raised) if raised else ""
        ids = [c.candidate_id for c in candidates]
        if output not in ids and not error:
            error = f"후보 밖 출력: {output!r}"
        applied = (output if output in ids
                   else MinCostPolicy().select(stage, target, candidates, context))
        self._sink({
            "prompt_version": PROMPT_VERSION,
            "stage": stage.value,
            "target": target,
            "candidates": [{"id": c.candidate_id, "cost": c.cost, "info": c.info}
                           for c in candidates],
            "context": dataclasses.asdict(context),
            "output": output,
            "applied": applied,
            "fallback": applied != output,
            "error": error,
            "raw": getattr(self.policy, "last_raw", None),
            "latency_s": round(latency, 4),
        })
        if raised is not None:
            raise raised
        return output          # the session rejects outputs outside the candidates
