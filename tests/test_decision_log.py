import json

from athome.execution.visit import VisitOutcome, VisitStatus
from athome.navigation import NavigationPlanner
from athome.scene_graph.query import SceneGraph
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.decision_log import DecisionLog
from athome.search.policy import PolicyError
from athome.testing import toy_env


class Fixed:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.last_raw = answer, error, "stale"

    def select(self, stage, target, candidates, context):
        if self.error:
            raise PolicyError(self.error)
        return self.answer or candidates[-1].candidate_id


def first_decisions(policy, n=3):
    records = []
    s = SearchSession(SceneGraph(toy_env.toy_graph(["remote"])),
                      NavigationPlanner(toy_env.toy_grid()), ["remote"],
                      policy=DecisionLog(policy, records.append))
    pose, order = Pose2D(2.5, 3.0, 0.0), []
    for _ in range(n):
        d = s.next_decision(pose)
        order.append(d.location_id)
        s.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED))
        pose = d.goals[0]
    return order, records, s


def test_records_every_query_without_changing_the_search():
    plain = SearchSession(SceneGraph(toy_env.toy_graph(["remote"])),
                          NavigationPlanner(toy_env.toy_grid()), ["remote"], policy=Fixed())
    pose, expected = Pose2D(2.5, 3.0, 0.0), []
    for _ in range(3):
        d = plain.next_decision(pose)
        expected.append(d.location_id)
        plain.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED))
        pose = d.goals[0]
    order, records, _ = first_decisions(Fixed())
    assert order == expected
    assert [r["stage"] for r in records[:2]] == ["room", "workspace"]
    assert all(not r["fallback"] and r["raw"] is None for r in records)  # stale raw cleared
    json.dumps(records)                                                  # publishable as-is


def test_unusable_outputs_are_logged_as_the_fallback_the_session_applies():
    order, records, s = first_decisions(Fixed(answer="C99"), n=1)
    assert records[0]["fallback"] and "C99" in records[0]["error"]
    assert s.history[0].policy_fallback
    _, records, s = first_decisions(Fixed(error="timeout"), n=1)
    assert records[0]["error"] == "timeout" and records[0]["output"] is None
    assert records[0]["applied"] == s.history[0].decision.room_id   # min-cost room
    assert s.history[0].fallback_reason == "timeout"
