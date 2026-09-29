import math

import pytest

from athome.inference.prompts import SYSTEM, TEACHER_SYSTEM, build_messages
from athome.search.policy import Candidate, PlanningContext, PolicyError, Stage
from athome.training.teacher import (
    MinCostTeacher, TeacherPolicy, parse_likelihoods, search_index_choice)

CANDS = [Candidate("_1", 6.0, {"room_label": "bathroom"}),
         Candidate("_2", 1.0, {"room_label": "bedroom"})]


def answer(p1, p2):
    return {"reasoning": "towels are kept in bathrooms",
            "likelihoods": [{"candidate": "C1", "likelihood": p1},
                            {"candidate": "C2", "likelihood": p2}]}


def test_teacher_uses_student_input_and_selects_by_search_index():
    seen = {}

    def complete(messages, schema):
        seen["messages"], seen["schema"] = messages, schema
        return answer(0.9, 0.1)

    teacher = TeacherPolicy(complete, step_cost=3.0)
    assert teacher.select(Stage.ROOM, "towel", CANDS, PlanningContext()) == "_1"
    student, _ = build_messages(Stage.ROOM, "towel", CANDS, PlanningContext())
    rec = teacher.records[0]
    assert rec.student_messages == student and student[0]["content"] == SYSTEM
    assert seen["messages"][0]["content"] == TEACHER_SYSTEM
    # Planner input without the search coverage line (applied by Bayes' rule).
    assert seen["messages"][1] == build_messages(Stage.ROOM, "towel", CANDS, PlanningContext(),
                                                 coverage=False)[0][1]
    assert "target not seen" in student[1]["content"]
    assert "target not seen" not in seen["messages"][1]["content"]
    assert rec.likelihoods == {"_1": 0.9, "_2": 0.1}
    assert rec.reason.startswith("towels")
    # 0.9/9 = 0.1 > 0.1/4 = 0.025 at lambda 3; small lambda prefers the near room
    assert rec.selection_by_step_cost["3"] == "_1"
    assert rec.selection_by_step_cost["inf"] == "_1"


def test_search_index_tradeoff_and_ties():
    p = {"_1": 0.5, "_2": 0.4}
    assert search_index_choice(CANDS, p, 0.5) == "_2"     # 0.5/6.5 < 0.4/1.5
    assert search_index_choice(CANDS, p, math.inf) == "_1"
    assert search_index_choice(CANDS, {"_1": 0.3, "_2": 0.3}, math.inf) == "_2"  # tie -> nearer


def test_invalid_likelihoods_rejected():
    aliases = {"C1": "_1", "C2": "_2"}
    with pytest.raises(ValueError):
        parse_likelihoods({"likelihoods": [{"candidate": "C1", "likelihood": 0.5}]}, aliases)
    with pytest.raises(ValueError):
        parse_likelihoods(answer(1.5, 0.1), aliases)
    teacher = TeacherPolicy(lambda m, s: answer(1.5, 0.1))
    with pytest.raises(PolicyError):
        teacher.select(Stage.ROOM, "towel", CANDS, PlanningContext())
    assert teacher.records[0].error


def test_single_candidate_needs_no_call_and_standin_is_nearest():
    teacher = TeacherPolicy(lambda m, s: pytest.fail("no call expected"))
    assert teacher.select(Stage.ROOM, "cup", CANDS[:1], PlanningContext()) == "_1"
    assert teacher.records[0].single_candidate
    assert MinCostTeacher().select(Stage.ROOM, "cup", CANDS, PlanningContext()) == "_2"


def test_room_likelihood_after_unsuccessful_search():
    from athome.search.policy import Candidate
    from athome.training.teacher import after_search, search_index_choice
    searched = Candidate("_6", 0.1, {"searched_locations": 4, "total_locations": 5})
    fresh = Candidate("_4", 0.1, {"searched_locations": 0, "total_locations": 5})
    post = after_search([searched, fresh], {"_6": 0.7, "_4": 0.2})
    assert post["_6"] == pytest.approx(0.14) and post["_4"] == pytest.approx(0.2)
    # The repeatedly searched room loses to the fresh one at equal cost.
    assert search_index_choice([searched, fresh], post, 3.0) == "_4"
    location = Candidate("ws:1", 1.0, {"kind": "workspace"})
    assert after_search([location], {"ws:1": 0.5}) == {"ws:1": 0.5}


def test_observed_fraction_drives_the_room_posterior():
    from athome.search.policy import Candidate
    from athome.training.teacher import after_search
    seen_all = Candidate("_16", 0.1, {"searched_locations": 1, "total_locations": 13, "observed_fraction": 1.0})
    assert after_search([seen_all], {"_16": 0.7}) == {"_16": 0.0}
