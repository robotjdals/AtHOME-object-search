"""Looking again: a turn to centre what no heading faces, the Known object
first, and visited locations again from another place."""

import math
from types import SimpleNamespace

from athome.execution import VisitConfig, VisitExecutor, VisitRequest, VisitStatus
from athome.execution.fake import FakeClock, FakeMotionClient, FakeObservationSource
from athome.execution.command import CommandPhase
from athome.execution.status import describe
from athome.execution.visit import Phase, VisitOutcome
from athome.navigation import NavigationPlanner
from athome.scene_graph.query import SceneGraph
from athome.schemas import ObservationFrame, ObservedObject, Pose2D
from athome.search import SearchSession, TargetStatus
from athome.search.matching import LabelMatcher
from athome.search.policy import Stage
from athome.symbolic.environment import SymbolicEnvironment, ViewpointDetectionModel
from athome.testing import toy_env

GRID = toy_env.toy_grid()
START = Pose2D(0.0, 0.0, 0.0)


def obj(x, y, confidence=0.8, label="remote"):
    return ObservedObject(object_id=-1, label=label, confidence=confidence, centroid=(x, y, 0.8), is_static=True)


def session(confirmations=2):
    return SearchSession(SceneGraph(toy_env.toy_graph(["remote"])), NavigationPlanner(GRID), ["remote"],
                         matcher=LabelMatcher(min_confidence=0.15), confirmations=confirmations)


def at_bearing(degrees, r=2.0):
    a = math.radians(degrees)
    return r * math.cos(a), r * math.sin(a)


class Robot:
    """A visit at the goal facing yaw 0, turning as commanded."""

    def __init__(self, s, relook_max=2, look_first=()):
        self.clock, self.perception = FakeClock(), FakeObservationSource()
        self.motion = FakeMotionClient(self.clock)
        self.visit = VisitExecutor(self.motion, self.perception, self.clock, VisitConfig(
            observation_window=0.5, observation_min_frames=1, observation_timeout=25.0,
            perception_timeout=15.0, heading_order="front_first", relook_max=relook_max))
        self.s = s
        self.visit.start(VisitRequest(
            "loc", (START,), "map", interest=s.is_candidate, confirm_frames=s.confirmations,
            satisfied=s.all_found_in, relook=s.relook_points, relook_result=s.relook_result,
            look_first=look_first))
        self._arrive(START)

    def _arrive(self, pose):
        self.motion.last.accept()
        self.clock.advance(1.0)
        self.motion.last.succeed(final_pose=pose)
        self.visit.step()

    def look(self, *objects):
        self.clock.advance(0.4)
        self.perception.publish(self.clock(), list(objects))
        self.clock.advance(0.4)
        self.visit.step()

    def turned_to(self):
        return round(math.degrees(self.motion.last.request.target_yaw))

    def finish_turn(self):
        self._arrive(Pose2D(0.0, 0.0, self.motion.last.request.target_yaw))


def test_near_miss_on_the_seam_is_centred_and_then_found():
    r = Robot(session())
    r.look(obj(*at_bearing(-30), confidence=0.08))    # between the 0 and -60 views
    assert r.visit.phase == Phase.ROTATING and r.turned_to() == -30
    r.finish_turn()
    r.look(obj(*at_bearing(-30), confidence=0.5))
    r.look(obj(*at_bearing(-30), confidence=0.5))
    assert r.visit.outcome.status == VisitStatus.COMPLETED
    assert r.visit.outcome.relooks[0]["kind"] == "near_miss" and r.visit.outcome.relooks[0]["passed"]
    assert r.visit.outcome.headings_observed == 1


def test_a_look_waits_for_the_frames_a_flicker_gets():
    r = Robot(session())
    r.look(obj(*at_bearing(-30), confidence=0.08))
    r.finish_turn()
    for _ in range(5):                                 # nothing yet
        r.look()
        assert r.visit.phase == Phase.OBSERVING
    r.look()
    assert r.visit.phase == Phase.ROTATING and r.turned_to() == -60    # on with the headings
    assert not r.visit._relooks[0]["passed"]


def test_near_miss_a_heading_faces_needs_no_turn():
    r = Robot(session())
    r.look(obj(*at_bearing(10), confidence=0.08))
    assert r.turned_to() == -60


def test_other_labels_never_trigger():
    r = Robot(session())
    r.look(obj(*at_bearing(-30), label="cup"))
    assert r.turned_to() == -60


def test_found_on_the_seam_stands_when_seen_again():
    s = session()
    r = Robot(s)
    r.look(obj(*at_bearing(-30)))
    r.look(obj(*at_bearing(-30)))                      # confirmed, off-centre
    assert r.visit.phase == Phase.ROTATING and r.turned_to() == -30
    r.finish_turn()
    centred = obj(*at_bearing(-30), confidence=0.7)
    r.look(centred)
    r.look(centred)
    assert r.visit.outcome.status == VisitStatus.COMPLETED
    assert r.visit.outcome.relooks == [dict(r.visit.outcome.relooks[0], kind="found", passed=True)]
    assert s._centred["remote"] == centred


def test_found_on_the_seam_is_taken_back_when_not_seen_again():
    s = session()
    r = Robot(s)
    r.look(obj(*at_bearing(-30)))
    r.look(obj(*at_bearing(-30)))
    r.finish_turn()
    for _ in range(6):
        r.look()
    assert r.visit.phase == Phase.ROTATING and r.turned_to() == -60    # search goes on
    assert s._confirmed("remote", r.visit._frames) is None


def test_found_a_heading_faces_ends_the_visit_without_a_turn():
    r = Robot(session())
    r.look(obj(*at_bearing(5)))
    r.look(obj(*at_bearing(5)))
    assert r.visit.outcome.status == VisitStatus.COMPLETED and not r.visit.outcome.relooks


def test_limit_per_visit():
    r = Robot(session(), relook_max=1)
    r.look(obj(*at_bearing(-30), confidence=0.1), obj(*at_bearing(30), confidence=0.09))
    assert r.turned_to() == -30
    r.finish_turn()
    for _ in range(6):
        r.look()
    assert r.turned_to() == -60


def test_frames_while_turning_only_add_candidates():
    s = session()
    r = Robot(s)
    r.look()
    assert r.turned_to() == -60
    r.clock.advance(0.5)
    r.perception.publish(r.clock(), [obj(*at_bearing(-30))])   # while turning
    r.perception.publish(r.clock() + 0.1, [obj(*at_bearing(-30))])
    r.clock.advance(0.5)
    r.finish_turn()
    r.look()
    assert r.visit.phase == Phase.ROTATING and r.turned_to() == -30
    assert s._confirmed("remote", r.visit._frames) is None


def test_known_object_is_looked_at_first():
    r = Robot(session(), look_first=(at_bearing(100),))
    assert r.visit.phase == Phase.ROTATING and r.turned_to() == 100
    r.finish_turn()
    r.look()
    assert r.turned_to() == 0                          # then the regular headings


def test_known_object_in_front_needs_no_extra_look():
    r = Robot(session(), look_first=(at_bearing(5),))
    assert r.visit.phase == Phase.OBSERVING and r.visit.progress()["heading_count"] == 6


def test_status_says_a_look_is_running():
    r = Robot(session())
    r.look(obj(*at_bearing(-30), confidence=0.08))

    command = SimpleNamespace(phase=CommandPhase.VISITING, session=r.s, decision=None)
    status = describe(command, r.visit.progress())
    assert status["state"] == "relooking" and "remote" in status["text"]


def known_session():
    return SearchSession(SceneGraph(toy_env.toy_graph([])), NavigationPlanner(GRID), ["remote"],
                         matcher=LabelMatcher(min_confidence=0.15))


def test_known_decision_says_where_the_object_is():
    s = known_session()
    decision = s.next_decision(Pose2D(0.5, 0.5, 0.0))
    assert decision.stage == Stage.KNOWN and len(decision.look_at) == 1


def _visit_all(s, frames_at=lambda d: []):
    decisions = []
    pose = Pose2D(0.5, 0.5, 0.0)
    while (d := s.next_decision(pose)) is not None:
        decisions.append(d)
        pose = d.goals[0]
        s.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED, goal=pose, final_pose=pose,
                                 observations=frames_at(d)))
    return decisions


def recheck_session(separation=1.0, confirmations=1):
    return SearchSession(SceneGraph(toy_env.toy_graph(["remote"])), NavigationPlanner(GRID), ["remote"],
                         matcher=LabelMatcher(min_confidence=0.15), recheck_separation_m=separation,
                         confirmations=confirmations)


def test_no_second_look_by_default():
    decisions = _visit_all(session(1))
    assert all(d.stage != Stage.RECHECK for d in decisions)


def test_second_look_goes_elsewhere_once_per_location():
    s = recheck_session()
    decisions = _visit_all(s)
    first = {d.location_id: d.goals[0] for d in decisions if d.stage != Stage.RECHECK}
    again = [d for d in decisions if d.stage == Stage.RECHECK]
    assert again and len({d.location_id for d in again}) == len(again)
    for d in again:
        stood = first[d.location_id]
        assert all(math.hypot(g.x - stood.x, g.y - stood.y) >= 1.0 for g in d.goals)
    assert s.targets[0].status == TargetStatus.FAILED


def test_second_look_starts_with_the_near_miss():
    near_at = {}

    def frames_at(d):
        if d.stage == Stage.RECHECK or near_at:
            return []
        if len(seen) == 1:          # the second location visited
            near_at[d.location_id] = True
            return [ObservationFrame(1.0, (obj(1.0, 1.0, confidence=0.08),))]
        seen.append(d.location_id)
        return []
    seen = []
    decisions = _visit_all(recheck_session(0.1), frames_at)
    again = [d.location_id for d in decisions if d.stage == Stage.RECHECK]
    assert near_at and again[0] in near_at


def test_location_without_another_place_is_skipped():
    decisions = _visit_all(recheck_session(1000.0))
    assert all(d.stage != Stage.RECHECK for d in decisions)


def test_found_at_the_second_look_counts():
    s = recheck_session(0.1)

    def frames_at(d):
        return ([ObservationFrame(1.0, (obj(1.0, 1.0),))] if d.stage == Stage.RECHECK else [])
    decisions = _visit_all(s, frames_at)
    assert decisions[-1].stage == Stage.RECHECK and s.targets[0].status == TargetStatus.FOUND


def test_simulated_detection_differs_only_from_another_place():
    model = ViewpointDetectionModel(0.5, 0.0, (0.0,), "scene")
    env = SymbolicEnvironment.__new__(SymbolicEnvironment)
    env._location_spots = {}
    a = env._viewpoint("loc", Pose2D(0.0, 0.0, 0.0))
    assert a == "loc" and env._viewpoint("loc", Pose2D(0.3, 0.0, 1.0)) == "loc"
    other = env._viewpoint("loc", Pose2D(2.0, 0.0, 0.0))
    assert other != "loc" and env._viewpoint("loc", Pose2D(2.1, 0.0, 0.0)) == other
    draws = {model.detected(f"o{i}", "loc", 1) == model.detected(f"o{i}", other, 1) for i in range(50)}
    assert draws == {True, False}


def _places(s, near_at_visit, confidence):
    """Visits in order; the ``near_at_visit``-th shows a candidate."""
    seen = []

    def frames_at(d):
        seen.append(d)
        if len(seen) == near_at_visit:
            return [ObservationFrame(1.0, (obj(1.0, 1.0, confidence=confidence),))]
        return []
    return _visit_all(s, frames_at)


def test_likely_candidate_is_looked_at_again_at_once():
    decisions = _places(recheck_session(0.1, confirmations=2), 1, 0.5)     # one frame of the two needed
    assert decisions[1].stage == Stage.RECHECK
    assert decisions[1].location_id == decisions[0].location_id
    again = [d.location_id for d in decisions if d.stage == Stage.RECHECK]
    assert len(again) == len(set(again))                  # still once per location


def test_weak_candidate_waits_until_nothing_is_left():
    decisions = _places(recheck_session(0.1), 1, 0.08)
    assert decisions[1].stage != Stage.RECHECK
    first_recheck = next(d for d in decisions if d.stage == Stage.RECHECK)
    assert first_recheck.location_id == decisions[0].location_id     # but first then


def test_plain_miss_moves_on():
    decisions = _visit_all(recheck_session(0.1))
    stages = [d.stage for d in decisions]
    assert Stage.RECHECK in stages
    assert all(s != Stage.RECHECK for s in stages[:stages.index(Stage.RECHECK)])
    assert all(s == Stage.RECHECK for s in stages[stages.index(Stage.RECHECK):])


def test_taken_back_find_is_looked_at_again_at_once():
    s = recheck_session(0.1)
    d = s.next_decision(Pose2D(0.5, 0.5, 0.0))
    s._rejected.append(("remote", 1.0, 1.0))
    s.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED, goal=d.goals[0], final_pose=d.goals[0]))
    again = s.next_decision(d.goals[0])
    assert again.stage == Stage.RECHECK and again.location_id == d.location_id
    assert not s._rejected


def test_no_look_again_without_the_setting():
    decisions = _places(session(2), 1, 0.5)
    assert all(d.stage != Stage.RECHECK for d in decisions)


def test_found_target_needs_no_second_look():
    s = recheck_session(0.1)
    decisions = _places(s, 1, 0.5)
    assert len(decisions) == 1 and s.targets[0].status == TargetStatus.FOUND


def test_first_visit_in_a_room_starts_from_its_entry_goal():
    entry = Pose2D(1.0, 1.0, 0.5)
    s = SearchSession(SceneGraph(toy_env.toy_graph(["remote"])), NavigationPlanner(GRID), ["remote"],
                      matcher=LabelMatcher(min_confidence=0.15),
                      room_entry_goals={"R_A": entry, "R_B": entry})
    d = s.next_decision(Pose2D(0.5, 0.5, 0.0))
    assert d.goals[0] == entry and len(d.goals) > 1          # the location's own goals follow
    s.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED, goal=entry, final_pose=entry))
    later = s.next_decision(Pose2D(0.5, 0.5, 0.0))
    assert later.room_id == d.room_id and entry not in later.goals    # same room again: its own goals only
