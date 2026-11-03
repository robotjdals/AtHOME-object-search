"""A detection counts as a target only with enough confidence and when seen
again at the same place (robot setting; the symbolic environment keeps 1)."""

import pytest

from athome.execution import VisitConfig, VisitExecutor, VisitRequest
from athome.execution.command import CommandExecutor, CommandStatus
from athome.execution.fake import FakeClock, FakeMotionClient, FakeObservationSource
from athome.execution.visit import Phase
from athome.navigation import NavigationPlanner
from athome.scene_graph.query import SceneGraph
from athome.schemas import ObservationFrame, ObservedObject, Pose2D
from athome.search import SearchSession, TargetStatus
from athome.search.matching import LabelMatcher
from athome.testing import toy_env
from athome.testing.sim import SimRobot

GRID = toy_env.toy_grid()


def obj(label, x, y, confidence=0.8, static=True):
    return ObservedObject(object_id=-1, label=label, confidence=confidence,
                          centroid=(x, y, 0.8), is_static=static)


def session(confirmations, matcher=None):
    return SearchSession(SceneGraph(toy_env.toy_graph(["remote"])), NavigationPlanner(GRID),
                         ["remote"], matcher=matcher or LabelMatcher(),
                         confirmations=confirmations)


def frames(*objects_per_frame):
    return [ObservationFrame(float(i), tuple(objs)) for i, objs in enumerate(objects_per_frame)]


def test_one_detection_is_enough_by_default():
    assert session(1)._confirmed("remote", frames([obj("remote", 1.0, 1.0)])) is not None


def test_confirmation_needs_the_same_place_twice():
    s = session(2)
    once = frames([obj("remote", 1.0, 1.0)], [obj("cup", 3.0, 3.0)])
    assert s._confirmed("remote", once) is None
    elsewhere = frames([obj("remote", 1.0, 1.0)], [obj("remote", 4.0, 1.0)])
    assert s._confirmed("remote", elsewhere) is None
    twice = frames([obj("remote", 1.0, 1.0, 0.6)], [obj("remote", 1.2, 1.1, 0.9)])
    found = s._confirmed("remote", twice)
    assert found is not None and found.confidence == pytest.approx(0.9)


def test_low_confidence_and_dynamic_detections_never_count():
    s = session(1, LabelMatcher(min_confidence=0.4))
    assert s._confirmed("remote", frames([obj("remote", 1.0, 1.0, confidence=0.3)])) is None
    assert s._confirmed("remote", frames([obj("remote", 1.0, 1.0, static=False)])) is None
    assert s.is_candidate(obj("remote", 0, 0)) and not s.is_candidate(obj("cup", 0, 0))


def _observing(confirm_frames):
    clock, motion, perception = FakeClock(), None, FakeObservationSource()
    motion = FakeMotionClient(clock)
    visit = VisitExecutor(motion, perception, clock, VisitConfig(
        observation_window=0.5, observation_min_frames=1, observation_timeout=25.0,
        perception_timeout=15.0))
    visit.start(VisitRequest("loc", (Pose2D(1.0, 0.0, 0.0),), "map",
                             interest=lambda o: o.label == "remote",
                             confirm_frames=confirm_frames))
    motion.last.accept()
    clock.advance(0.1)
    motion.last.succeed(final_pose=Pose2D(1.0, 0.0, 0.0))
    visit.step()
    assert visit.phase == Phase.OBSERVING
    return clock, motion, perception, visit


def test_heading_waits_for_a_second_frame_only_when_a_target_shows_up():
    clock, motion, perception, visit = _observing(confirm_frames=2)
    clock.advance(7.0)
    perception.publish(clock(), [obj("cup", 1, 1)])          # nothing of interest
    visit.step()
    assert visit.phase == Phase.ROTATING                      # one frame is enough
    motion.last.accept()
    clock.advance(1.0)
    motion.last.succeed(final_pose=Pose2D(1.0, 0.0, 1.05))
    visit.step()
    clock.advance(7.0)
    perception.publish(clock(), [obj("remote", 1, 1)])       # possible target
    visit.step()
    assert visit.phase == Phase.OBSERVING                     # stay for a second look
    clock.advance(7.0)
    perception.publish(clock(), [obj("remote", 1, 1)])
    visit.step()
    assert visit.phase == Phase.ROTATING


def test_unconfirmed_candidate_does_not_pause_the_visit():
    clock, motion, perception, visit = _observing(confirm_frames=2)
    clock.advance(7.0)
    perception.publish(clock(), [obj("remote", 1, 1)])
    waited = 0.0
    while visit.phase == Phase.OBSERVING and waited < 60:      # no further frame
        clock.advance(1.0)
        waited += 1.0
        visit.step()
    assert visit.phase == Phase.ROTATING                      # moved on, not PERCEPTION_LOST
    assert 17.0 <= waited <= 20.0       # 7 s + this = observation timeout (25 s) after stopping


def test_slow_perception_with_confirmation_still_finds_the_target():
    timing = VisitConfig(observation_window=0.5, observation_min_frames=1,
                         observation_timeout=25.0, perception_timeout=15.0)
    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0), perception_rate=1 / 7.0)
    s = SearchSession(SceneGraph(toy_env.toy_graph(["remote"])), NavigationPlanner(GRID),
                      ["remote"], confirmations=2)
    command = CommandExecutor(VisitExecutor(robot, robot, clock, timing), robot, "toy")
    command.start(s)
    result = None
    while result is None and clock() < 3600:
        result = command.step()
        robot.update(0.05)
    assert result.status == CommandStatus.COMPLETED
    assert result.targets[0].status == TargetStatus.FOUND


def test_found_target_keeps_the_capture_time_of_its_frame():
    from athome.execution.visit import VisitOutcome, VisitStatus
    s = session(2)
    d = s.next_decision(Pose2D(2.5, 3.0, 0.0))
    seen = frames([obj("remote", 1.0, 1.0, 0.6)], [obj("cup", 3.0, 3.0)], [obj("remote", 1.1, 1.0, 0.9)])
    s.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED, observations=tuple(seen)))
    t = s.targets[0]
    assert t.status == TargetStatus.FOUND and t.found_object.confidence == pytest.approx(0.9)
    assert t.found_stamp == 2.0                       # the frame of the most confident sighting


def _run(satisfied_stop: bool):
    timing = VisitConfig(observation_window=0.5, observation_min_frames=1)
    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0), perception_rate=2.0)
    s = SearchSession(SceneGraph(toy_env.toy_graph(["remote"])), NavigationPlanner(GRID),
                      ["remote"], confirmations=2)
    if not satisfied_stop:
        s.all_found_in = lambda frames: False
    command = CommandExecutor(VisitExecutor(robot, robot, clock, timing), robot, "toy")
    outcomes = []
    command._on_step = lambda d, r, o: outcomes.append(o)
    command.start(s)
    result = None
    while result is None and clock() < 3600:
        result = command.step()
        robot.update(0.05)
    return result, outcomes, clock()


def test_the_visit_stops_at_the_heading_that_confirms_the_last_target():
    full, full_visits, full_time = _run(satisfied_stop=False)
    early, early_visits, early_time = _run(satisfied_stop=True)
    assert full.targets[0].status == early.targets[0].status == TargetStatus.FOUND
    assert full_visits[-1].headings_observed == 6
    assert early_visits[-1].headings_observed < 6 and early_time < full_time
    # Visits before the finding one still observe every heading.
    assert all(o.headings_observed == 6 for o in early_visits[:-1])


def test_a_candidate_the_next_frames_do_not_show_was_a_flicker():
    clock, motion, perception, visit = _observing(confirm_frames=2)
    clock.advance(0.6)
    perception.publish(clock(), [obj("remote", 1, 1)])
    visit.step()
    assert visit.phase == Phase.OBSERVING                     # stay for a second look
    waited = 0.0
    while visit.phase == Phase.OBSERVING and waited < 25:     # 3 Hz, never again
        clock.advance(0.33)
        waited += 0.33
        perception.publish(clock(), [])
        visit.step()
    assert visit.phase == Phase.ROTATING and waited < 3.0     # not the 25 s observation timeout
