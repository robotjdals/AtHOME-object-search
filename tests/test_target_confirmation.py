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
