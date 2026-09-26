import math

import pytest

from athome.execution import (
    VisitConfig,
    VisitExecutor,
    VisitReason,
    VisitRequest,
    VisitStatus,
)
from athome.execution.fake import (
    FakeClock,
    FakeMotionClient,
    FakeObservationSource,
)
from athome.execution.visit import Phase
from athome.schemas import MotionReason, NavigationRequest, Pose2D

G1 = Pose2D(1.0, 0.0, 0.0)
G2 = Pose2D(0.0, 1.0, math.pi / 2)


@pytest.fixture
def rig():
    clock = FakeClock()
    motion = FakeMotionClient(clock)
    perception = FakeObservationSource()
    executor = VisitExecutor(motion, perception, clock, VisitConfig())
    return clock, motion, perception, executor


def start(executor, goals=(G1,)):
    executor.start(VisitRequest("loc-1", tuple(goals), "map-v1"))


def observe_heading(clock, perception, executor, frames=3, dt=0.4):
    for _ in range(frames):
        clock.advance(dt)
        perception.publish(clock())
        executor.step()


def arrive(clock, motion, executor, pose=G1):
    motion.last.accept()
    clock.advance(0.1)
    motion.last.succeed(final_pose=pose)
    executor.step()
    assert executor.phase == Phase.OBSERVING


def test_visit_completes_after_four_headings(rig):
    clock, motion, perception, executor = rig
    start(executor)
    arrive(clock, motion, executor)

    for _ in range(3):
        observe_heading(clock, perception, executor)
        assert executor.phase == Phase.ROTATING
        motion.last.accept()
        motion.last.succeed()
        executor.step()
    observe_heading(clock, perception, executor)

    outcome = executor.outcome
    assert outcome.status == VisitStatus.COMPLETED
    assert outcome.goal == G1
    assert len(outcome.observations) == 12
    yaws = [h.request.target_yaw for h in motion.handles[1:]]
    assert yaws == pytest.approx([math.pi / 2, math.pi, -math.pi / 2])


def test_frames_captured_before_arrival_are_ignored(rig):
    clock, motion, perception, executor = rig
    start(executor)
    motion.last.accept()
    clock.advance(1.0)
    motion.last.succeed(final_pose=G1)
    executor.step()

    # Delayed frames captured while still moving arrive after the stop.
    for stamp in (0.2, 0.5, 0.8):
        perception.publish(stamp)
    clock.advance(1.2)
    perception.publish(clock())
    executor.step()
    assert executor.phase == Phase.OBSERVING


def test_transient_failure_retries_goal_then_next_candidate(rig):
    clock, motion, _, executor = rig
    start(executor, goals=(G1, G2))

    for _ in range(3):
        motion.last.accept()
        motion.last.fail(MotionReason.NO_PROGRESS)
        executor.step()
        if executor.phase == Phase.WAITING:   # retry wait before the same goal
            clock.advance(3.1)
            executor.step()

    goals = [h.request.goal for h in motion.handles]
    assert goals == [G1, G1, G2, G2]
    motion.last.accept()
    motion.last.fail(MotionReason.BLOCKED)
    executor.step()

    outcome = executor.outcome
    assert outcome.status == VisitStatus.FAILED
    assert outcome.reason == VisitReason.NAVIGATION_FAILED
    assert outcome.observations == []
    assert outcome.nav_attempts == 4


def test_invalid_goal_skips_to_next_candidate(rig):
    _, motion, _, executor = rig
    start(executor, goals=(G1, G2))
    motion.last.accept()
    motion.last.fail(MotionReason.INVALID_GOAL)
    executor.step()
    assert motion.last.request.goal == G2
    assert len(motion.handles) == 2


def test_localization_lost_pauses(rig):
    _, motion, _, executor = rig
    start(executor, goals=(G1, G2))
    motion.last.accept()
    motion.last.fail(MotionReason.LOCALIZATION_LOST)
    outcome = executor.step()
    assert outcome.status == VisitStatus.PAUSED
    assert outcome.reason == VisitReason.LOCALIZATION_LOST
    assert len(motion.handles) == 1


def test_user_cancel_waits_for_stop_confirmation(rig):
    clock, motion, _, executor = rig
    start(executor)
    motion.last.accept()
    executor.cancel()
    assert motion.last.cancel_requested

    clock.advance(0.5)
    assert executor.step() is None
    assert executor.phase == Phase.CANCELING
    with pytest.raises(RuntimeError):
        start(executor)

    motion.last.finish_canceled(stopped=True)
    outcome = executor.step()
    assert outcome.status == VisitStatus.CANCELED
    assert len(motion.handles) == 1


def test_cancel_timeout_blocks_new_motion_until_stopped(rig):
    clock, motion, _, executor = rig
    start(executor)
    canceled = motion.last
    canceled.accept()
    executor.cancel()
    clock.advance(6.0)
    outcome = executor.step()
    assert outcome.status == VisitStatus.PAUSED
    assert outcome.reason == VisitReason.STOP_UNCONFIRMED

    with pytest.raises(RuntimeError):
        start(executor)

    canceled.finish_canceled(stopped=True)
    start(executor)
    assert executor.phase == Phase.NAVIGATING


def test_feedback_loss_cancels_then_pauses(rig):
    clock, motion, _, executor = rig
    start(executor)
    motion.last.accept()
    clock.advance(0.5)
    motion.last.feedback()
    clock.advance(1.5)
    executor.step()
    assert motion.last.cancel_requested

    motion.last.finish_canceled()
    outcome = executor.step()
    assert outcome.status == VisitStatus.PAUSED
    assert outcome.reason == VisitReason.MOTION_UNRESPONSIVE


def test_goal_never_accepted_pauses(rig):
    clock, motion, _, executor = rig
    start(executor)
    clock.advance(6.0)
    executor.step()
    assert motion.last.cancel_requested
    motion.last.reject()
    outcome = executor.step()
    assert outcome.status == VisitStatus.PAUSED
    assert outcome.reason == VisitReason.MOTION_UNRESPONSIVE


def test_result_without_stop_pauses(rig):
    _, motion, _, executor = rig
    start(executor)
    motion.last.accept()
    motion.last.succeed(final_pose=G1, stopped=False)
    outcome = executor.step()
    assert outcome.status == VisitStatus.PAUSED
    assert outcome.reason == VisitReason.STOP_UNCONFIRMED


def test_perception_loss_during_observation_pauses(rig):
    clock, motion, _, executor = rig
    start(executor)
    arrive(clock, motion, executor)
    clock.advance(3.5)
    outcome = executor.step()
    assert outcome.status == VisitStatus.PAUSED
    assert outcome.reason == VisitReason.PERCEPTION_LOST


def test_rotation_failure_retries_then_fails(rig):
    clock, motion, perception, executor = rig
    start(executor)
    arrive(clock, motion, executor)
    observe_heading(clock, perception, executor)

    for _ in range(2):
        motion.last.accept()
        motion.last.fail(MotionReason.BLOCKED)
        executor.step()

    outcome = executor.outcome
    assert outcome.status == VisitStatus.FAILED
    assert outcome.reason == VisitReason.ROTATION_FAILED
    assert len(motion.handles) == 3


def test_cancel_while_observing_finishes_immediately(rig):
    clock, motion, _, executor = rig
    start(executor)
    arrive(clock, motion, executor)
    executor.cancel()
    assert executor.outcome.status == VisitStatus.CANCELED


def test_each_request_gets_its_own_id(rig):
    clock, motion, _, executor = rig
    start(executor, goals=(G1, G2))
    motion.last.accept()
    motion.last.fail(MotionReason.NO_PROGRESS)
    executor.step()
    clock.advance(3.1)
    executor.step()
    ids = [h.request_id for h in motion.handles]
    assert len(set(ids)) == 2
    assert all(isinstance(h.request, NavigationRequest) for h in motion.handles)


def test_retry_of_same_goal_waits_first(rig):
    clock, motion, _, executor = rig
    start(executor)
    motion.last.accept()
    motion.last.fail(MotionReason.BLOCKED)
    executor.step()
    assert executor.phase == Phase.WAITING
    clock.advance(2.0)
    executor.step()
    assert len(motion.handles) == 1
    clock.advance(1.5)
    executor.step()
    assert len(motion.handles) == 2 and motion.last.request.goal == G1


def test_executor_navigation_timeout_cancels_then_retries(rig):
    clock, motion, _, executor = rig
    start(executor)
    first = motion.last
    first.accept()
    for _ in range(181):            # feedback keeps coming, never finishes
        clock.advance(1.0)
        first.feedback()
        executor.step()
    assert first.cancel_requested and executor.phase == Phase.CANCELING
    first.finish_canceled(stopped=True)
    executor.step()
    assert executor.phase == Phase.WAITING      # TIMEOUT is retryable
    clock.advance(3.1)
    executor.step()
    assert len(motion.handles) == 2


def test_user_cancel_during_timeout_cancel_wins(rig):
    clock, motion, _, executor = rig
    start(executor)
    motion.last.accept()
    for _ in range(181):
        clock.advance(1.0)
        motion.last.feedback()
        executor.step()
    executor.cancel()
    motion.last.finish_canceled(stopped=True)
    outcome = executor.step()
    assert outcome.status == VisitStatus.CANCELED
    assert len(motion.handles) == 1


def test_short_perception_gap_is_waited_out(rig):
    clock, motion, perception, executor = rig
    start(executor)
    arrive(clock, motion, executor)
    clock.advance(2.0)              # stream silent for 2 s
    assert executor.step() is None
    observe_heading(clock, perception, executor)
    assert executor.phase == Phase.ROTATING
