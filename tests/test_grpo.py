import pytest

from athome.training.grpo import Decision, Rollout, group_advantages, is_trained, reward, step_advantages


def roll(r, success=True, trainable=True):
    return Rollout(decisions=[Decision("workspace", ["a", "b"], "a", trainable)], success=success, reward=r)


def test_reward_is_proposal_formula():
    assert reward(12.5, 4, 3.0) == pytest.approx(-24.5)


def test_group_relative_advantage():
    group = [roll(-10.0), roll(-20.0), roll(-30.0)]
    assert group_advantages(group) is None
    assert [round(r.advantage, 3) for r in group] == [1.225, 0.0, -1.225]


def test_unusable_groups_are_left_out():
    assert group_advantages([roll(-10.0), roll(-20.0, success=False)]) == "failed_rollout"
    assert group_advantages([roll(-10.0), roll(-10.0)]) == "equal_rewards"
    assert group_advantages([roll(-10.0, trainable=False), roll(-20.0, trainable=False)]) == "no_trainable_decision"


def step(selected, distance, visits, trainable=True):
    return Decision("workspace", ["a", "b"], selected, trainable, aliases={"A": "a", "B": "b"},
                    distance_before_m=distance, visits_before=visits)


def test_step_advantages_compare_decisions_in_the_same_state():
    # Same first decision (anchor ()), then two different states.
    r1 = Rollout(decisions=[step("a", 0.0, 0), step("a", 4.0, 1)], success=True, distance_m=10.0, visits=2)
    r2 = Rollout(decisions=[step("a", 0.0, 0), step("b", 4.0, 1)], success=True, distance_m=20.0, visits=2)
    r3 = Rollout(decisions=[step("b", 0.0, 0)], success=True, distance_m=5.0, visits=1)
    for r in (r1, r2, r3):
        r.reward = reward(r.distance_m, r.visits, 3.0)
    assert step_advantages([r1, r2, r3], 3.0) == 2
    # Anchor (): returns -16, -26, -8.
    first = [r.decisions[0].step_advantage for r in (r1, r2, r3)]
    assert first == pytest.approx([0.0905, -1.2675, 1.1770], abs=1e-3)
    # Anchor (a): return-to-go -(10 - 4) - 3 = -9 vs -(20 - 4) - 3 = -19.
    assert r1.decisions[1].step_advantage == pytest.approx(1.0)
    assert r2.decisions[1].step_advantage == pytest.approx(-1.0)


def test_step_advantage_is_zero_without_a_shared_state():
    r1 = Rollout(decisions=[Decision("room", ["x", "y"], "x", False), step("a", 2.0, 1)], distance_m=6.0, visits=2)
    r2 = Rollout(decisions=[Decision("room", ["x", "y"], "y", False), step("a", 3.0, 1)], distance_m=9.0, visits=2)
    assert step_advantages([r1, r2], 3.0) == 0
    assert [r.decisions[1].step_advantage for r in (r1, r2)] == [0.0, 0.0]


def test_untrained_and_single_candidate_decisions_are_not_trained():
    assert is_trained(step("a", 0.0, 0))
    assert not is_trained(step("a", 0.0, 0, trainable=False))
    assert not is_trained(Decision("workspace", ["a"], "a", True, aliases={"A": "a"}))
