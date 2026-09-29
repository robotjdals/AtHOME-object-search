import pytest

from athome.training.grpo import Decision, Rollout, group_advantages, reward


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
