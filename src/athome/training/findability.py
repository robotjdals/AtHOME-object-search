"""Is a target findable from a start pose? (dataset construction only)

The proposal's rollout reward R = -D - lambda_s N assumes every rollout ends
by finding the target (rollout until GT is found, proposal 6-2). That holds
only for start states whose target can be seen from some Search Location the
robot can reach, so training starts are restricted to those, as navigation
datasets keep only solvable episodes (Habitat episode generation) and GRPO
training drops prompts without a learning signal (DAPO dynamic sampling).
GT is used here to select start states only, never as planner input.

Rule ``any_goal``: a goal candidate of some Search Location, reachable from
the start cell, observes a target instance. This is optimistic: the session
stands at one goal per visit (the nearest from where the robot is then), so a
rollout can still end without the target (0.6% of the v5 Teacher episodes).
Rule ``every_goal``: every reachable goal candidate of some Search Location
observes a target, so visiting that location finds it from whichever side
the robot comes; a search that visits every location succeeds. Evaluation
and GRPO start states use ``every_goal``.
"""
from __future__ import annotations

from typing import Iterable, Optional

from athome.navigation import NavigationPlanner
from athome.navigation.path_cost import shortest_paths
from athome.schemas import Pose2D


def findable_from(navigation: NavigationPlanner, location_ids: Iterable[str], start: Pose2D,
                  observer, target_semantic_ids: Iterable[int], rule: str = "any_goal") -> Optional[str]:
    """First Search Location (sorted IDs) from which a target is observed, or None."""
    if rule not in ("any_goal", "every_goal"):
        raise ValueError(f"알 수 없는 규칙: {rule}")
    targets = set(target_semantic_ids)
    if not targets:
        return None
    grid = navigation.grid
    start_cell = grid.to_cell(start.x, start.y)
    candidates = {lid: navigation.candidate_goals(lid) for lid in sorted(location_ids)}
    goals = {cell for goals in candidates.values() for cell, _ in goals}
    reached, _ = shortest_paths(grid.free, start_cell, goals, grid.resolution)
    for lid, goals in candidates.items():
        sees = [bool(set(observer.observe(pose.x, pose.y, 0.0, pose.yaw)) & targets)
                for cell, pose in goals if cell in reached]
        if sees and (any(sees) if rule == "any_goal" else all(sees)):
            return lid
    return None


def oracle_distance(navigation: NavigationPlanner, location_ids: Iterable[str], start: Pose2D,
                    observer, target_semantic_ids: Iterable[int]) -> Optional[float]:
    """Shortest path cost [m] from the start to a goal pose that observes a
    target: the l* of SPL (Anderson et al. 2018) for object search, where the
    success region is the set of views of the target (Habitat ObjectNav
    view points)."""
    targets = set(target_semantic_ids)
    grid = navigation.grid
    seeing = {cell for lid in location_ids for cell, pose in navigation.candidate_goals(lid)
              if set(observer.observe(pose.x, pose.y, 0.0, pose.yaw)) & targets}
    if not seeing:
        return None
    reached, _ = shortest_paths(grid.free, grid.to_cell(start.x, start.y), seeing, grid.resolution)
    costs = [reached[c] for c in seeing if c in reached]
    return min(costs) if costs else None


def findable_starts(grid, count: int, seed: int, is_findable) -> list:
    """Uniform free cells (seeded permutation), keeping findable ones.
    Reachability is the same inside a 4-connected free region (8-connected
    moves without corner cutting), so ``is_findable(cell)`` runs once per region."""
    import numpy as np
    from scipy.ndimage import label
    regions, _ = label(grid.free)
    by_region = {}
    free = np.argwhere(grid.free)
    picks = []
    for i in np.random.default_rng(seed).permutation(len(free)):
        cell = tuple(int(v) for v in free[i])
        region = regions[cell]
        if region not in by_region:
            by_region[region] = is_findable(cell)
        if by_region[region]:
            picks.append(cell)
            if len(picks) == count:
                break
    return sorted(picks)
