"""Ground-truth verification of Teacher selections (proposal 6-3).

GT stays outside the planner input; it is used only here.
- Workspace selection: the chosen Workspace is a GT Workspace (the target is
  on it) or its goal pose observes a target, i.e. choosing it finds the
  target. Search success is observing the target (MoMa-LLM, Habitat
  ObjectNav); a standalone target seen from a Workspace counts as well, the
  same rule as the Standalone stage. ``workspace_rule="gt_workspace"`` keeps
  the former rule (GT Workspace only) for comparison.
- Standalone selection: from the goal pose the robot would use from its
  current position, the observation model sees a target instance.
- Room selection: the room contains a target instance (GT room layout,
  HM3DSem regions = the proposal's "GT Room") and still has an open,
  reachable Search Location. Room-level correctness by containment is how
  hierarchical scene-graph search is judged (HOV-SG assigns objects to GT
  room layouts; MoMa-LLM assigns objects to rooms by position). The former
  rule ("a location of the room observes the target") could mark a room
  correct because the target is visible from it through a doorway, and the
  room the target is in wrong; ``room_rule="observation"`` keeps it for
  comparison.
A failed selection is excluded from SFT, never replaced by a GT answer.
"""
from __future__ import annotations

from typing import Dict, Iterable, Set

from athome.schemas import Pose2D


class GroundTruth:
    def __init__(self, graph, navigation, observer, target_semantic_ids: Iterable[int],
                 gt_workspaces: Iterable[str], gt_rooms: Iterable[str] = (),
                 room_rule: str = "containment", workspace_rule: str = "observation"):
        if workspace_rule not in ("observation", "gt_workspace"):
            raise ValueError(f"알 수 없는 Workspace 검증 규칙: {workspace_rule}")
        if room_rule not in ("containment", "observation"):
            raise ValueError(f"알 수 없는 방 검증 규칙: {room_rule}")
        self.graph = graph
        self.navigation = navigation
        self.observer = observer
        self.targets = set(target_semantic_ids)
        self.gt_workspaces = {w for w in gt_workspaces if w in graph.locations}
        self.gt_rooms = set(gt_rooms)
        self.room_rule = room_rule
        self.workspace_rule = workspace_rule

    def valid_locations(self, pose: Pose2D, open_ids: Iterable[str]) -> Dict[str, dict]:
        """Open reachable locations that count as GT, with the reason."""
        costs = self.navigation.evaluate(pose, list(open_ids))
        out = {}
        for lid, cost in costs.items():
            goal = cost.goals[0]
            seen = set(self.observer.observe(goal.x, goal.y, 0.0, goal.yaw)) & self.targets
            is_ws = lid in self.gt_workspaces
            if seen or is_ws:
                out[lid] = {"gt_workspace": is_ws, "observes_target": bool(seen),
                            "goal": [goal.x, goal.y, goal.yaw]}
        return out

    def verify(self, stage: str, selected: str, candidate_ids, pose: Pose2D,
               visited: Set[str]) -> dict:
        open_ids = [l for l in self.graph.locations if l not in visited]
        valid = self.valid_locations(pose, open_ids)
        if stage == "room" and self.room_rule == "containment":
            reachable = self.navigation.evaluate(pose, open_ids)
            rooms = {self.graph.locations[l].room_id for l in reachable} & self.gt_rooms
            return {"passed": selected in rooms,
                    "valid_candidates": sorted(set(candidate_ids) & rooms),
                    "room_rule": "containment"}
        if stage == "room":
            rooms = {self.graph.locations[l].room_id for l in valid}
            return {"passed": selected in rooms,
                    "valid_candidates": sorted(set(candidate_ids) & rooms)}
        valid_ids = set(valid)
        if stage == "workspace":
            ok = ({c for c in candidate_ids if c in valid_ids} if self.workspace_rule == "observation"
                  else {c for c in candidate_ids if c in self.gt_workspaces and c in valid_ids})
            return {"passed": selected in ok, "valid_candidates": sorted(ok),
                    "selected_observes_target": bool(valid.get(selected, {}).get("observes_target"))}
        if stage == "standalone":
            ok = {c for c in candidate_ids if valid.get(c, {}).get("observes_target")}
            return {"passed": selected in ok, "valid_candidates": sorted(ok)}
        raise ValueError(f"검증 대상이 아닌 단계: {stage}")
