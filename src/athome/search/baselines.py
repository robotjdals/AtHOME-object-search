"""Baselines for the scene graph comparison: the same search with other
graph representations, and a planner without an LLM.

Representations (the planner sees the candidates these produce):
- unstructured object list (the "Unstructured LLM" baseline of MoMa-LLM,
  Honerkamp et al. RA-L 2024: object categories and positions without rooms,
  furniture grouping or relations): one decision over every object of the
  scene (``flat_graph`` + ``FlatSearchSession``);
- without the workspace layer (HSG-ON's ablation): rooms, then every object
  of the room, no workspaces, function labels or grouping (``flat_graph`` +
  SearchSession with group_standalone=False);
- without the observation state (``WithoutObservation``): rooms carry no
  searched / observed share, so nothing discounts a searched room.

``CooccurrencePolicy`` keeps the graph and replaces the LLM's semantic
likelihood by object-placement statistics of the training scenes (Kollar &
Roy, ICRA 2009; Zeng et al., semantic linking maps, ICRA 2020), with the
Teacher's selection rule (athome.training.teacher).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

from athome.scene_graph.query import normalize_category
from athome.search.policy import Candidate, PlanningContext, Stage
from athome.search.session import SearchSession


def flat_graph(graph: dict) -> dict:
    """The scene graph without its workspace layer: every object is its own
    node (role standalone) in its room, at its own position."""
    flat = copy.deepcopy(graph)
    flat["workspaces"] = []
    for obj in flat["objects"]:
        obj["role"] = "standalone"
        obj["workspace_id"] = None
        obj["parent_id"] = None
        obj["parent_type"] = None
    for room in flat["rooms"]:
        room["workspace_ids"] = []
        room["source_object_ids"] = []
        room["standalone_object_ids"] = sorted(
            o["object_id"] for o in flat["objects"] if o["room_id"] == room["room_id"])
    return flat


class FlatSearchSession(SearchSession):
    """One decision over every open object of the scene: no room stage, no
    grouping (a flat object graph has neither)."""

    def __init__(self, *args, **kwargs):
        kwargs["group_standalone"] = False
        kwargs["coverage"] = None
        super().__init__(*args, **kwargs)

    def _plan(self, target, pose):
        if target.known and not target.switched_to_unknown:
            return super()._plan(target, pose)
        costs = self.navigation.evaluate(pose, [l for l in self.graph.locations if self._open(l)])
        if not costs:
            return None
        candidates = [self._location_candidate(c) for _, c in sorted(costs.items())]
        chosen = self._select(Stage.STANDALONE, target.name, candidates)
        return self._decision(target, Stage.STANDALONE, costs[chosen])


OBSERVATION_KEYS = ("observed_fraction", "searched_locations", "total_locations")


class WithoutObservation:
    """Policy wrapper: room candidates without the observation state."""

    def __init__(self, policy):
        self.policy = policy

    def select(self, stage, target, candidates, context):
        stripped = [replace(c, info={k: v for k, v in c.info.items() if k not in OBSERVATION_KEYS})
                    for c in candidates]
        return self.policy.select(stage, target, stripped, context)

    def __getattr__(self, name):
        return getattr(self.policy, name)


@dataclass
class PlacementStats:
    """How often a target category was found in each kind of room / on or at
    each furniture category, out of how many of those there were (training
    scenes). Keys are normalized categories and room labels."""
    room: Dict[Tuple[str, str], int] = field(default_factory=dict)        # (target, room label) -> rooms with it
    rooms: Dict[str, int] = field(default_factory=dict)                   # room label -> rooms
    anchor: Dict[Tuple[str, str], int] = field(default_factory=dict)      # (target, furniture category) -> anchors with it
    anchors: Dict[str, int] = field(default_factory=dict)                 # furniture category -> instances
    targets: Dict[str, int] = field(default_factory=dict)                 # target -> scenes it was counted in

    def add_scene(self, graph: dict, instances: Mapping[str, Sequence[dict]]) -> None:
        """One training scene: its full graph and the target catalog
        (target -> instances with room_id and unmasked_graph_anchor)."""
        objects = {o["object_id"]: o for o in graph["objects"]}
        labels = {r["room_id"]: r["room_label"] for r in graph["rooms"]}
        for label in labels.values():
            self.rooms[label] = self.rooms.get(label, 0) + 1
        sources = {w["workspace_id"]: normalize_category(objects[w["source_object_id"]]["semantic_tag"])
                   for w in graph["workspaces"]}
        for category in sources.values():
            self.anchors[category] = self.anchors.get(category, 0) + 1
        for obj in graph["objects"]:
            if obj.get("role") == "standalone":
                c = normalize_category(obj["semantic_tag"])
                self.anchors[c] = self.anchors.get(c, 0) + 1
        for target, items in instances.items():
            t = normalize_category(target)
            self.targets[t] = self.targets.get(t, 0) + 1
            for label in {labels[i["room_id"]] for i in items if i["room_id"] in labels}:
                self.room[(t, label)] = self.room.get((t, label), 0) + 1
            anchored = set()
            for i in items:
                a = i.get("unmasked_graph_anchor") or {}
                if a.get("type") == "workspace" and a.get("id") in sources:
                    anchored.add(("w", a["id"], sources[a["id"]]))
                elif a.get("type") == "standalone" and a.get("id") in objects:
                    anchored.add(("s", a["id"], normalize_category(objects[a["id"]]["semantic_tag"])))
            for _, _, category in anchored:
                self.anchor[(t, category)] = self.anchor.get((t, category), 0) + 1

    def room_likelihood(self, target: str, label: str, prior: float = 1.0) -> float:
        """Share of rooms with this label that held the target (Beta(prior,
        prior) smoothing: unseen pairs and labels get 0.5 / few counts)."""
        t = normalize_category(target)
        return (self.room.get((t, label), 0) + prior) / (self.rooms.get(label, 0) + 2 * prior)

    def anchor_likelihood(self, target: str, category: str, prior: float = 1.0) -> float:
        t = normalize_category(target)
        c = normalize_category(category)
        return (self.anchor.get((t, c), 0) + prior) / (self.anchors.get(c, 0) + 2 * prior)

    def knows(self, target: str) -> bool:
        return normalize_category(target) in self.targets


class CooccurrencePolicy:
    """Likelihood from placement statistics instead of an LLM, then the
    Teacher's rule: after unsuccessful search, argmax p / (cost + step_cost).
    A target never seen in the training scenes gets no statistics: every
    candidate is equally likely, i.e. nearest first."""

    def __init__(self, stats: PlacementStats, step_cost: float = 3.0):
        self.stats = stats
        self.step_cost = step_cost

    def likelihoods(self, stage: Stage, target: str, candidates: Sequence[Candidate]) -> Dict[str, float]:
        if not self.stats.knows(target):
            return {c.candidate_id: 0.5 for c in candidates}
        out = {}
        for c in candidates:
            if stage == Stage.ROOM:
                out[c.candidate_id] = self.stats.room_likelihood(target, c.info.get("room_label", "unknown"))
            else:
                out[c.candidate_id] = self.stats.anchor_likelihood(target, c.info.get("category", "unknown"))
        return out

    def select(self, stage: Stage, target: str, candidates: Sequence[Candidate],
               context: Optional[PlanningContext] = None) -> str:
        from athome.training.teacher import after_search, search_index_choice

        posterior = after_search(candidates, self.likelihoods(stage, target, candidates))
        return search_index_choice(candidates, posterior, self.step_cost)
