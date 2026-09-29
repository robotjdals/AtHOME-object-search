"""File layout of one HM3D scene pipeline version.

The component-stage scripts (floors, components A, B, ..., grids, goal
candidates, paths, symbolic review) read their inputs from a layout. The
default is the original single-scene layout under ``outputs/hm3d``; another
version is selected with ``ATHOME_HM3D_LAYOUT=<layout.json>``:

    {"scene_id": "wcojb4TFT35", "root": "outputs/hm3d_v3",
     "graph": "outputs/hm3d_v3/wcojb4TFT35.workspace_graph.json",
     "catalog": "outputs/hm3d_v3/wcojb4TFT35.target_catalog.json",
     "annotations": "outputs/hm3d_v3/wcojb4TFT35.annotations.json",
     "masked_graphs": "outputs/hm3d_v3/wcojb4TFT35/masked_graphs",
     "stair_selection": "outputs/hm3d/wcojb4TFT35/stair_triangle_selection.review.json"}

Relative paths are resolved from the repository root. The stair selection
depends only on the NavMesh and may be shared between versions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ENV = "ATHOME_HM3D_LAYOUT"


@dataclass(frozen=True)
class Layout:
    scene_id: str
    root: Path
    graph: Path
    catalog: Path
    annotations: Path
    masked_graphs: Path
    stair_selection: Path
    source: str = "default"
    masking_audit_dir: Path = None  # default: next to masked_graphs
    targets: Path = REPO / "configs/data/pilot_targets.json"  # target categories and aliases
    robot: Path = None  # robot config (configs/robot/*.yaml); None: dataset NavMesh agent (earlier layouts)
    # Search Location policy (configs/data/search_locations.json); None: DEFAULT_EXCLUDED_CATEGORIES.
    search_locations: Path = None

    def location_exclusions(self):
        from athome.scene_graph.location_policy import excluded_categories
        from athome.scene_graph.query import DEFAULT_EXCLUDED_CATEGORIES
        return (DEFAULT_EXCLUDED_CATEGORIES if self.search_locations is None
                else excluded_categories(self.search_locations))

    @property
    def masking_audit(self) -> Path:
        """<target>.audit.json of build_masked_inputs.py (objects hidden by masking)."""
        return self.masking_audit_dir or self.masked_graphs.parent / "masking_audit"

    @property
    def scene_dir(self) -> Path:
        return self.root / self.scene_id

    @property
    def floors(self) -> Path:
        return self.scene_dir / "floor_environments.json"

    @property
    def room_report(self) -> Path:
        return self.scene_dir / "component_room_candidates.review.json"

    @property
    def membership(self) -> Path:
        return self.scene_dir / "component_membership.review.json"

    @property
    def decisions(self) -> Path:
        return self.scene_dir / "component_membership.decisions.review.json"

    @property
    def navigation(self) -> Path:
        return self.scene_dir / "component_navigation.review.json"

    def component_names(self):
        """Navigable components of the room report (A, B, ...), in report order."""
        report = json.loads(self.room_report.read_text(encoding="utf-8"))
        return [c["component"] for c in report["components"]]

    @property
    def grid_dir(self) -> Path:
        """Named by the room-report hash, as build_component_grids.py does.
        Without the report (a layout not built on this machine, e.g. the
        default scene on a training server) the path does not exist, so
        importing modules that read it stays possible and a use fails with
        FileNotFoundError."""
        if not self.room_report.exists():
            return self.scene_dir / "component_grids" / "missing_room_report"
        digest = hashlib.sha256(self.room_report.read_bytes()).hexdigest()
        return self.scene_dir / "component_grids" / f"{digest[:12]}_5cm"


def default_layout() -> Layout:
    root = REPO / "outputs/hm3d"
    scene = "wcojb4TFT35"
    return Layout(
        scene_id=scene, root=root,
        graph=root / f"{scene}.workspace_graph.v2.json",
        catalog=root / f"{scene}.target_catalog.json",
        annotations=root / f"{scene}.annotations.json",
        masked_graphs=root / scene / "masked_graphs",
        stair_selection=root / scene / "stair_triangle_selection.review.json",
        masking_audit_dir=root / scene / "masking_audit_v2",
    )


def load_layout(path) -> Layout:
    path = Path(path)
    if not path.is_absolute():  # same base as the paths inside the file
        path = REPO / path
    data = json.loads(path.read_text(encoding="utf-8"))
    resolve = lambda key: (REPO / data[key]).resolve()  # noqa: E731
    return Layout(
        scene_id=data["scene_id"], root=resolve("root"), graph=resolve("graph"),
        catalog=resolve("catalog"), annotations=resolve("annotations"),
        masked_graphs=resolve("masked_graphs"), stair_selection=resolve("stair_selection"),
        source=str(Path(path).resolve()),
        masking_audit_dir=resolve("masking_audit") if "masking_audit" in data else None,
        **({"targets": resolve("targets")} if "targets" in data else {}),
        **({"robot": resolve("robot")} if "robot" in data else {}),
        **({"search_locations": resolve("search_locations")} if "search_locations" in data else {}),
    )


def repo_path(path) -> Path:
    """A path recorded in an output file, on this machine. Records made on
    another checkout hold that machine's absolute path; the part from the
    repository's ``data/`` or ``outputs/`` folder on is resolved against this
    checkout (e.g. the GPU server)."""
    path = Path(path)
    if path.exists():
        return path
    parts = path.parts
    for anchor in ("data", "outputs"):
        if anchor in parts:
            candidate = REPO.joinpath(*parts[parts.index(anchor):])
            if candidate.exists():
                return candidate
    return path


def current_layout() -> Layout:
    path = os.environ.get(ENV)
    return load_layout(path) if path else default_layout()
