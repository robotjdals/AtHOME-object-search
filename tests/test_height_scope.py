import copy
import importlib.util
from pathlib import Path

from athome.scene_graph.query import SceneGraph
from athome.testing import toy_env


def raised(graph, oid, dz):
    graph = copy.deepcopy(graph)
    obj = next(o for o in graph["objects"] if o["object_id"] == oid)
    obj["bbox"]["min"][2] += dz
    obj["bbox"]["max"][2] += dz
    return graph


def test_high_child_does_not_make_target_known():
    base = toy_env.toy_graph()
    child = next(o for o in base["objects"] if o["role"] == "child")
    category = child["semantic_tag"]
    assert SceneGraph(base).known_locations(category)
    others = [o for o in base["objects"] if o["semantic_tag"] == category and o is not child]
    graph = raised(base, child["object_id"], 3.0)
    for o in others:
        graph = raised(graph, o["object_id"], 3.0)
    assert SceneGraph(graph).known_locations(category) == []


def with_parents(graph):
    """Toy graphs omit parent_id, which workspace_builder graphs carry."""
    graph = copy.deepcopy(graph)
    parent = {c: w["workspace_id"] for w in graph["workspaces"] for c in w["child_object_ids"]}
    for obj in graph["objects"]:
        obj.setdefault("parent_id", parent.get(obj["object_id"], obj["room_id"]))
    graph.setdefault("length_unit", "meter")
    return graph


def test_catalog_excludes_high_instances():
    path = Path(__file__).resolve().parents[1] / "scripts/build_target_catalog.py"
    spec = importlib.util.spec_from_file_location("build_target_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = with_parents(toy_env.toy_graph())
    child = next(o for o in base["objects"] if o["role"] == "child")
    config = {"target_categories": [child["semantic_tag"]], "category_aliases": {}}
    low = module.build_catalog(base, config, "toy", "g.json")
    high = module.build_catalog(raised(base, child["object_id"], 3.0), config, "toy", "g.json")
    ids = lambda c: {i["object_id"] for i in c["targets"][0]["instances"]}  # noqa: E731
    assert child["object_id"] in ids(low) and child["object_id"] not in ids(high)
    assert child["object_id"] in high["height_scope"]["excluded_object_ids"][child["semantic_tag"]]


def test_catalog_counts_subtypes_for_the_parent_target():
    path = Path(__file__).resolve().parents[1] / "scripts/build_target_catalog.py"
    spec = importlib.util.spec_from_file_location("build_target_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = with_parents(toy_env.toy_graph())
    child = next(o for o in base["objects"] if o["role"] == "child")
    tag = child["semantic_tag"]
    config = {"target_categories": ["gadget", tag], "category_aliases": {},
              "target_members": {"gadget": [tag]}}
    catalog = module.build_catalog(base, config, "toy", "g.json")
    ids = {t["target_category"]: {i["object_id"] for i in t["instances"]} for t in catalog["targets"]}
    assert child["object_id"] in ids["gadget"] and child["object_id"] in ids[tag]


def test_masking_removes_subtypes_of_the_target():
    from athome.data.hm3d.target_masking import mask_target_inputs
    obj = lambda oid, tag: {"object_id": oid, "semantic_tag": tag, "neighbors": [],  # noqa: E731
                            "child_candidate_ids": []}
    inputs = {"rooms": [{"room_id": "r", "coordinate_frame": "z_up", "up_axis": "z", "length_unit": "m",
                         "geometry_thresholds": {},
                         "objects": [obj("a", "lamp"), obj("b", "bedside lamp"), obj("c", "paper towel")]}]}
    masked, audit = mask_target_inputs(inputs, "lamp", {}, members=["bedside lamp"])
    assert [o["object_id"] for o in masked["rooms"][0]["objects"]] == ["c"]
    assert {r["reason"] for r in audit["removed_objects"]} == {"target_category", "target_subtype"}
