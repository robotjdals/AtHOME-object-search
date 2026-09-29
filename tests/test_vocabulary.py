from pathlib import Path

from athome.scene_graph.query import SceneGraph
from athome.scene_graph.vocabulary import Vocabulary
from athome.schemas import ObservedObject
from athome.search.matching import LabelMatcher
from athome.testing import toy_env

ROOT = Path(__file__).resolve().parents[1]


def test_raw_names_group_into_training_target_categories():
    v = Vocabulary.load(ROOT / "configs/data/target_categories.v2.json",
                        ROOT / "configs/data/search_locations.json")
    assert v.canonical("Bags") == "bag"          # raw_names of the category
    assert v.canonical("albums") == "album"      # merged_plurals
    assert v.canonical("Coffee  Table") == "coffee table"
    # Labels the HM3DSem mapping lacks are reported, not silently dropped.
    assert v.unmapped(["bag", "coffee table", "cupboard"]) == ["cupboard"]


def test_known_lookup_and_match_use_the_same_grouping():
    v = Vocabulary({"cups": "cup"})
    graph = toy_env.toy_graph()
    for obj in graph["objects"]:
        if obj["semantic_tag"] == "cup":
            obj["semantic_tag"] = "cups"
    # Planner-facing tags keep the raw name; the target still resolves.
    assert SceneGraph(graph).known_locations("cup") == []
    assert SceneGraph(graph, category_key=v.canonical).known_locations("cup") == [
        "workspace:R_A:table_1"]
    matcher = LabelMatcher(category=v.canonical)
    assert matcher.matches("cup", ObservedObject(0, "cups", 0.9, (0, 0, 0)))
    assert not LabelMatcher().matches("cup", ObservedObject(0, "cups", 0.9, (0, 0, 0)))


def test_subtypes_count_as_the_target_as_in_training_gt():
    v = Vocabulary.load(ROOT / "configs/data/target_categories.v5.json")
    assert "bedside lamp" in v.target_keys("lamp")
    assert v.is_target("lamp", "Table Lamp") and not v.is_target("bedside lamp", "lamp")
    # The training vocabulary merges "glasses" into "glass" (drinking glass):
    # eyeglasses must be named "eyeglasses" (command prompt 0.4).
    assert v.canonical("glasses") == "glass"

    graph = toy_env.toy_graph()
    for obj in graph["objects"]:
        if obj["semantic_tag"] == "lamp":
            obj["semantic_tag"] = "floor lamp"
    g = SceneGraph(graph, category_key=v.canonical, target_keys=v.target_keys)
    assert g.known_locations("lamp")                       # found through the subtype
    matcher = LabelMatcher(category=v.canonical, target_keys=v.target_keys)
    assert matcher.matches("lamp", ObservedObject(0, "desk lamp", 0.9, (0, 0, 0)))
    assert not matcher.matches("lamp", ObservedObject(0, "lampshade", 0.9, (0, 0, 0)))
