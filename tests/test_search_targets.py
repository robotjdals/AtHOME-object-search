from athome.scene_graph.vocabulary import Vocabulary
from athome.search.session import TargetRecord, TargetStatus
from athome.search.targets import search_targets
from athome.scene_graph.location_policy import REPO

VOCAB = Vocabulary.load(REPO / "configs/data/target_categories.v5.json")


def test_open_targets_with_their_query_names():
    targets = [TargetRecord("lamp", known=False),
               TargetRecord("cup", known=False, status=TargetStatus.FOUND),
               TargetRecord("remote control", known=False, status=TargetStatus.SEARCHING),
               TargetRecord("umbrella", known=False, status=TargetStatus.FAILED)]
    out = dict(search_targets(targets, VOCAB.target_keys))
    assert set(out) == {"lamp", "remote control"}          # found and failed are left out
    # A subtype also counts as the target (training GT rule), so perception queries it.
    assert out["lamp"][0] == "lamp" and "desk lamp" in out["lamp"]
    assert out["remote control"][0] == "remote control"
    assert search_targets([], VOCAB.target_keys) == []
