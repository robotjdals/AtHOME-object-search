from athome.schemas import ObservedObject
from athome.search.matching import AreaMatcher, ClipRankVerifier, LabelMatcher, explain


def obj(label, conf=0.5, x=1.0, feature=(1.0, 0.0)):
    return ObservedObject(object_id=-1, label=label, confidence=conf, centroid=(x, 0.0, 0.5),
                          clip_feature=list(feature))


def matcher():
    # CLIP vocabulary: cup = (1, 0), bowl = (0, 1); a feature (0, 1) ranks cup 2nd.
    encode = lambda texts: [[1.0, 0.0] if "cup" in t else [0.0, 1.0] for t in texts]  # noqa: E731
    verify = ClipRankVerifier(LabelMatcher(0.15), encode, ["cup", "bowl"], max_rank=1)
    return AreaMatcher(verify, lambda x, y: x < 5.0)


def test_explain_gives_the_first_failing_check():
    m = matcher()
    assert explain(m, "cup", obj("cup")) == "ok"
    assert explain(m, "cup", obj("bowl")) == "label"
    assert explain(m, "cup", obj("cup", conf=0.1)) == "confidence"
    assert explain(m, "cup", obj("cup", feature=(0.0, 1.0))) == "clip:2"
    assert explain(m, "cup", obj("cup", x=6.0)) == "outside"
    for o in (obj("cup"), obj("bowl"), obj("cup", conf=0.1), obj("cup", feature=(0.0, 1.0)), obj("cup", x=6.0)):
        assert (explain(m, "cup", o) == "ok") == m.matches("cup", o)
