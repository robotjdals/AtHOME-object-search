import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from placement_typicality import count, shares  # noqa: E402


def test_shares_count_instances_per_room_type_over_the_home_rooms_only():
    graph = {"rooms": [{"room_id": "a", "room_label": "office"}, {"room_id": "b", "room_label": "bedroom"},
                       {"room_id": "c", "room_label": "bathroom"}],
             "objects": [{"semantic_tag": "Laptop", "room_id": "a"}, {"semantic_tag": "laptop", "room_id": "a"},
                         {"semantic_tag": "laptop", "room_id": "b"}, {"semantic_tag": "laptop", "room_id": "c"},
                         {"semantic_tag": "laptop", "room_id": "missing"},
                         {"semantic_tag": "stack of books", "room_id": "b"}]}
    exact = count([graph, graph], {"laptop": {"laptop"}, "book": {"book"}})
    assert exact[("laptop", "office")] == 4 and exact[("laptop", "bathroom")] == 2
    s = shares(exact, "laptop", ["bedroom", "kitchen", "office"])      # bathroom is not a room of the home
    assert s == {"bedroom": 2 / 6, "kitchen": 0.0, "office": 4 / 6}
    assert shares(exact, "book", ["office"]) == {"office": 0.0}
    assert count([graph], {"book": {"book", "stack of books"}})[("book", "bedroom")] == 1
