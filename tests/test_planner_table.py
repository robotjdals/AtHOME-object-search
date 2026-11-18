import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from planner_table import metrics, success  # noqa: E402


def run(found=True, distance=10.0, visits=4, spl=0.5):
    return {"found": found, "distance_m": distance, "visits": visits, "spl": spl}


def test_success_needs_the_objectnav_budget():
    assert success(run(distance=10, visits=4))            # cost 22
    assert not success(run(distance=100, visits=10))      # cost 130 > 125
    assert not success(run(found=False))


def test_failure_has_zero_spl():
    assert metrics(run(distance=100, visits=10))["SPL"] == 0.0
    assert metrics(run())["SPL"] == 0.5
