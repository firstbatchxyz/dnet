import pytest

from dnet.api.utils import (
    postprocess_single_round,
    compute_layer_assignments,
    optimize_device_ordering,
)
from distilp.solver import HALDAResult
from tests.fakes import FakeTBConn

pytestmark = pytest.mark.api


def _make_solution(w, n, k):
    return HALDAResult(
        w=w[:], n=n[:], k=k, obj_value=0.0, sets={"M1": [], "M2": [], "M3": []}
    )


# postprocess_single_round


def _is_adjacent_ring(order, a, b):
    i = order.index(a)
    return order[(i - 1) % len(order)] == b or order[(i + 1) % len(order)] == b


def test_postprocess_single_round_multiple_rounds_no_change():
    names = ["A", "B"]
    sol = _make_solution(w=[2, 1], n=[1, 0], k=2)

    out_names, out_sol = postprocess_single_round(names[:], sol)
    assert out_names == names
    assert out_sol.w == [2, 1]
    assert out_sol.n == [1, 0]


# B removed, counts re-assigned to left neighbor A (2 < 3)
def test_postprocess_single_round_basic():
    names = ["A", "B", "C"]
    sol = _make_solution(w=[2, 1, 3], n=[1, 1, 2], k=1)

    out_names, out_sol = postprocess_single_round(names[:], sol)
    assert out_names == ["A", "C"]
    assert out_sol.w == [3, 3]
    assert out_sol.n == [2, 2]
    assert 1 not in out_sol.w


def test_postprocess_single_round_multi():
    names = ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6"]
    sol = _make_solution(w=[4, 1, 5, 1, 1, 2], n=[4, 1, 4, 1, 1, 1], k=1)

    out_names, out_sol = postprocess_single_round(names[:], sol)

    assert out_names == ["dev1", "dev3", "dev5", "dev6"]
    assert out_sol.k == 1
    assert out_sol.w == [5, 5, 2, 2]
    assert out_sol.n == [5, 4, 2, 1]


def test_compute_assignments_basic():
    shard_order = ["A", "B"]
    w = [2, 1]
    n = [1, 0]
    k = 2

    assigns = compute_layer_assignments(shard_order, w, n, k)
    by_name = {a.instance: a for a in assigns}
    a = by_name["A"]
    b = by_name["B"]

    assert a.layers == [[0, 1], [3, 4]]
    assert b.layers == [[2], [5]]
    assert a.window_size == 2 and a.residency_size == 1
    assert b.window_size == 1 and b.residency_size == 0
    assert a.next_instance == "B"
    assert b.next_instance == "A"


# compute_layer_assignments


def test_compute_assignments_three_devices_two_rounds():
    shard_order = ["dev1", "dev2", "dev3"]
    w = [3, 5, 4]
    n = [3, 4, 2]
    k = 2

    assigns = compute_layer_assignments(shard_order, w, n, k)

    # dev1
    a0 = assigns[0]
    assert a0.instance == "dev1"
    assert a0.layers == [[0, 1, 2], [12, 13, 14]]
    assert a0.residency_size == 3
    assert a0.next_instance == "dev2"

    # dev2
    a1 = assigns[1]
    assert a1.instance == "dev2"
    assert a1.layers == [[3, 4, 5, 6, 7], [15, 16, 17, 18, 19]]
    assert a1.residency_size == 4
    assert a1.next_instance == "dev3"

    # dev3
    a2 = assigns[2]
    assert a2.instance == "dev3"
    assert a2.layers == [[8, 9, 10, 11], [20, 21, 22, 23]]
    assert a2.residency_size == 2
    assert a2.next_instance == "dev1"


def test_compute_assignments_single_device_three_rounds():
    shard_order = ["dev1"]
    w = [2]
    n = [2]
    k = 3

    assigns = compute_layer_assignments(shard_order, w, n, k)

    a0 = assigns[0]
    assert a0.instance == "dev1"
    assert a0.layers == [[0, 1], [2, 3], [4, 5]]
    assert a0.residency_size == 2
    assert a0.next_instance == "dev1"


def test_compute_assignments_mismatch_raises():
    with pytest.raises(ValueError):
        compute_layer_assignments(["A", "B"], [2], [1, 0], 1)


# optimize_device_ordering


def test_optimize_ordering_adjacent_tb():
    profiles = {"A": object(), "B": object(), "C": object(), "D": object()}
    tb = {
        "A": {"B": FakeTBConn("10.0.0.2", "B")},
        "B": {"A": FakeTBConn("10.0.0.1", "A"), "C": FakeTBConn("10.0.0.3", "C")},
        "C": {"B": FakeTBConn("10.0.0.2", "B")},
        "D": {},  # D isolated
    }

    ordered = optimize_device_ordering(profiles, tb)
    assert sorted(ordered) == sorted(profiles.keys())

    # Due to greedy ordering and set iteration, exact order is not deterministic.
    # Assert at least one TB edge is adjacent in the ring.
    adj = 0
    adj += 1 if _is_adjacent_ring(ordered, "A", "B") else 0
    adj += 1 if _is_adjacent_ring(ordered, "B", "C") else 0
    assert adj >= 1


def test_optimize_ordering_two_pairs():
    profiles = {
        k: object() for k in ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6", "dev7"]
    }
    tb = {
        "dev3": {"dev1": FakeTBConn("10.0.0.2", "dev1")},
        "dev1": {"dev3": FakeTBConn("10.0.0.3", "dev3")},
        "dev2": {"dev6": FakeTBConn("10.0.0.6", "dev6")},
        "dev6": {"dev2": FakeTBConn("10.0.0.2", "dev2")},
    }
    ordered = optimize_device_ordering(profiles, tb)

    def adj(order, a, b):  # ensure both pairs are adjacent in the ring ordering
        return _is_adjacent_ring(order, a, b)

    assert adj(ordered, "dev1", "dev3")
    assert adj(ordered, "dev2", "dev6")


def test_optimize_ordering_unknown_neighbors():
    profiles = {"X": object(), "Y": object()}
    tb = {
        "X": {"Z": FakeTBConn("10.0.0.9", "Z")},  # Z not in profiles
        "Y": {},
    }
    ordered = optimize_device_ordering(profiles, tb)
    assert sorted(ordered) == ["X", "Y"]
