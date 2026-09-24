"""Tests for the optimization substrate envelope."""
from quilt_optimization.substrate import (
    OptimizationReceipt, OptimizationStatus, status_to_polarity,
    ACCEPT, DRIFT, REFUSE,
)


def test_status_to_polarity_optimal():
    assert status_to_polarity(OptimizationStatus.OPTIMAL) == ACCEPT


def test_status_to_polarity_feasible_found():
    assert status_to_polarity(OptimizationStatus.FEASIBLE_FOUND) == ACCEPT


def test_status_to_polarity_time_limit():
    assert status_to_polarity(OptimizationStatus.TIME_LIMIT) == DRIFT


def test_status_to_polarity_infeasible():
    assert status_to_polarity(OptimizationStatus.INFEASIBLE) == REFUSE


def test_status_to_polarity_unknown_defaults_to_drift():
    assert status_to_polarity("SomeUnknownStatus") == DRIFT


def test_receipt_build_basic():
    r = OptimizationReceipt.build(
        cell_id="test-cell",
        substrate="cuopt-lp",
        status=OptimizationStatus.OPTIMAL,
        payload={"objective_value": 42.0},
    )
    assert r.polarity == ACCEPT
    assert r.status == OptimizationStatus.OPTIMAL
    assert r.cell_id == "test-cell"
    assert r.substrate == "cuopt-lp"
    assert r.witness_id != ""
    assert r.prev_witness_id == ""


def test_receipt_chain():
    r1 = OptimizationReceipt.build(
        cell_id="c",
        substrate="cuopt-routing",
        status=OptimizationStatus.OPTIMAL,
        payload={"a": 1},
    )
    r2 = OptimizationReceipt.build(
        cell_id="c",
        substrate="cuopt-routing",
        status=OptimizationStatus.OPTIMAL,
        payload={"a": 2},
        prev_witness_id=r1.witness_id,
    )
    assert r2.prev_witness_id == r1.witness_id
    # r2 chains after r1: r2.prev_witness_id == r1.witness_id → r2.chain_with(r1) is True
    # r1 doesn't chain after r2: r1.prev_witness_id (empty) != r2.witness_id → False
    assert r2.chain_with(r1) is True
    assert r1.chain_with(r2) is False


def test_receipt_determinism():
    """Same inputs → same witness_id (sha256-of-canonical)."""
    r1 = OptimizationReceipt.build(
        cell_id="c",
        substrate="cuopt-lp",
        status=OptimizationStatus.OPTIMAL,
        payload={"x": 1},
        timestamp=1000,
    )
    r2 = OptimizationReceipt.build(
        cell_id="c",
        substrate="cuopt-lp",
        status=OptimizationStatus.OPTIMAL,
        payload={"x": 1},
        timestamp=1000,
    )
    assert r1.witness_id == r2.witness_id


def test_receipt_to_dict():
    r = OptimizationReceipt.build(
        cell_id="c",
        substrate="cuopt-lp",
        status=OptimizationStatus.OPTIMAL,
        payload={"x": 1},
    )
    d = r.to_dict()
    assert d["cell_id"] == "c"
    assert d["substrate"] == "cuopt-lp"
    assert d["polarity"] == ACCEPT
    assert "witness_id" in d
    assert "timestamp" in d
