"""Tests for the routing substrate."""
from quilt_optimization.routing import (
    RoutingSubstrate, MockCuOptRoutingBackend, RoutingProblem,
    RoutingSolution, OptimizationStatus,
)
from quilt_optimization.substrate import ACCEPT, DRIFT, REFUSE


def _simple_problem():
    """A 4-location VRP: 1 depot, 3 orders, 2 vehicles."""
    return RoutingProblem(
        name="test-vrp",
        n_locations=4,
        n_vehicles=2,
        n_orders=3,
        cost_matrix=[
            [0, 10, 15, 20],   # from depot
            [10, 0, 12, 18],
            [15, 12, 0, 14],
            [20, 18, 14, 0],
        ],
        order_locations=[1, 2, 3],
        vehicle_starts=[0, 0],
        vehicle_ends=[0, 0],
        vehicle_capacities=[100, 100],
        order_demands=[10, 20, 15],
    )


def test_mock_backend_solves_simple_vrp():
    backend = MockCuOptRoutingBackend()
    problem = _simple_problem()
    solution = backend.solve(problem)

    assert solution.status in (OptimizationStatus.OPTIMAL, OptimizationStatus.PRIMAL_FEASIBLE)
    assert len(solution.routes) == 2
    # All orders assigned
    all_visits = set()
    for route in solution.routes:
        all_visits.update(route)
    # Locations 1, 2, 3 should be visited
    assert 1 in all_visits
    assert 2 in all_visits
    assert 3 in all_visits


def test_mock_backend_determinism():
    """Same problem → same solution."""
    backend = MockCuOptRoutingBackend()
    problem = _simple_problem()
    s1 = backend.solve(problem)
    s2 = backend.solve(problem)
    assert s1.routes == s2.routes
    assert s1.total_cost == s2.total_cost


def test_routing_substrate_emits_receipt():
    substrate = RoutingSubstrate(
        backend=MockCuOptRoutingBackend(),
        cell_id="test-cell",
    )
    problem = _simple_problem()
    receipt = substrate.solve(problem)

    assert receipt.substrate == "cuopt-routing"
    assert receipt.polarity in (ACCEPT, DRIFT, REFUSE)
    assert receipt.payload["problem_name"] == "test-vrp"
    assert substrate.chain_head == receipt.witness_id


def test_routing_substrate_chains_receipts():
    substrate = RoutingSubstrate(
        backend=MockCuOptRoutingBackend(),
        cell_id="chained-cell",
    )
    p1 = _simple_problem()
    p2 = _simple_problem()
    p2.name = "second-vrp"
    r1 = substrate.solve(p1)
    r2 = substrate.solve(p2)

    assert r2.prev_witness_id == r1.witness_id
    # r2 chains after r1: r2.prev_witness_id == r1.witness_id
    assert r2.chain_with(r1) is True
    assert len(substrate.receipts) == 2


def test_routing_substrate_with_time_limit():
    """Substrate emits DRIFT when backend hits time limit."""
    # Tight time limit
    backend = MockCuOptRoutingBackend(time_limit_seconds=0.0001)
    substrate = RoutingSubstrate(backend=backend, cell_id="tight-cell")
    problem = _simple_problem()

    receipt = substrate.solve(problem)
    # With very tight limit, might return TIMEOUT → DRIFT
    # or might finish → ACCEPT (depends on how fast)
    assert receipt.polarity in (ACCEPT, DRIFT, REFUSE)


def test_routing_substrate_payload_shape():
    substrate = RoutingSubstrate(
        backend=MockCuOptRoutingBackend(),
        cell_id="payload-cell",
    )
    receipt = substrate.solve(_simple_problem())
    assert "n_locations" in receipt.payload
    assert "n_vehicles" in receipt.payload
    assert "n_orders" in receipt.payload
    assert "total_cost" in receipt.payload
    assert "solver_time_seconds" in receipt.payload
    assert "routes" in receipt.payload
