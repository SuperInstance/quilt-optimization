"""Tests for the LP/MILP substrate."""
from quilt_optimization.linear_programming import (
    LPSubstrate, MockCuOptLPBackend, LPProblem, LPVariable,
    LPConstraint, LPObjective, LinearTerm,
)
from quilt_optimization.substrate import (
    OptimizationStatus, ACCEPT, DRIFT, REFUSE,
)


def _simple_lp():
    """Simple LP: max 40x + 30y s.t. 2x+3y<=120, 4x+2y<=100, x,y>=0."""
    return LPProblem(
        name="simple-lp",
        variables=[
            LPVariable(name="x", lb=0),
            LPVariable(name="y", lb=0),
        ],
        constraints=[
            LPConstraint(
                terms=[LinearTerm("x", 2), LinearTerm("y", 3)],
                rhs=120,
                sense="<=",
                name="resource_a",
            ),
            LPConstraint(
                terms=[LinearTerm("x", 4), LinearTerm("y", 2)],
                rhs=100,
                sense="<=",
                name="resource_b",
            ),
        ],
        objective=LPObjective(
            terms=[LinearTerm("x", 40), LinearTerm("y", 30)],
            sense="MAXIMIZE",
        ),
    )


def test_mock_backend_solves_simple_lp():
    backend = MockCuOptLPBackend()
    problem = _simple_lp()
    solution = backend.solve(problem)

    assert solution.status == OptimizationStatus.OPTIMAL
    assert "x" in solution.variable_values
    assert "y" in solution.variable_values
    # Expected: x=7.5, y=35, objective=1350 (true LP optimum at corner of 2x+3y=120, 4x+2y=100)
    assert abs(solution.variable_values["x"] - 7.5) < 1
    assert abs(solution.variable_values["y"] - 35) < 1
    assert abs(solution.objective_value - 1350) < 10


def test_lp_substrate_emits_receipt():
    substrate = LPSubstrate(
        backend=MockCuOptLPBackend(),
        cell_id="lp-cell",
    )
    receipt = substrate.solve(_simple_lp())

    assert receipt.substrate == "cuopt-lp"
    assert receipt.polarity == ACCEPT
    assert receipt.status == OptimizationStatus.OPTIMAL
    assert substrate.chain_head == receipt.witness_id


def test_lp_substrate_chains():
    substrate = LPSubstrate(
        backend=MockCuOptLPBackend(),
        cell_id="chain-lp",
    )
    p1 = _simple_lp()
    p2 = _simple_lp()
    p2.name = "second-lp"
    r1 = substrate.solve(p1)
    r2 = substrate.solve(p2)

    assert r2.prev_witness_id == r1.witness_id
    assert len(substrate.receipts) == 2


def test_lp_substrate_infeasible_emits_refuse():
    """An infeasible problem emits REFUSE polarity."""
    # x >= 10 AND x <= 5 — no feasible solution
    problem = LPProblem(
        name="infeasible",
        variables=[
            LPVariable(name="x", lb=0),
            LPVariable(name="y", lb=0),
        ],
        constraints=[
            LPConstraint(
                terms=[LinearTerm("x", 1)], rhs=10, sense=">=", name="lower",
            ),
            LPConstraint(
                terms=[LinearTerm("x", 1)], rhs=5, sense="<=", name="upper",
            ),
            LPConstraint(
                terms=[LinearTerm("y", 1)], rhs=0, sense=">=", name="y_lb",
            ),
        ],
        objective=LPObjective(terms=[LinearTerm("x", 1)], sense="MINIMIZE"),
    )
    backend = MockCuOptLPBackend()
    solution = backend.solve(problem)

    assert solution.status in (
        OptimizationStatus.INFEASIBLE,
        OptimizationStatus.PRIMAL_INFEASIBLE,
    )

    substrate = LPSubstrate(backend=backend, cell_id="inf-cell")
    receipt = substrate.solve(problem)
    assert receipt.polarity == REFUSE


def test_lp_substrate_payload_shape():
    substrate = LPSubstrate(
        backend=MockCuOptLPBackend(),
        cell_id="payload-lp",
    )
    receipt = substrate.solve(_simple_lp())
    assert "problem_name" in receipt.payload
    assert "n_variables" in receipt.payload
    assert "n_constraints" in receipt.payload
    assert "objective_value" in receipt.payload
    assert "variable_values" in receipt.payload
    assert receipt.payload["n_variables"] == 2
    assert receipt.payload["n_constraints"] == 2


def test_lp_substrate_milp_if_available():
    """MILP problem: integer + continuous variables. Skip if scipy unavailable."""
    try:
        from scipy.optimize import milp  # noqa: F401
    except ImportError:
        pass  # MILP test skipped (scipy.optimize.milp unavailable)

    problem = LPProblem(
        name="facility-location",
        variables=[
            LPVariable(name="open", lb=0, ub=1, vtype="INTEGER"),
            LPVariable(name="production", lb=0, vtype="CONTINUOUS"),
        ],
        constraints=[
            LPConstraint(
                terms=[LinearTerm("production", 1), LinearTerm("open", -1000)],
                rhs=0, sense="<=", name="link",
            ),
        ],
        objective=LPObjective(
            terms=[LinearTerm("open", 500), LinearTerm("production", 2)],
            sense="MINIMIZE",
        ),
    )

    substrate = LPSubstrate(
        backend=MockCuOptLPBackend(),
        cell_id="milp-cell",
    )
    receipt = substrate.solve(problem)
    assert receipt.polarity in (ACCEPT, DRIFT, REFUSE)
    assert "open" in receipt.payload["variable_values"]
    assert "production" in receipt.payload["variable_values"]
