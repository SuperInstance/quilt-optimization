# 🌱 quilt-optimization

> NVIDIA cuOpt as a Quilt substrate — vehicle routing, linear programming, mixed integer programming, quadratic programming.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-21/21-brightgreen.svg)](tests/)
[![Substrates](https://img.shields.io/badge/substrates-2_(routing%2C_lp)-purple.svg)](src/quilt_optimization/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![cuOpt](https://img.shields.io/badge/cuOpt-26.10-76b900.svg)](https://github.com/NVIDIA/cuopt)

## What is this?

`quilt-optimization` wraps [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt) (GPU-accelerated optimization) as a [Quilt](https://github.com/SuperInstance/superinstance) substrate. The substrate walker pattern (Mavis) composes it the same way as Vibe, Collective Unconscious, and other substrates: every solve emits a `CellReceipt` with `polarity ∈ {ACCEPT, DRIFT, REFUSE}`, and the receipt chains via `prev_witness_id`.

Two problem kinds:

- **Routing** — Vehicle routing problems (VRP), TSP, Pickup-Delivery (PDP)
- **Linear programming** — LP, MILP (mixed integer), QP (quadratic, MINIMIZE only)

Two backends per problem kind:

- **`Mock*Backend`** — deterministic, offline, no GPU. Tests + small demos.
- **`CuOpt*Backend`** — real cuOpt via `import cuopt`. Requires NVIDIA GPU + `pip install cuopt-server-cu12`.

## Why?

cuOpt is GPU-accelerated. It solves LP/MILP problems with millions of variables and constraints in seconds, where CPU solvers take minutes. Quilt is built around cells that decide — and many decisions are *optimization decisions*. Which routes do my agents take? What's the optimal production mix? Should I open this facility?

The substrate walker pattern lets any cell compose any solver, with the same witness chain, the same polarity vocabulary, and the same legalese layer. This is the pattern repeated: ~340 LOC wrapper + ~150 LOC tests + a demo.

## Install

```bash
# Mock backends only (no GPU required) — for tests, demos, learning
pip install quilt-optimization

# Or with real cuOpt (requires NVIDIA GPU)
pip install --extra-index-url=https://pypi.nvidia.com \
    nvidia-cuda-runtime-cu12==12.9.* \
    cuopt-server-cu12==26.10.* cuopt-sh-client==26.10.*
pip install quilt-optimization
```

## Quickstart

```python
from quilt_optimization import (
    RoutingSubstrate, RoutingProblem,
    LPSubstrate, LPProblem, LPVariable, LPConstraint, LPObjective, LinearTerm,
)

# Vehicle Routing Problem
vrp = RoutingProblem(
    name="warehouse-delivery",
    n_locations=4,
    n_vehicles=2,
    n_orders=3,
    cost_matrix=[
        [0,  10, 15, 20],
        [10, 0,  12, 18],
        [15, 12, 0,  14],
        [20, 18, 14, 0],
    ],
    order_locations=[1, 2, 3],
    vehicle_starts=[0, 0],
    vehicle_ends=[0, 0],
    vehicle_capacities=[100, 100],
    order_demands=[10, 20, 15],
)
routing = RoutingSubstrate()
receipt = routing.solve(vrp)
print(receipt.polarity, receipt.payload["total_cost"])

# Linear Programming
lp = LPProblem(
    name="production-mix",
    variables=[LPVariable(name="x", lb=0), LPVariable(name="y", lb=0)],
    constraints=[
        LPConstraint(terms=[LinearTerm("x", 2), LinearTerm("y", 3)], rhs=120, sense="<=", name="machine_a"),
        LPConstraint(terms=[LinearTerm("x", 4), LinearTerm("y", 2)], rhs=100, sense="<=", name="machine_b"),
    ],
    objective=LPObjective(terms=[LinearTerm("x", 40), LinearTerm("y", 30)], sense="MAXIMIZE"),
)
lp_substrate = LPSubstrate(prev_witness_id=routing.chain_head)
receipt = lp_substrate.solve(lp)
print(receipt.polarity, receipt.payload["objective_value"])
```

## Substrate walker pattern

The substrate walker composes optimization substrates just like Vibe and CU:

```
┌─────────────────────────────────────────────────────────────┐
│  Substrate Walker (Mavis)                                   │
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │ RoutingSub.  │    │  LPSubstrate │    │  Holodeck    │  │
│   │  (VRP/TSP)   │───▶│  (LP/MILP)   │───▶│  (Lens comp.) │ │
│   └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                     │                    │         │
│         ▼                     ▼                    ▼         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           OptimizationReceipt (canonical)           │   │
│   │  { witness_id, prev_witness_id, polarity, payload } │   │
│   └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │       LegaleseNetwork (quilt-seed) — Claims          │   │
│   │       Vessel (quilt-seed) — decides                  │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

Three properties close the substrate walker pattern:

1. **Same envelope** — every solve emits an `OptimizationReceipt` with the same shape.
2. **Same chain** — `prev_witness_id` links consecutive solves (across substrates).
3. **Same polarity** — `ACCEPT | DRIFT | REFUSE` maps from cuOpt's status vocabulary.

## Status → polarity mapping

| cuOpt status        | Quilt polarity | Meaning                                  |
|---------------------|----------------|------------------------------------------|
| `Optimal`           | **ACCEPT**     | Proven optimal solution found            |
| `PrimalFeasible`    | **ACCEPT**     | Feasible but not proven optimal (LP/QP)  |
| `FeasibleFound`     | **ACCEPT**     | High-quality feasible within gap (MILP)  |
| `TimeLimit`         | **DRIFT**      | Hit time limit; may be suboptimal        |
| `IterationLimit`    | **DRIFT**      | Hit iteration limit                      |
| `NoTermination`     | **DRIFT**      | Solver stopped without conclusion        |
| `Infeasible`        | **REFUSE**     | No feasible solution exists              |
| `Unbounded`         | **REFUSE**     | Objective unbounded                      |
| `NumericalError`    | **REFUSE**     | Numerical instability                    |
| `PrimalInfeasible`  | **REFUSE**     | LP/QP: primal infeasible                 |
| `DualInfeasible`    | **REFUSE**     | LP/QP: dual infeasible (unbounded)       |
| `FAIL`              | **REFUSE**     | Routing: solver failed                   |
| `EMPTY`             | **REFUSE**     | Routing: no solution found               |

Unknown statuses default to **DRIFT** (the substrate's view of the world is uncertain).

## Examples

The `examples/` directory:

- **`demo.py`** — 5-act walkthrough: VRP → LP → MILP, then chain verification + polarity summary. No GPU required.
- **`with_real_cuopt.py`** — same flow using `CuOptRoutingBackend` / `CuOptLPBackend` (requires GPU + cuopt pip install).

Run: `PYTHONPATH=src python3 examples/demo.py`

## Doctrines

1. **The substrate walker pattern is closed.** Each new substrate = ~340 LOC wrapper + ~150 LOC tests + ~50 LOC demo. The envelope is constant.
2. **Status vocabulary is the polarity.** cuOpt's `Optimal` ⇔ ACCEPT; `Infeasible` ⇔ REFUSE; `TimeLimit` ⇔ DRIFT. The mapping is single-source-of-truth.
3. **Mock backends are the truth for tests.** Deterministic, offline, no GPU. Tests are reproducible across sandboxes.
4. **Real backends are for production.** cuOpt requires NVIDIA GPU + cuOpt pip install. The substrate walker composes both — production swap is one line.
5. **The receipt chains, not the substrate.** Each substrate emits a receipt; the chain is across substrates. cuOpt-routing → cuOpt-lp → cuOpt-qp is the same chain as Jepa → JEV → MOTH.
6. **The cell decides, the solver obeys.** Quilt's cell model: the cell asks a substrate for an answer, the substrate emits a receipt, the cell decides whether to accept the answer (ACCEPT), hold it tentatively (DRIFT), or refuse it (REFUSE).

## Cross-project doctrine

The substrate walker pattern applies to any system where:

- A bounded component accumulates external computations (solvers, models, APIs)
- Each computation result is reduced to a small envelope (status, value, timestamp)
- The envelope's status has a polarity (ACCEPT/DRIFT/REFUSE)
- The components chain their envelopes into a witness log
- A higher-level decision-maker (vessel, cell, agent) consumes the witness log

Maps to: any AI agent that calls multiple external APIs, any CI/CD pipeline that runs multiple tools, any analytics platform that runs multiple models.

## Related

- [quilt-seed](https://github.com/SuperInstance/quilt-seed) — vessel, legalese, stations (where the receipts go to be witnessed)
- [quilt-spreadsheet-inference](https://github.com/SuperInstance/quilt-spreadsheet-inference) — Holodeck (the substrate walker composes lens substrates)
- [quilt-cli](https://github.com/SuperInstance/quilt-cli) — CLI for substrate composition
- [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt) — the solver we wrap

## License

Apache-2.0
