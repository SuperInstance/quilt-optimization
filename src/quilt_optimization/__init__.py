"""
quilt-optimization — NVIDIA cuOpt as a Quilt substrate.

This package wraps NVIDIA's cuOpt solver (LP, MILP, QP, routing/VRP/TSP/PDP)
as a Quilt substrate for the substrate walker (Mavis) pattern.

The substrate walker composes:
  - cuOpt routing (DataModel + Solve) → Quilt routing substrate
  - cuOpt LP/MILP (Problem + SolverSettings) → Quilt LP substrate
  - Each solve emits a CellReceipt (ACCEPT/DRIFT/REFUSE polarity, prev_witness_id chain)
  - The legalese layer (quilt-seed) witnesses the receipts
  - The vessel decides what to do with the solution

Two backends per substrate:
  - MockCuOptBackend: deterministic offline (no GPU required, for tests)
  - CuOptBackend: real cuOpt via import (requires NVIDIA GPU + cuopt pip install)

Three problem kinds:
  - routing: VRP/TSP/PDP via cuopt.routing
  - linear_programming: LP via cuopt.linear_programming
  - mixed_integer: MILP via cuopt.linear_programming (with INTEGER variables)

Status vocabulary (mirrors cuOpt's PascalCase):
  - "Optimal"     → ACCEPT
  - "FeasibleFound" → ACCEPT (MILP only — within gap tolerance)
  - "PrimalFeasible" → ACCEPT (LP/QP only)
  - "TimeLimit"   → DRIFT (hit time limit, may be suboptimal)
  - "IterationLimit" → DRIFT
  - "Infeasible"  → REFUSE
  - "Unbounded"   → REFUSE
  - "NumericalError" → REFUSE
  - "NoTermination" → DRIFT

Why cuOpt:
  - GPU-accelerated — near real-time solves for large problems
  - Open-source (Apache-2.0) — fits the SuperInstance doctrine
  - Already ships AI agent skills + MCP server — composable with our agents
  - LP/MILP/QP for resource allocation, routing for vehicle/agent dispatch
  - The substrate walker composes it; the legalese records it; the vessel decides.

This is the substrate walker pattern repeated:
  199 LOC wrapper + 130 LOC tests + 60 LOC demo = ~400 LOC per problem kind.
"""

from .routing import (
    RoutingSubstrate, MockCuOptRoutingBackend, CuOptRoutingBackend,
    RoutingProblem, RoutingSolution, RoutingStatus,
)
from .linear_programming import (
    LPSubstrate, MockCuOptLPBackend, CuOptLPBackend,
    LPProblem, LPSolution, LPStatus,
    LPVariable, LPConstraint, LPObjective, LinearTerm,
)
from .substrate import (
    OptimizationStatus, OptimizationReceipt,
    ACCEPT, DRIFT, REFUSE,
    status_to_polarity,
)

__all__ = [
    # Routing (VRP/TSP/PDP)
    "RoutingSubstrate", "MockCuOptRoutingBackend", "CuOptRoutingBackend",
    "RoutingProblem", "RoutingSolution", "RoutingStatus",
    # Linear programming (LP/MILP/QP)
    "LPSubstrate", "MockCuOptLPBackend", "CuOptLPBackend",
    "LPProblem", "LPSolution", "LPStatus",
    "LPVariable", "LPConstraint", "LPObjective", "LinearTerm",
    # Common
    "OptimizationStatus", "OptimizationReceipt",
    "ACCEPT", "DRIFT", "REFUSE", "status_to_polarity",
]

__version__ = "0.1.0"
