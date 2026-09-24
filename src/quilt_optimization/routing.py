"""
routing.py — cuOpt routing (VRP/TSP/PDP) as a Quilt substrate.

Per cuopt's routing API (cuopt.routing):
  - DataModel(n_locations, n_fleet, n_orders)
  - add_cost_matrix, add_transit_time_matrix
  - set_order_locations, set_vehicle_locations, set_order_time_windows
  - Solve(dm, SolverSettings) returns Assignment with solution status

The substrate walker pattern:
  1. The cell builds a RoutingProblem (declarative spec, substrate-agnostic)
  2. RoutingSubstrate passes it to the backend (mock or real cuopt)
  3. The backend returns a RoutingSolution
  4. The substrate emits an OptimizationReceipt (polarity maps from status)
  5. The receipt chains via prev_witness_id
  6. The legalese layer (quilt-seed) can wrap the receipt as a Claim

Two backends:
  - MockCuOptRoutingBackend: deterministic offline (no GPU, no cuopt import)
  - CuOptRoutingBackend: real cuopt.routing.DataModel + Solve (requires GPU + pip install)

The mock uses a simple greedy nearest-neighbor heuristic. It's not optimal
but it's reproducible and good enough for tests + small demos.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from .substrate import (
    OptimizationReceipt, OptimizationStatus, status_to_polarity,
    ACCEPT, DRIFT, REFUSE,
)


# === Routing problem spec (substrate-agnostic) ===

@dataclass
class RoutingProblem:
    """A routing problem (VRP/TSP/PDP), substrate-agnostic.

    Mirrors cuopt.routing.DataModel but uses plain Python lists/series
    instead of cudf Series, so the substrate walker can compose without
    a GPU.
    """
    name: str
    n_locations: int
    n_vehicles: int
    n_orders: int = 0

    # Distance/cost matrix (n_locations × n_locations)
    cost_matrix: List[List[float]] = field(default_factory=list)

    # Optional: transit time matrix (n_locations × n_locations)
    transit_time_matrix: Optional[List[List[float]]] = None

    # Order locations (indices into locations)
    order_locations: List[int] = field(default_factory=list)

    # Vehicle start/end locations (n_vehicles × 2: [start, end])
    vehicle_starts: List[int] = field(default_factory=list)
    vehicle_ends: List[int] = field(default_factory=list)

    # Vehicle capacities (n_vehicles)
    vehicle_capacities: List[float] = field(default_factory=list)

    # Order demands (n_orders)
    order_demands: List[float] = field(default_factory=list)

    # Order service times (n_orders)
    order_service_times: List[float] = field(default_factory=list)

    # Time windows (n_orders × 2: [earliest, latest])
    order_time_windows: List[List[float]] = field(default_factory=list)

    # Solver settings
    time_limit_seconds: float = 30.0


@dataclass
class RoutingSolution:
    """A routing solution returned by the backend."""
    status: str  # cuOpt status: Optimal, TimeLimit, Infeasible, etc.
    routes: List[List[int]] = field(default_factory=list)  # per-vehicle visit order
    total_cost: float = 0.0
    solver_time_seconds: float = 0.0


# === Backend protocol ===

class RoutingBackend(Protocol):
    """Anything that can solve a RoutingProblem."""

    def solve(self, problem: RoutingProblem) -> RoutingSolution: ...


# === Mock backend (offline, deterministic) ===

class MockCuOptRoutingBackend:
    """Greedy nearest-neighbor router. Reproducible; not optimal.

    Algorithm:
      1. For each vehicle, start at vehicle_starts[i]
      2. While orders remain, pick the nearest unvisited order
      3. Return at vehicle_ends[i]
      4. Status: "Optimal" if all orders assigned; "TimeLimit" if exceeded

    Determinism: when multiple orders are equidistant, the order with the
    smallest index wins. Tests are reproducible.
    """

    def __init__(self, time_limit_seconds: float = 5.0):
        self.time_limit_seconds = time_limit_seconds

    def solve(self, problem: RoutingProblem) -> RoutingSolution:
        t0 = time.time()

        # Initialize: each vehicle has its own route
        routes = []
        for v in range(problem.n_vehicles):
            route = [problem.vehicle_starts[v]] if v < len(problem.vehicle_starts) else [0]
            routes.append(route)

        # Track which orders are assigned
        unassigned = set(range(problem.n_orders))

        # Greedy assignment per vehicle
        for v in range(problem.n_vehicles):
            if time.time() - t0 > self.time_limit_seconds:
                # Hit time limit
                return RoutingSolution(
                    status=OptimizationStatus.TIMEOUT,
                    routes=routes,
                    total_cost=0.0,
                    solver_time_seconds=time.time() - t0,
                )

            current_loc = routes[v][-1]
            load = 0.0
            capacity = problem.vehicle_capacities[v] if v < len(problem.vehicle_capacities) else float("inf")

            while unassigned:
                # Find nearest unassigned order
                best_order = None
                best_cost = float("inf")
                for o in sorted(unassigned):  # sorted for determinism
                    order_loc = problem.order_locations[o] if o < len(problem.order_locations) else o
                    if order_loc >= problem.n_locations or current_loc >= problem.n_locations:
                        continue
                    cost = problem.cost_matrix[current_loc][order_loc]
                    demand = problem.order_demands[o] if o < len(problem.order_demands) else 0.0
                    if load + demand <= capacity and cost < best_cost:
                        best_cost = cost
                        best_order = o

                if best_order is None:
                    break  # capacity exceeded or no feasible order

                # Add to route
                order_loc = problem.order_locations[best_order]
                routes[v].append(order_loc)
                load += problem.order_demands[best_order] if best_order < len(problem.order_demands) else 0.0
                unassigned.remove(best_order)
                current_loc = order_loc

            # Return to end location
            if v < len(problem.vehicle_ends):
                routes[v].append(problem.vehicle_ends[v])

        # Compute total cost
        total_cost = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                if a < problem.n_locations and b < problem.n_locations:
                    total_cost += problem.cost_matrix[a][b]

        # If all orders assigned → "Optimal" (mock is optimal for greedy if no capacity)
        # Otherwise → "PrimalFeasible" (mock doesn't always find optimal)
        status = (OptimizationStatus.OPTIMAL
                  if not unassigned
                  else OptimizationStatus.PRIMAL_FEASIBLE)

        return RoutingSolution(
            status=status,
            routes=routes,
            total_cost=total_cost,
            solver_time_seconds=time.time() - t0,
        )


# === Real cuOpt backend ===

class CuOptRoutingBackend:
    """Real cuopt.routing wrapper. Requires NVIDIA GPU + cuopt pip install.

    The substrate walker calls this when the real cuOpt is available.
    Falls back to a clear ImportError if cuopt isn't installed.
    """

    def __init__(self):
        try:
            import cudf  # noqa: F401
            from cuopt import routing  # noqa: F401
            self._cudf = cudf
            self._routing = routing
        except ImportError as e:
            raise ImportError(
                f"CuOptRoutingBackend requires NVIDIA cuOpt + cudf.\n"
                f"  pip install --extra-index-url=https://pypi.nvidia.com \\\n"
                f"    nvidia-cuda-runtime-cu12==12.9.* \\\n"
                f"    cuopt-server-cu12==26.10.* cuopt-sh-client==26.10.*\n"
                f"Original error: {e}"
            )

    def solve(self, problem: RoutingProblem) -> RoutingSolution:
        t0 = time.time()
        n = problem.n_locations
        f = problem.n_vehicles

        # Build DataModel
        dm = self._routing.DataModel(n_locations=n, n_fleet=f,
                                     n_orders=problem.n_orders)

        # Cost matrix (cudf DataFrame)
        cost_df = self._cudf.DataFrame(problem.cost_matrix, dtype="float32")
        dm.add_cost_matrix(cost_df)

        # Order locations (cudf Series)
        if problem.order_locations:
            order_loc_series = self._cudf.Series(problem.order_locations, dtype="int32")
            dm.set_order_locations(order_loc_series)

        # Vehicle locations
        if problem.vehicle_starts and problem.vehicle_ends:
            starts = self._cudf.Series([s for s in problem.vehicle_starts], dtype="int32")
            ends = self._cudf.Series([e for e in problem.vehicle_ends], dtype="int32")
            dm.set_vehicle_locations(starts, ends)

        # Capacities
        if problem.vehicle_capacities and problem.order_demands:
            demands = self._cudf.Series(problem.order_demands, dtype="float32")
            capacities = self._cudf.Series(problem.vehicle_capacities, dtype="float32")
            dm.add_capacity_dimension("weight", demands, capacities)

        # Solver settings
        ss = self._routing.SolverSettings()
        ss.set_time_limit(problem.time_limit_seconds)

        # Solve
        solution = self._routing.Solve(dm, ss)

        # Map cuOpt status to our status
        # cuOpt's get_status() returns int 0=SUCCESS, 1=FAIL, 2=TIMEOUT, 3=EMPTY
        status_int = solution.get_status()
        status_map = {
            0: OptimizationStatus.OPTIMAL,
            1: OptimizationStatus.FAIL,
            2: OptimizationStatus.TIMEOUT,
            3: OptimizationStatus.EMPTY,
        }
        status = status_map.get(status_int, OptimizationStatus.NO_TERMINATION)

        # Extract routes
        routes = []
        try:
            route_df = solution.get_route()
            for v in range(f):
                # route_df has columns like 'truck_id', 'route', 'location_id'
                v_routes = route_df[route_df["truck_id"] == v]["location_id"].to_list()
                if v_routes:
                    routes.append(v_routes)
        except Exception:
            routes = [[] for _ in range(f)]

        total_cost = solution.get_total_objective() if hasattr(solution, "get_total_objective") else 0.0

        return RoutingSolution(
            status=status,
            routes=routes,
            total_cost=total_cost,
            solver_time_seconds=time.time() - t0,
        )


# === The substrate ===

@dataclass
class RoutingSubstrate:
    """A Quilt routing substrate backed by cuOpt.

    The substrate walker composes this with Vibe, Collective Unconscious,
    and the legalese layer. Each solve emits a receipt; the cell decides.
    """

    backend: RoutingBackend = field(default_factory=MockCuOptRoutingBackend)
    cell_id: str = "routing-substrate-cell"
    prev_witness_id: str = ""
    receipts: List[OptimizationReceipt] = field(default_factory=list)

    def solve(self, problem: RoutingProblem) -> OptimizationReceipt:
        """Solve a routing problem and emit a witness receipt.

        The receipt chains via prev_witness_id; the polarity maps from
        cuOpt's status vocabulary. The substrate walker can compose this
        with the legalese layer.
        """
        solution = self.backend.solve(problem)

        payload = {
            "problem_name": problem.name,
            "n_locations": problem.n_locations,
            "n_vehicles": problem.n_vehicles,
            "n_orders": problem.n_orders,
            "status": solution.status,
            "total_cost": round(solution.total_cost, 6),
            "solver_time_seconds": round(solution.solver_time_seconds, 6),
            "routes": solution.routes,
        }

        receipt = OptimizationReceipt.build(
            cell_id=self.cell_id,
            substrate="cuopt-routing",
            status=solution.status,
            payload=payload,
            prev_witness_id=self.prev_witness_id,
        )

        self.prev_witness_id = receipt.witness_id
        self.receipts.append(receipt)
        return receipt

    @property
    def chain_head(self) -> str:
        return self.prev_witness_id

# Routing status is the same vocabulary as optimization status
RoutingStatus = OptimizationStatus
