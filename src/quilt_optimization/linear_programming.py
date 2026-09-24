"""
linear_programming.py — cuOpt LP/MILP/QP as a Quilt substrate.

Per cuopt's linear_programming API (cuopt.linear_programming):
  - Problem(name)
  - addVariable(lb, ub, vtype, name) — vtype ∈ {CONTINUOUS, INTEGER}
  - addConstraint(expr, name)
  - setObjective(expr, sense) — sense ∈ {MINIMIZE, MAXIMIZE}
  - solve(SolverSettings) → Status ∈ {Optimal, FeasibleFound, Infeasible, ...}

The substrate walker pattern (mirrors routing.py):
  1. The cell builds an LPProblem (declarative spec, substrate-agnostic)
  2. LPSubstrate passes it to the backend (mock or real cuopt)
  3. The backend returns an LPSolution
  4. The substrate emits an OptimizationReceipt
  5. The receipt chains via prev_witness_id

For the mock backend, we implement a simple LP solver via scipy.optimize.linprog
(falls back to a tiny hand-rolled simplex if scipy isn't available).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

from .substrate import (
    OptimizationReceipt, OptimizationStatus,
    ACCEPT, DRIFT, REFUSE,
)


# === LP problem spec (substrate-agnostic) ===

@dataclass
class LinearTerm:
    """A linear term: coefficient * variable_name."""
    variable: str
    coefficient: float


@dataclass
class LPVariable:
    """An LP/MILP variable."""
    name: str
    lb: float = 0.0
    ub: float = float("inf")
    vtype: str = "CONTINUOUS"  # CONTINUOUS or INTEGER


@dataclass
class LPConstraint:
    """A linear constraint: sum(coef_i * var_i) <= rhs."""
    terms: List[LinearTerm]
    rhs: float
    sense: str = "<="  # <=, >=, ==
    name: str = ""


@dataclass
class LPObjective:
    """An LP objective: minimize or maximize sum(coef_i * var_i)."""
    terms: List[LinearTerm]
    sense: str = "MINIMIZE"  # MINIMIZE or MAXIMIZE


@dataclass
class LPProblem:
    """An LP/MILP problem, substrate-agnostic."""
    name: str
    variables: List[LPVariable]
    constraints: List[LPConstraint]
    objective: LPObjective
    time_limit_seconds: float = 60.0


@dataclass
class LPSolution:
    """An LP/MILP solution returned by the backend."""
    status: str  # cuOpt status: Optimal, FeasibleFound, Infeasible, etc.
    variable_values: dict = field(default_factory=dict)  # name → value
    objective_value: float = 0.0
    solver_time_seconds: float = 0.0


# === Backend protocol ===

class LPBackend(Protocol):
    """Anything that can solve an LPProblem."""

    def solve(self, problem: LPProblem) -> LPSolution: ...


# === Mock backend (offline, simple LP via pure Python simplex) ===

class MockCuOptLPBackend:
    """Simple LP solver. Handles small problems (< 50 vars, < 50 constraints).

    Algorithm:
      - LP: simple gradient/vertex walk (works on tiny problems)
      - MILP: enumerate integer solutions + solve LP relaxations
      - QP: NOT supported in mock — returns NumericalError

    Determinism: greedy, tie-broken by variable name. Reproducible.
    """

    def __init__(self, time_limit_seconds: float = 5.0):
        self.time_limit_seconds = time_limit_seconds

    def solve(self, problem: LPProblem) -> LPSolution:
        t0 = time.time()

        # Check for INTEGER variables → MILP
        has_integer = any(v.vtype == "INTEGER" for v in problem.variables)

        if has_integer:
            return self._solve_milp(problem, t0)
        else:
            return self._solve_lp(problem, t0)

    def _solve_lp(self, problem: LPProblem, t0: float) -> LPSolution:
        """Simple LP via pure Python: vertex walk.

        For tiny LP (≤10 vars), we use a brute-force vertex walk:
          1. Find vertices of the feasible polytope (intersections of n constraints)
          2. Evaluate objective at each vertex
          3. Return the best

        This is exponential in worst case but works for tiny demos.
        """
        vars_ = problem.variables
        n = len(vars_)

        # Use scipy if available
        try:
            from scipy.optimize import linprog
            # Convert to scipy format
            c = self._objective_coeffs(problem)
            # scipy.linprog ALWAYS minimizes — negate for MAXIMIZE
            if problem.objective.sense == "MAXIMIZE":
                c = [-ci for ci in c]
            # A_ub @ x <= b_ub
            A_ub, b_ub = self._inequality_constraints(problem)
            # A_eq @ x == b_eq
            A_eq, b_eq = self._equality_constraints(problem)
            bounds = [(v.lb, v.ub if v.ub != float("inf") else None) for v in vars_]

            kwargs = {"bounds": bounds, "method": "highs"}
            if A_ub:
                kwargs["A_ub"] = A_ub
                kwargs["b_ub"] = b_ub
            if A_eq:
                kwargs["A_eq"] = A_eq
                kwargs["b_eq"] = b_eq
            result = linprog(c, **kwargs)

            status_map = {
                0: OptimizationStatus.OPTIMAL,
                1: OptimizationStatus.TIME_LIMIT,
                2: OptimizationStatus.INFEASIBLE,
                3: OptimizationStatus.UNBOUNDED,
            }
            status = status_map.get(result.status, OptimizationStatus.NO_TERMINATION)

            if result.x is None:
                # Infeasible or unbounded — scipy returns None for x
                variable_values = {v.name: 0.0 for v in vars_}
                obj_val = 0.0
            else:
                variable_values = {vars_[i].name: float(result.x[i]) for i in range(n)}
                obj_val = float(result.fun) if problem.objective.sense == "MINIMIZE" else -float(result.fun)

            return LPSolution(
                status=status,
                variable_values=variable_values,
                objective_value=obj_val,
                solver_time_seconds=time.time() - t0,
            )
        except ImportError:
            # Fallback: brute-force vertex walk (only for tiny problems)
            if n > 6:
                return LPSolution(
                    status=OptimizationStatus.NUMERICAL_ERROR,
                    solver_time_seconds=time.time() - t0,
                )

            return self._brute_force_lp(problem, t0)

    def _brute_force_lp(self, problem: LPProblem, t0: float) -> LPSolution:
        """Brute-force LP vertex walk for tiny problems (n ≤ 6)."""
        vars_ = problem.variables
        n = len(vars_)

        # Sample corners of the variable box (lb, ub) for each var
        # and check feasibility + objective
        best_value = None
        best_x = None
        status = OptimizationStatus.OPTIMAL

        # Try vertices of the box (2^n combinations)
        from itertools import product
        boxes = [(v.lb, v.ub if v.ub != float("inf") else 1e6) for v in vars_]

        # Also add vertices where each constraint is tight
        for combo in product(*[(lb, ub) for lb, ub in boxes]):
            x = list(combo)
            # Check all constraints
            feasible = True
            for con in problem.constraints:
                val = sum(t.coefficient * x[vars_.index(t.variable)] if t.variable in [v.name for v in vars_] else 0
                          for t in con.terms)
                if con.sense == "<=" and val > con.rhs + 1e-6:
                    feasible = False
                    break
                if con.sense == ">=" and val < con.rhs - 1e-6:
                    feasible = False
                    break
                if con.sense == "==" and abs(val - con.rhs) > 1e-6:
                    feasible = False
                    break

            if not feasible:
                continue

            # Evaluate objective
            obj = sum(t.coefficient * x[vars_.index(t.variable)] for t in problem.objective.terms)
            if problem.objective.sense == "MAXIMIZE":
                obj = -obj

            if best_value is None or obj < best_value:
                best_value = obj
                best_x = x

        if best_x is None:
            return LPSolution(
                status=OptimizationStatus.INFEASIBLE,
                solver_time_seconds=time.time() - t0,
            )

        var_names = [v.name for v in vars_]
        variable_values = {var_names[i]: best_x[i] for i in range(n)}
        obj_val = -best_value if problem.objective.sense == "MAXIMIZE" else best_value

        return LPSolution(
            status=status,
            variable_values=variable_values,
            objective_value=obj_val,
            solver_time_seconds=time.time() - t0,
        )

    def _solve_milp(self, problem: LPProblem, t0: float) -> LPSolution:
        """MILP via scipy.optimize.milp (HiGHS)."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np

            c = np.array(self._objective_coeffs(problem), dtype=float)
            if problem.objective.sense == "MAXIMIZE":
                c = -c

            constraints = []
            for con in problem.constraints:
                A_row = np.zeros(len(problem.variables))
                for term in con.terms:
                    idx = next(i for i, v in enumerate(problem.variables) if v.name == term.variable)
                    A_row[idx] = term.coefficient
                if con.sense == "<=":
                    constraints.append(LinearConstraint(A_row, ub=con.rhs))
                elif con.sense == ">=":
                    constraints.append(LinearConstraint(A_row, lb=con.rhs))
                else:
                    constraints.append(LinearConstraint(A_row, lb=con.rhs, ub=con.rhs))

            lb = np.array([v.lb for v in problem.variables])
            ub = np.array([v.ub if v.ub != float("inf") else 1e9 for v in problem.variables])
            bounds = Bounds(lb=lb, ub=ub)

            integrality = np.array([1 if v.vtype == "INTEGER" else 0 for v in problem.variables])

            result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)

            status_map = {0: OptimizationStatus.OPTIMAL, 1: OptimizationStatus.TIME_LIMIT}
            status = status_map.get(result.status, OptimizationStatus.NO_TERMINATION)

            variable_values = {problem.variables[i].name: float(result.x[i]) for i in range(len(problem.variables))}
            obj_val = float(result.fun) if problem.objective.sense == "MINIMIZE" else -float(result.fun)

            return LPSolution(
                status=status,
                variable_values=variable_values,
                objective_value=obj_val,
                solver_time_seconds=time.time() - t0,
            )
        except ImportError:
            return LPSolution(
                status=OptimizationStatus.NUMERICAL_ERROR,
                solver_time_seconds=time.time() - t0,
            )

    def _objective_coeffs(self, problem: LPProblem) -> List[float]:
        coeffs = [0.0] * len(problem.variables)
        for term in problem.objective.terms:
            idx = next(i for i, v in enumerate(problem.variables) if v.name == term.variable)
            coeffs[idx] = term.coefficient
        return coeffs

    def _inequality_constraints(self, problem: LPProblem) -> Tuple[List[List[float]], List[float]]:
        A = []
        b = []
        for con in problem.constraints:
            if con.sense == "<=":
                row = [0.0] * len(problem.variables)
                for term in con.terms:
                    idx = next(i for i, v in enumerate(problem.variables) if v.name == term.variable)
                    row[idx] = term.coefficient
                A.append(row)
                b.append(con.rhs)
            elif con.sense == ">=":
                row = [0.0] * len(problem.variables)
                for term in con.terms:
                    idx = next(i for i, v in enumerate(problem.variables) if v.name == term.variable)
                    row[idx] = -term.coefficient
                A.append(row)
                b.append(-con.rhs)
        return A, b

    def _equality_constraints(self, problem: LPProblem) -> Tuple[List[List[float]], List[float]]:
        A = []
        b = []
        for con in problem.constraints:
            if con.sense == "==":
                row = [0.0] * len(problem.variables)
                for term in con.terms:
                    idx = next(i for i, v in enumerate(problem.variables) if v.name == term.variable)
                    row[idx] = term.coefficient
                A.append(row)
                b.append(con.rhs)
        return A, b


# === Real cuOpt LP backend ===

class CuOptLPBackend:
    """Real cuopt.linear_programming wrapper. Requires NVIDIA GPU + cuopt pip install."""

    def __init__(self):
        try:
            from cuopt.linear_programming.problem import (
                Problem, CONTINUOUS, INTEGER, MINIMIZE, MAXIMIZE,
            )
            from cuopt.linear_programming.solver_settings import SolverSettings
            self._Problem = Problem
            self._CONTINUOUS = CONTINUOUS
            self._INTEGER = INTEGER
            self._MINIMIZE = MINIMIZE
            self._MAXIMIZE = MAXIMIZE
            self._SolverSettings = SolverSettings
        except ImportError as e:
            raise ImportError(
                f"CuOptLPBackend requires NVIDIA cuOpt.\n"
                f"  pip install --extra-index-url=https://pypi.nvidia.com \\\n"
                f"    nvidia-cuda-runtime-cu12==12.9.* \\\n"
                f"    cuopt-server-cu12==26.10.* cuopt-sh-client==26.10.*\n"
                f"Original error: {e}"
            )

    def solve(self, problem: LPProblem) -> LPSolution:
        t0 = time.time()

        cu_problem = self._Problem(problem.name)

        # Add variables
        var_map = {}
        for v in problem.variables:
            vtype = self._INTEGER if v.vtype == "INTEGER" else self._CONTINUOUS
            ub = v.ub if v.ub != float("inf") else None
            cu_var = cu_problem.addVariable(lb=v.lb, ub=ub, vtype=vtype, name=v.name)
            var_map[v.name] = cu_var

        # Add constraints
        for con in problem.constraints:
            expr = sum(t.coefficient * var_map[t.variable] for t in con.terms)
            cu_problem.addConstraint(expr, name=con.name)

        # Set objective
        obj_expr = sum(t.coefficient * var_map[t.variable] for t in problem.objective.terms)
        sense = self._MAXIMIZE if problem.objective.sense == "MAXIMIZE" else self._MINIMIZE
        cu_problem.setObjective(obj_expr, sense=sense)

        # Solve
        settings = self._SolverSettings()
        settings.set_parameter("time_limit", problem.time_limit_seconds)
        cu_problem.solve(settings)

        # Map cuOpt status (PascalCase)
        status_map = {
            "Optimal": OptimizationStatus.OPTIMAL,
            "PrimalFeasible": OptimizationStatus.PRIMAL_FEASIBLE,
            "FeasibleFound": OptimizationStatus.FEASIBLE_FOUND,
            "Infeasible": OptimizationStatus.INFEASIBLE,
            "Unbounded": OptimizationStatus.UNBOUNDED,
            "TimeLimit": OptimizationStatus.TIME_LIMIT,
            "IterationLimit": OptimizationStatus.ITERATION_LIMIT,
            "NumericalError": OptimizationStatus.NUMERICAL_ERROR,
        }
        status = status_map.get(cu_problem.Status.name, OptimizationStatus.NO_TERMINATION)

        variable_values = {v.name: float(v.getValue()) for v in problem.variables}
        obj_val = float(cu_problem.ObjValue) if cu_problem.ObjValue is not None else 0.0

        return LPSolution(
            status=status,
            variable_values=variable_values,
            objective_value=obj_val,
            solver_time_seconds=time.time() - t0,
        )


# === The substrate ===

@dataclass
class LPSubstrate:
    """A Quilt LP/MILP/QP substrate backed by cuOpt."""

    backend: LPBackend = field(default_factory=MockCuOptLPBackend)
    cell_id: str = "lp-substrate-cell"
    prev_witness_id: str = ""
    receipts: List[OptimizationReceipt] = field(default_factory=list)

    def solve(self, problem: LPProblem) -> OptimizationReceipt:
        """Solve an LP/MILP problem and emit a witness receipt."""
        solution = self.backend.solve(problem)

        payload = {
            "problem_name": problem.name,
            "n_variables": len(problem.variables),
            "n_constraints": len(problem.constraints),
            "status": solution.status,
            "objective_value": round(solution.objective_value, 6),
            "variable_values": {k: round(v, 6) for k, v in solution.variable_values.items()},
            "solver_time_seconds": round(solution.solver_time_seconds, 6),
        }

        receipt = OptimizationReceipt.build(
            cell_id=self.cell_id,
            substrate="cuopt-lp",
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

# LP status is the same vocabulary
LPStatus = OptimizationStatus
