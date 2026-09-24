# cuOpt Research & Quilt Integration

> Deep dive on NVIDIA cuOpt — what it does, how it works, and how it slots into Quilt.

**Date:** 2026-09-24
**Sources:**
- [github.com/NVIDIA/cuopt](https://github.com/NVIDIA/cuopt) (cloned, 5389+ open issues, Apache-2.0)
- [cuOpt documentation](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- [cuOpt examples](https://github.com/NVIDIA/cuopt-examples)
- AGENTS.md, skills/, python/cuopt/, python/cuopt_mcp/

## 1. What is cuOpt?

NVIDIA's **GPU-accelerated optimization engine** for:

- **LP** — Linear Programming
- **MILP** — Mixed-Integer Linear Programming (beta)
- **QP** — Quadratic Programming (beta)
- **QCQP** — Quadratically Constrained QP (beta)
- **SOCP** — Second-Order Cone Programming (beta)
- **Routing** — VRP, TSP, Pickup-Delivery

Solved with one solver across all problem types (LP/MILP share infrastructure, routing has its own).

### Core engine

- **C++ core**, wrapped with:
  - **C API** (LP/MILP/QP — no C routing)
  - **Python API** (`cuopt.routing`, `cuopt.linear_programming`)
  - **Server API** (REST: `/cuopt/request`, `/cuopt/solution/{reqId}`)
  - **MCP server** (`cuopt_mcp` — exposes LP/MILP over MCP for AI agents)

### Performance

- "Near real-time solutions for large-scale LPs with millions of variables and constraints"
- GPU-parallelized, runs on Volta+ (Compute Capability >= 7.0)
- Open-source (Apache-2.0), also hosted as a COIN-OR project

### Agent-first design

cuOpt ships **AI-agent skills** (in `skills/`) and an **MCP server** explicitly to make it composable with agents. From `gemini-extension.json`:

```json
{
  "name": "nvidia-cuopt-skills",
  "description": "Agent skills for NVIDIA cuOpt optimization engine: routing, LP/MILP/QP, installation, and server.",
  "contextFileName": "AGENTS.md"
}
```

This is exactly the substrate walker pattern (Mavis) — cuOpt is *already* designed to be wrapped by agents.

## 2. The Python API surface

### Routing (`cuopt.routing`)

```python
from cuopt import routing

# Build the problem
dm = routing.DataModel(n_locations=4, n_fleet=2, n_orders=3)
dm.add_cost_matrix(cost_matrix)              # n_locations × n_locations
dm.set_order_locations(order_loc_series)     # cudf Series
dm.set_vehicle_locations(starts, ends)
dm.add_capacity_dimension(name, demand, capacity)
dm.set_order_time_windows(earliest, latest)
dm.set_pickup_delivery_pairs(pickup_idx, delivery_idx)

# Solve
ss = routing.SolverSettings()
ss.set_time_limit(30)
solution = routing.Solve(dm, ss)

# Inspect
status = solution.get_status()               # 0=SUCCESS, 1=FAIL, 2=TIMEOUT, 3=EMPTY
solution.get_route()                          # cudf DataFrame: truck_id, route, location_id
solution.get_total_objective()                # float
```

### Linear programming (`cuopt.linear_programming`)

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, INTEGER, MINIMIZE, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

problem = Problem("MyLP")
x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

problem.addConstraint(2*x + 3*y <= 120, name="resource_a")
problem.setObjective(40*x + 30*y, sense=MAXIMIZE)

settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# CRITICAL: status uses PascalCase
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(problem.ObjValue)
```

**Gotcha:** status names are `Optimal`, `FeasibleFound`, etc. — NOT `OPTIMAL`. The lowercase form silently never matches.

## 3. The Quilt integration

### Mapping the substrate walker pattern

The substrate walker (Mavis) composes any external tool as a substrate. cuOpt is a perfect fit:

| Quilt substrate walker | NVIDIA cuOpt              |
|------------------------|---------------------------|
| Substrate              | cuOpt API (routing or LP) |
| Cell                   | A routing/optimization cell |
| Backend (mock)         | Greedy nearest-neighbor (routing) / scipy.optimize (LP) |
| Backend (real)         | `cuopt.routing.Solve` / `cuopt.linear_programming.Problem.solve` |
| Receipt polarity       | ACCEPT / DRIFT / REFUSE   |
| Receipt chain          | `prev_witness_id` (sha256-of-canonical) |
| Legalese layer         | OptimizationReceipt → Claim (in quilt-seed) |
| Vessel decision        | Cell decides what to do with the optimal solution |

### The envelope

```python
@dataclass
class OptimizationReceipt:
    witness_id: str              # sha256-of-canonical
    prev_witness_id: str         # chain link
    cell_id: str                 # which cell emitted
    substrate: str               # "cuopt-routing" or "cuopt-lp"
    polarity: str                # ACCEPT | DRIFT | REFUSE
    payload: dict                # solution details
    timestamp: int
    status: str                  # raw cuOpt status (Optimal, TimeLimit, etc.)
```

This is a **subset** of `quilt_seed.vibe.CellReceipt`. If `quilt-seed` is installed, an `OptimizationReceipt` can be promoted to a full `CellReceipt` by wrapping it. If not, it stands alone.

### Status → polarity mapping

The substrate walker needs a *single source of truth* for which cuOpt statuses count as which polarities:

| cuOpt status             | Polarity | Why                                    |
|--------------------------|----------|----------------------------------------|
| `Optimal`                | ACCEPT   | Proven optimal                         |
| `PrimalFeasible`         | ACCEPT   | Feasible (LP/QP; not proven optimal)   |
| `FeasibleFound`          | ACCEPT   | MILP within gap tolerance              |
| `TimeLimit`              | DRIFT    | Suboptimal but not refused             |
| `IterationLimit`         | DRIFT    | Suboptimal but not refused             |
| `NoTermination`          | DRIFT    | Solver stopped without conclusion      |
| `Infeasible`             | REFUSE   | No feasible solution exists            |
| `Unbounded`              | REFUSE   | Objective unbounded                    |
| `NumericalError`         | REFUSE   | Numerical instability                  |
| `PrimalInfeasible`       | REFUSE   | LP/QP: primal infeasible               |
| `DualInfeasible`         | REFUSE   | LP/QP: dual infeasible (unbounded)     |
| `FAIL` (routing)         | REFUSE   | Routing: solver failed                 |
| `EMPTY` (routing)        | REFUSE   | Routing: no solution                   |
| *unknown*                | DRIFT    | The substrate's view is uncertain      |

### Two backends per substrate

**`Mock*Backend`** — deterministic, offline, no GPU. Uses:
- Routing: greedy nearest-neighbor heuristic (suboptimal but reproducible)
- LP/MILP: scipy.optimize.linprog / milp (HiGHS solver)

**`CuOpt*Backend`** — real cuOpt via `import cuopt`. Requires:
- NVIDIA GPU (Volta+)
- `pip install --extra-index-url=https://pypi.nvidia.com nvidia-cuda-runtime-cu12 cuopt-server-cu12 cuopt-sh-client`

The substrate walker composes both backends transparently. Tests use mocks; production uses real.

## 4. Where this fits in the Quilt ecosystem

### Existing substrates (Holodeck, in `quilt-spreadsheet-inference`)

- **`jepa_substrate.py`** — predictive geometry (JEPA)
- **`jev_substrate.py`** — truth oracle (JEV)
- **`moth_substrate.py`** — physics substrate (MOTH)
- **`llm_substrate.py` + `llm_zoo.py`** — multi-provider LLM substrate (5 providers)
- **`origami_substrate.py`** — permutation substrates (Miura fold, rotation, swap, commutator)

### What `quilt-optimization` adds

- **`cuopt-routing`** — VRP/TSP/PDP substrate
- **`cuopt-lp`** — LP/MILP/QP substrate

### The cross-substrate witness chain

A cell can compose any combination:

```
CellPlan →
  cuopt-lp (decide production mix)
    → ACCEPT
  cuopt-routing (decide fleet routes)
    → ACCEPT (chained)
  jev-substrate (verify the solution against canon)
    → ACCEPT (chained)
  vessel (quilt-seed) decides
```

Every step's receipt is a witness. The legalese layer (`LegaleseNetwork`) records the receipts as Claims. The vessel decides whether the chain is convincing.

## 5. Use cases for cuOpt in Quilt

### Resource allocation

A Quilt cell managing a fleet of agents, a factory's output, or a workspace's budget has an LP problem every time it needs to allocate:

```python
# 5 products, 3 machines, 2 raw materials → maximize revenue
lp = LPProblem(
    name="monthly-allocation",
    variables=[LPVariable(name=f"p{i}", lb=0) for i in range(5)],
    constraints=[
        LPConstraint(terms=..., rhs=hours_available, sense="<=", name="machine_a"),
        LPConstraint(terms=..., rhs=raw_material, sense="<=", name="raw_a"),
    ],
    objective=LPObjective(terms=..., sense="MAXIMIZE"),
)
```

### Vehicle routing / fleet dispatch

A fleet of agents (Quilt cells in different rooms) needs to visit orders:

```python
vrp = RoutingProblem(
    name="daily-delivery",
    n_locations=20,
    n_vehicles=5,
    n_orders=15,
    cost_matrix=distance_matrix,
    order_locations=order_indices,
    vehicle_starts=[0]*5,
    vehicle_ends=[0]*5,
    vehicle_capacities=[100]*5,
    order_demands=[10, 20, 15, ...],
    order_time_windows=[[0, 480], [0, 480], ...],
)
```

### Multi-objective Pareto

The `cuopt-multi-objective-exploration` skill (in cuOpt) walks a Pareto frontier via weighted-sum or ε-constraint. The substrate walker can compose this with the legalese layer to record each frontier point as a Claim with a confidence interval.

### Where cuOpt doesn't fit (and what to use instead)

- **Combinatorial problems without linear structure** (graph coloring, sudoku, scheduling with complex precedence): use OR-Tools (`google-ortools` Python package)
- **Nonlinear / non-convex optimization**: cuOpt does not support general nonlinear; use scipy.optimize or IPOPT
- **Stochastic / online optimization**: cuOpt is batch; for online, use a custom bandit or RL approach

## 6. The Quilt substrate walker doctrine, codified

This package is the **fifth** substrate walker wrapper (after Vibe, CU, Holodeck, and the candor substrate offered in `quilt-canon-witness#1`):

| Substrate | Wrapper | LOC | Tests | Status |
|-----------|---------|-----|-------|--------|
| Vibe (Elephant cynicism) | `quilt-seed/vibe.py` | 199 | 17 | ✓ shipped |
| Collective Unconscious (RAG) | `quilt-seed/cu_substrate.py` | 199 | 12 | ✓ shipped |
| Optimization (cuOpt routing) | `quilt-optimization/routing.py` | 340 | 12 | ✓ shipped |
| Optimization (cuOpt LP) | `quilt-optimization/linear_programming.py` | 445 | 16 | ✓ shipped |
| Candor receipt | TBD | ~150 | TBD | offered |

The pattern is now stable. Each new substrate is:

1. **~340 LOC wrapper** — substrate-agnostic spec + backend protocol + mock + real
2. **~150 LOC tests** — backend determinism, substrate envelope, chain integrity
3. **~50 LOC demo** — 5-act walkthrough showing composition
4. **CellReceipt-compatible envelope** — chains via prev_witness_id

## 7. Open questions / next steps

- **Multi-objective Pareto via cuOpt**: would compose `cuopt-multi-objective-exploration` skill as a meta-substrate (1 call → N solves → N receipts)
- **Pareto frontier + legalese claims**: each frontier point is a Claim; the legalese records the range
- **cuOpt MCP server**: could expose the substrate over MCP for any agent (not just Quilt)
- **Online re-routing**: `cuopt.routing.re_routing` is in the codebase — would be a `ReRoutingSubstrate` for live fleet updates
- **Coupling charter**: where does the optimization cell sit in the coupling hierarchy (charter/fluidics)? Currently undecided.
- **GPU fleet**: the substrate walker is GPU-optional; production cuOpt requires GPU access. The fleet will need GPU nodes.

## 8. References

- cuOpt GitHub: <https://github.com/NVIDIA/cuopt>
- cuOpt documentation: <https://docs.nvidia.com/cuopt/user-guide/latest/>
- cuOpt examples: <https://github.com/NVIDIA/cuopt-examples>
- cuOpt MCP server: <https://github.com/NVIDIA/cuopt/tree/main/python/cuopt_mcp>
- The Quilt substrate walker doctrine: see `quilt-seed/SKILLS.md` and `STATE-OF-MIND.md`
