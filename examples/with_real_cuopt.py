"""
with_real_cuopt.py — same demo as demo.py, but using the real cuOpt backends.

Requires:
  - NVIDIA GPU (Volta or newer)
  - pip install --extra-index-url=https://pypi.nvidia.com \
        nvidia-cuda-runtime-cu12==12.9.* \
        cuopt-server-cu12==26.10.* cuopt-sh-client==26.10.*
  - pip install quilt-optimization[cuopt]

Run: PYTHONPATH=src python3 examples/with_real_cuopt.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_optimization import (
    RoutingSubstrate, CuOptRoutingBackend, RoutingProblem,
    LPSubstrate, CuOptLPBackend, LPProblem, LPVariable,
    LPConstraint, LPObjective, LinearTerm,
)


def main():
    print("\n🌱 quilt-optimization — Real NVIDIA cuOpt backends (requires GPU)\n")

    # Routing
    vrp = RoutingProblem(
        name="warehouse-delivery-real",
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
    routing = RoutingSubstrate(backend=CuOptRoutingBackend())
    r = routing.solve(vrp)
    print(f"VRP: {r.polarity} | cost={r.payload['total_cost']} | routes={r.payload['routes']}")

    # LP
    lp = LPProblem(
        name="production-mix-real",
        variables=[LPVariable(name="x", lb=0), LPVariable(name="y", lb=0)],
        constraints=[
            LPConstraint(terms=[LinearTerm("x", 2), LinearTerm("y", 3)], rhs=120, sense="<=", name="a"),
            LPConstraint(terms=[LinearTerm("x", 4), LinearTerm("y", 2)], rhs=100, sense="<=", name="b"),
        ],
        objective=LPObjective(terms=[LinearTerm("x", 40), LinearTerm("y", 30)], sense="MAXIMIZE"),
    )
    lp_substrate = LPSubstrate(
        backend=CuOptLPBackend(),
        prev_witness_id=routing.chain_head,
    )
    r2 = lp_substrate.solve(lp)
    print(f"LP:  {r2.polarity} | obj=${r2.payload['objective_value']} | x={r2.payload['variable_values']['x']}, y={r2.payload['variable_values']['y']}")

    print(f"\n  Chain head: {lp_substrate.chain_head}")
    print("  ✓ Demo complete with REAL cuOpt backends.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\n  ✗ {e}")
        print("  Install cuOpt first: see header docstring")
        sys.exit(1)
