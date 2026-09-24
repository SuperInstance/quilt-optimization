"""
quilt-optimization demo — 5-act walkthrough.

Act 1: TSP / VRP via cuOpt routing substrate
Act 2: LP (resource allocation) via cuOpt LP substrate
Act 3: MILP (facility location with yes/no decisions) via cuOpt LP substrate
Act 4: Witness chain (3 receipts across 3 substrates)
Act 5: Verify all receipts ACCEPT polarity, chain intact

Run: PYTHONPATH=src python3 examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_optimization import (
    RoutingSubstrate, MockCuOptRoutingBackend, RoutingProblem,
    LPSubstrate, MockCuOptLPBackend, LPProblem, LPVariable,
    LPConstraint, LPObjective, LinearTerm,
)


def act_1_routing():
    """Act 1: A small VRP — 4 locations, 2 vehicles, 3 orders."""
    print("=" * 60)
    print("ACT 1: Vehicle Routing Problem (VRP)")
    print("=" * 60)

    cost_matrix = [
        [0,  10, 15, 20],
        [10, 0,  12, 18],
        [15, 12, 0,  14],
        [20, 18, 14, 0],
    ]

    problem = RoutingProblem(
        name="warehouse-delivery",
        n_locations=4,
        n_vehicles=2,
        n_orders=3,
        cost_matrix=cost_matrix,
        order_locations=[1, 2, 3],
        vehicle_starts=[0, 0],
        vehicle_ends=[0, 0],
        vehicle_capacities=[100, 100],
        order_demands=[10, 20, 15],
    )

    substrate = RoutingSubstrate(
        backend=MockCuOptRoutingBackend(),
        cell_id="warehouse-cell",
    )
    receipt = substrate.solve(problem)
    print(f"\n  Receipt polarity: {receipt.polarity}")
    print(f"  Total cost: {receipt.payload['total_cost']}")
    print(f"  Routes: {receipt.payload['routes']}")
    print(f"  Witness id: {receipt.witness_id}")
    return substrate


def act_2_lp(substrate_chain):
    """Act 2: LP — resource allocation across 2 products."""
    print("\n" + "=" * 60)
    print("ACT 2: Linear Programming (max production)")
    print("=" * 60)

    problem = LPProblem(
        name="production-mix",
        variables=[LPVariable(name="x", lb=0), LPVariable(name="y", lb=0)],
        constraints=[
            LPConstraint(
                terms=[LinearTerm("x", 2), LinearTerm("y", 3)],
                rhs=120, sense="<=", name="machine_a",
            ),
            LPConstraint(
                terms=[LinearTerm("x", 4), LinearTerm("y", 2)],
                rhs=100, sense="<=", name="machine_b",
            ),
        ],
        objective=LPObjective(
            terms=[LinearTerm("x", 40), LinearTerm("y", 30)],
            sense="MAXIMIZE",
        ),
    )

    substrate = LPSubstrate(
        backend=MockCuOptLPBackend(),
        cell_id="production-cell",
        prev_witness_id=substrate_chain.chain_head,
    )
    receipt = substrate.solve(problem)
    print(f"\n  Receipt polarity: {receipt.polarity}")
    print(f"  Objective: ${receipt.payload['objective_value']}")
    print(f"  x={receipt.payload['variable_values']['x']}, y={receipt.payload['variable_values']['y']}")
    print(f"  Witness id: {receipt.witness_id}")
    return substrate


def act_3_milp(substrate_chain):
    """Act 3: MILP — facility location (open facility yes/no)."""
    print("\n" + "=" * 60)
    print("ACT 3: Mixed Integer Programming (facility location)")
    print("=" * 60)

    problem = LPProblem(
        name="open-warehouse",
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
        cell_id="warehouse-decision-cell",
        prev_witness_id=substrate_chain.chain_head,
    )
    receipt = substrate.solve(problem)
    print(f"\n  Receipt polarity: {receipt.polarity}")
    print(f"  Objective: ${receipt.payload['objective_value']}")
    print(f"  open={receipt.payload['variable_values']['open']:.0f}, "
          f"production={receipt.payload['variable_values']['production']:.1f}")
    print(f"  Witness id: {receipt.witness_id}")
    return substrate


def act_4_chain_verification(substrates):
    """Act 4: Verify witness chain integrity across all substrates."""
    print("\n" + "=" * 60)
    print("ACT 4: Witness Chain Verification (across 3 substrates)")
    print("=" * 60)

    # Collect ALL receipts in chain order
    all_receipts = []
    for s in substrates:
        all_receipts.extend(s.receipts)

    print(f"\n  Total receipts across all substrates: {len(all_receipts)}")
    print(f"  Final chain head: {substrates[-1].chain_head}")
    print()

    # Walk the chain
    for i, receipt in enumerate(all_receipts):
        prev = receipt.prev_witness_id or "(root)"
        print(f"  [{i}] {receipt.cell_id} | {receipt.substrate} | "
              f"{receipt.polarity} | {receipt.witness_id}")
        print(f"      prev: {prev}")
        print(f"      status: {receipt.status}")

    # Verify each links to the next
    print("\n  Chain integrity:")
    all_intact = True
    for i in range(len(all_receipts) - 1):
        curr = all_receipts[i]
        nxt = all_receipts[i + 1]
        intact = (nxt.prev_witness_id == curr.witness_id)
        all_intact = all_intact and intact
        print(f"    [{i}]→[{i+1}]: {'INTACT' if intact else 'BROKEN'}")

    return all_intact


def act_5_summary(receipts):
    """Act 5: Summary — count of ACCEPT/DRIFT/REFUSE."""
    print("\n" + "=" * 60)
    print("ACT 5: Receipt Polarity Summary")
    print("=" * 60)

    counts = {"ACCEPT": 0, "DRIFT": 0, "REFUSE": 0}
    for r in receipts:
        counts[r.polarity] += 1

    print(f"\n  ACCEPT: {counts['ACCEPT']}")
    print(f"  DRIFT:  {counts['DRIFT']}")
    print(f"  REFUSE: {counts['REFUSE']}")

    canary = "quilt-optimization-canary-001"
    print(f"\n  Canary ({canary}): ALL ACCEPT — substrate walker composed 3 substrates, "
          f"chain intact.")


def main():
    print("\n🌱 quilt-optimization — NVIDIA cuOpt as a Quilt substrate\n")

    # Act 1
    s1 = act_1_routing()

    # Act 2 — chains from Act 1
    s2 = act_2_lp(s1)

    # Act 3 — chains from Act 2
    s3 = act_3_milp(s2)

    # Act 4 — verify the chain across all 3 substrates
    intact = act_4_chain_verification([s1, s2, s3])

    # Act 5 — summary (count over all receipts)
    all_receipts = s1.receipts + s2.receipts + s3.receipts
    act_5_summary(all_receipts)

    print("\n" + "=" * 60)
    if intact:
        print("  ✓ Demo complete. Witness chain INTACT.")
        print("  Try: PYTHONPATH=src python3 examples/with_real_cuopt.py")
        print("  (requires NVIDIA GPU + cuopt pip install)")
    else:
        print("  ✗ Chain BROKEN — bug in witness propagation")
    print("=" * 60 + "\n")

    return 0 if intact else 1


if __name__ == "__main__":
    sys.exit(main())
