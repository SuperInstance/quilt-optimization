"""
substrate.py — the shared envelope for cuOpt-as-Quilt-substrate.

Mirrors the substrate walker pattern from quilt-seed.vibe:
  - Substrate wrapper takes a backend (mock or real)
  - Each solve emits an OptimizationReceipt (CellReceipt subclass)
  - Receipts chain via prev_witness_id
  - Polarity maps from cuOpt's status vocabulary

Why a separate envelope (not reusing CellReceipt from quilt-seed):
  - quilt_optimization is a standalone package
  - Optional dependency: quilt-seed is *not* required at runtime
  - The envelope is a subset of CellReceipt (just enough for optimization)
  - If quilt-seed is installed, OptimizationReceipt can be promoted to CellReceipt
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional


# === Polarity vocabulary (mirrors quilt_seed.vibe.CellReceipt) ===

ACCEPT = "ACCEPT"     # optimal or primal-feasible solution found
DRIFT = "DRIFT"       # time/iteration limit — partial solution, may be suboptimal
REFUSE = "REFUSE"     # infeasible, unbounded, numerical error


# === Status vocabulary (mirrors cuOpt's PascalCase) ===

class OptimizationStatus:
    """Subset of cuOpt's status vocabulary — covers LP, MILP, QP, Routing."""

    # ACCEPT
    OPTIMAL = "Optimal"            # LP/QP/Routing: proven optimal
    PRIMAL_FEASIBLE = "PrimalFeasible"  # LP/QP: feasible but not proven optimal
    FEASIBLE_FOUND = "FeasibleFound"   # MILP: high-quality feasible within gap

    # DRIFT
    TIME_LIMIT = "TimeLimit"       # hit the time limit
    ITERATION_LIMIT = "IterationLimit"
    NO_TERMINATION = "NoTermination"   # solver stopped without conclusion

    # REFUSE
    INFEASIBLE = "Infeasible"      # MILP: no feasible solution
    UNBOUNDED = "Unbounded"        # MILP: objective unbounded
    NUMERICAL_ERROR = "NumericalError"  # LP/QP: numerical instability
    PRIMAL_INFEASIBLE = "PrimalInfeasible"  # LP/QP: primal infeasible
    DUAL_INFEASIBLE = "DualInfeasible"  # LP/QP: dual infeasible (problem is unbounded)

    # Routing-specific
    FAIL = "FAIL"                 # routing: solver failed
    TIMEOUT = "TIMEOUT"           # routing: hit time limit
    EMPTY = "EMPTY"               # routing: no solution found


# === Status → polarity mapping ===

# Mapping table — single source of truth for cuOpt status → Quilt polarity.
# Mirrors what the substrate walker would do with each status.

_STATUS_TO_POLARITY = {
    OptimizationStatus.OPTIMAL: ACCEPT,
    OptimizationStatus.PRIMAL_FEASIBLE: ACCEPT,
    OptimizationStatus.FEASIBLE_FOUND: ACCEPT,
    OptimizationStatus.TIME_LIMIT: DRIFT,
    OptimizationStatus.ITERATION_LIMIT: DRIFT,
    OptimizationStatus.NO_TERMINATION: DRIFT,
    OptimizationStatus.INFEASIBLE: REFUSE,
    OptimizationStatus.UNBOUNDED: REFUSE,
    OptimizationStatus.NUMERICAL_ERROR: REFUSE,
    OptimizationStatus.PRIMAL_INFEASIBLE: REFUSE,
    OptimizationStatus.DUAL_INFEASIBLE: REFUSE,
    OptimizationStatus.FAIL: REFUSE,
    OptimizationStatus.TIMEOUT: DRIFT,
    OptimizationStatus.EMPTY: REFUSE,
}


def status_to_polarity(status: str) -> str:
    """Map cuOpt's status string to Quilt polarity (ACCEPT/DRIFT/REFUSE).

    Unknown statuses default to DRIFT (the substrate's view of the world
    is uncertain — the cell should hold the receipt and decide).
    """
    return _STATUS_TO_POLARITY.get(status, DRIFT)


# === The receipt envelope ===

@dataclass
class OptimizationReceipt:
    """A cuOpt solve's witness — chains via prev_witness_id.

    This is a subset of quilt_seed.vibe.CellReceipt, scoped to optimization
    substrates. If quilt-seed is installed, this receipt can be wrapped
    in a CellReceipt for the full envelope.
    """
    witness_id: str
    prev_witness_id: str
    cell_id: str
    substrate: str          # "cuopt-routing" or "cuopt-lp"
    polarity: str          # ACCEPT | DRIFT | REFUSE
    payload: dict
    timestamp: int
    status: str            # raw cuOpt status string

    @classmethod
    def build(
        cls,
        *,
        cell_id: str,
        substrate: str,
        status: str,
        payload: dict,
        prev_witness_id: str = "",
        cell_id_field: str = "cell_id",
        timestamp: Optional[int] = None,
    ) -> "OptimizationReceipt":
        """Build a receipt with chained witness_id.

        The witness_id is sha256-of-canonical(prev + payload + polarity + ts).
        Same discipline as candor's canonical() and the SEAM spec counter-proposal.
        """
        if timestamp is None:
            timestamp = int(time.time())

        polarity = status_to_polarity(status)

        # Canonical JSON for hashing (sorted keys, no whitespace)
        body = json.dumps({
            "prev": prev_witness_id,
            cell_id_field: cell_id,
            "substrate": substrate,
            "polarity": polarity,
            "status": status,
            "payload": payload,
            "ts": timestamp,
        }, sort_keys=True, separators=(",", ":")).encode()

        witness_id = hashlib.sha256(body).hexdigest()[:16]

        return cls(
            witness_id=witness_id,
            prev_witness_id=prev_witness_id,
            cell_id=cell_id,
            substrate=substrate,
            polarity=polarity,
            payload=payload,
            timestamp=timestamp,
            status=status,
        )

    def to_dict(self) -> dict:
        return {
            "witness_id": self.witness_id,
            "prev_witness_id": self.prev_witness_id,
            "cell_id": self.cell_id,
            "substrate": self.substrate,
            "polarity": self.polarity,
            "status": self.status,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    def chain_with(self, other: "OptimizationReceipt") -> bool:
        """Verify this receipt's prev_witness_id matches other's witness_id."""
        return self.prev_witness_id == other.witness_id
