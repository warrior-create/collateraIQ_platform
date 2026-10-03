"""
Collateral optimiser: cheapest-to-deliver allocation.

Uses scipy.optimize.linprog to solve the linear programme:
  Minimise: cost(collateral) = Σ_i Σ_j cost_ij × x_ij
  Subject to:
    - Each client's IM requirement is met: Σ_i x_ij × (1 - haircut_ij) ≥ IM_j
    - Inventory constraint: Σ_j x_ij ≤ inventory_i
    - Non-negativity: x_ij ≥ 0

Cost function: funding cost of each collateral type (opportunity cost of using it).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


def optimise_collateral_allocation(
    clients: list[dict],
    inventory: list[dict],
    funding_costs: dict[str, float] | None = None,
) -> dict:
    """
    Cheapest-to-deliver (CTD) collateral allocation.

    Parameters
    ----------
    clients   : list of {client_id, im_required_inr, eligible_collateral}
                eligible_collateral: list of asset types acceptable
    inventory : list of {asset_type, available_inr, haircut}
    funding_costs : dict of {asset_type: annual_funding_cost}
                   Higher cost = more expensive to use as collateral

    Returns
    -------
    dict: allocation matrix, total_cost, savings vs naive, per-client allocation
    """
    if funding_costs is None:
        funding_costs = {
            "cash_inr": 0.07,        # Opportunity cost of cash
            "cash_usd": 0.05,
            "gsec_short": 0.065,
            "gsec_long": 0.06,
            "equity_nifty50": 0.12,  # Higher cost (more volatile)
            "equity_single_stock": 0.15,
        }

    n_assets = len(inventory)
    n_clients = len(clients)

    if n_assets == 0 or n_clients == 0:
        return {"error": "No assets or clients", "allocation": {}}

    # Build indices
    asset_names = [a["asset_type"] for a in inventory]
    client_ids = [c["client_id"] for c in clients]

    # Decision variables: x[i,j] = INR of asset i allocated to client j
    n_vars = n_assets * n_clients

    # Objective: minimise funding cost
    c = np.zeros(n_vars)
    for i, asset in enumerate(inventory):
        cost = funding_costs.get(asset["asset_type"], 0.10)
        for j in range(n_clients):
            c[i * n_clients + j] = cost

    # Inequality constraints (≤ form for linprog)
    A_ub = []
    b_ub = []

    # 1. Inventory constraints: Σ_j x_ij ≤ available_i
    for i, asset in enumerate(inventory):
        row = np.zeros(n_vars)
        for j in range(n_clients):
            row[i * n_clients + j] = 1.0
        A_ub.append(row)
        b_ub.append(asset["available_inr"])

    # Equality constraints (≥ becomes ≤ with negation)
    A_eq = []
    b_eq = []

    # 2. Client IM requirements: Σ_i x_ij × (1 - haircut_i) ≥ IM_j
    # → -Σ_i x_ij × (1 - haircut_i) ≤ -IM_j
    for j, client in enumerate(clients):
        row = np.zeros(n_vars)
        for i, asset in enumerate(inventory):
            if asset["asset_type"] in [e["asset"] if isinstance(e, dict) else e
                                        for e in client.get("eligible_collateral", asset_names)]:
                haircut = asset.get("haircut", 0.0)
                row[i * n_clients + j] = -(1 - haircut)
        A_ub.append(row)
        b_ub.append(-client["im_required_inr"])

    # Bounds: x_ij ≥ 0
    bounds = [(0, None)] * n_vars

    result = linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        bounds=bounds,
        method="highs",
    )

    if result.success:
        x = result.x.reshape(n_assets, n_clients)
        total_cost = float(result.fun)

        # Naive allocation (always use cash first regardless of cost)
        naive_cost = _naive_allocation_cost(clients, inventory, funding_costs)
        savings = max(naive_cost - total_cost, 0)

        allocation = {
            client_ids[j]: {
                asset_names[i]: float(x[i, j])
                for i in range(n_assets)
                if x[i, j] > 1000  # Filter tiny allocations
            }
            for j in range(n_clients)
        }

        return {
            "status": "optimal",
            "total_cost_inr_pa": float(total_cost),
            "naive_cost_inr_pa": float(naive_cost),
            "savings_inr_pa": float(savings),
            "savings_pct": float(savings / naive_cost * 100) if naive_cost > 0 else 0,
            "allocation": allocation,
        }
    else:
        logger.warning(f"Collateral optimisation failed: {result.message}")
        return {"status": "infeasible", "message": result.message, "allocation": {}}


def _naive_allocation_cost(clients, inventory, funding_costs) -> float:
    """Naive: always use cheapest available asset without optimising."""
    total = 0.0
    remaining = {a["asset_type"]: a["available_inr"] for a in inventory}

    for client in clients:
        needed = client["im_required_inr"]
        # Use cheapest eligible asset first
        sorted_assets = sorted(
            [a for a in inventory if a["available_inr"] > 0],
            key=lambda a: funding_costs.get(a["asset_type"], 0.10),
        )
        for asset in sorted_assets:
            if needed <= 0:
                break
            avail = remaining[asset["asset_type"]]
            use = min(needed / (1 - asset.get("haircut", 0)), avail)
            remaining[asset["asset_type"]] -= use
            total += use * funding_costs.get(asset["asset_type"], 0.10)
            needed -= use * (1 - asset.get("haircut", 0))

    return total
