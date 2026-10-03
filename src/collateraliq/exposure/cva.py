"""
CVA and SA-CCR (simplified) calculation.

CVA formula:
  CVA = (1-R) × Σ DF(tᵢ) × EE(tᵢ) × ΔPD(tᵢ)

SA-CCR (Basel III CRE53):
  EAD = α × (RC + PFE_add-on)
  α = 1.4

References
----------
BCBS d279: The standardised approach for measuring counterparty credit risk exposures (SA-CCR).
Gregory, Counterparty Credit Risk and CVA, 2nd ed.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALPHA = 1.4  # SA-CCR regulatory multiplier (BCBS CRE53.33)


def compute_cva(
    ee_profile: list[float],
    time_grid: list[float],
    hazard_rate: float,
    recovery_rate: float = 0.40,
    risk_free_rate: float = 0.065,
) -> float:
    """
    Compute unilateral CVA.

    CVA = (1-R) × Σ DF(tᵢ) × EE(tᵢ) × ΔPD(tᵢ)

    Parameters
    ----------
    ee_profile    : expected exposure at each time step (INR)
    time_grid     : time points in years
    hazard_rate   : annualised default intensity (λ)
    recovery_rate : R (assumed constant)
    risk_free_rate: flat discount rate
    """
    lgd = 1.0 - recovery_rate
    cva = 0.0

    for i in range(1, len(time_grid)):
        t1 = time_grid[i - 1]
        t2 = time_grid[i]
        t_mid = (t1 + t2) / 2

        ee = ee_profile[i]
        df = np.exp(-risk_free_rate * t_mid)
        pd_incremental = (np.exp(-hazard_rate * t1) - np.exp(-hazard_rate * t2))

        cva += lgd * df * ee * pd_incremental

    return float(max(cva, 0))


def compute_sa_ccr(
    trades_df: pd.DataFrame,
    market: dict,
    net_mtm: float,
    im_posted: float = 0.0,
    vm_posted: float = 0.0,
    params: dict | None = None,
) -> dict:
    """
    Simplified SA-CCR EAD calculation.

    EAD = α × (RC + PFE_add-on)

    Replacement cost (RC):
      RC = max(V - C, 0) where V = net MTM, C = collateral held

    PFE add-on: asset-class supervisory factors × adjusted notionals.

    Parameters defined in BCBS CRE53. Alpha = 1.4 (CRE53.33).
    """
    if params is None:
        params = _default_sa_ccr_params()

    # Replacement cost (CRE53.37)
    collateral = vm_posted + im_posted
    rc = max(net_mtm - collateral, 0)

    # PFE add-on per asset class (CRE53.40–CRE53.67)
    pfe_addon = _compute_pfe_addon(trades_df, market, params)

    # Multiplier (CRE53.33): allows EAD to be lower when out-of-the-money
    net_over_addon = (net_mtm - collateral) / (2 * max(pfe_addon, 1)) if pfe_addon > 0 else 0
    multiplier = min(1.0, 0.05 + 0.95 * np.exp(net_over_addon / 0.95))

    ead = ALPHA * (rc + multiplier * pfe_addon)

    return {
        "ead_sa_ccr": float(ead),
        "replacement_cost": float(rc),
        "pfe_addon": float(pfe_addon),
        "multiplier": float(multiplier),
        "alpha": ALPHA,
    }


def _compute_pfe_addon(trades_df: pd.DataFrame, market: dict, params: dict) -> float:
    """
    Compute SA-CCR PFE add-on by asset class.

    Uses supervisory factors from BCBS CRE53 tables.
    """
    total_addon = 0.0

    for _, trade in trades_df.iterrows():
        product = trade.get("product_type", "")
        notional = trade.get("notional_inr", 0)
        import json
        trade_params = json.loads(trade.get("params", "{}")) if isinstance(trade.get("params"), str) else {}

        if product == "irs":
            tenor = trade_params.get("tenor_years", 5)
            sf = _irs_supervisory_factor(tenor, params)
            adj_notional = notional * _maturity_factor(tenor)
            total_addon += abs(sf * adj_notional)

        elif product == "fx_forward":
            sf = params["fx"]["sf"]
            total_addon += abs(sf * notional)

        elif product in ("equity_financing", "equity_options", "single_stock_options"):
            sf = params["equity"]["sf"]
            total_addon += abs(sf * notional)

        elif product == "cds_index":
            sf = params["credit_ig"]["sf"]
            total_addon += abs(sf * notional)

        elif product == "gsec_repo":
            tenor = trade_params.get("bond_maturity_years", 5)
            sf = _irs_supervisory_factor(tenor, params)
            adj_notional = notional * _maturity_factor(tenor)
            total_addon += abs(sf * adj_notional)

    return total_addon


def _irs_supervisory_factor(tenor_years: float, params: dict) -> float:
    """SA-CCR supervisory factor for IRS by maturity bucket."""
    if tenor_years <= 1:
        return params["interest_rate"]["sf_short"]
    elif tenor_years <= 5:
        return params["interest_rate"]["sf_medium"]
    else:
        return params["interest_rate"]["sf_long"]


def _maturity_factor(tenor_years: float) -> float:
    """SA-CCR maturity factor for IRS (CRE53.53)."""
    return min(np.sqrt(tenor_years / 5), 1.0)


def _default_sa_ccr_params() -> dict:
    """
    Default supervisory factors from BCBS CRE53 Table 2.

    Values as of Basel III final rules. Always verify against current BCBS standard.
    """
    return {
        "interest_rate": {
            "sf_short": 0.0050,   # CRE53 Table 2, <1yr
            "sf_medium": 0.0050,  # 1yr-5yr
            "sf_long": 0.0150,    # >5yr
        },
        "fx": {"sf": 0.04},
        "equity": {"sf": 0.32},
        "credit_ig": {"sf": 0.0038},
        "credit_hy": {"sf": 0.0106},
    }
