"""
IM add-ons: concentration, liquidity, wrong-way risk.

These are applied on top of the base IM to capture risks not captured
by historical simulation.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def concentration_addon(
    notional: float,
    adv: float,
    base_im: float,
    adv_threshold: float = 0.10,
    liquidation_days_assumption: int = 10,
    horizon_days: int = 10,
    sqrt_scaling: bool = True,
) -> dict:
    """
    Concentration add-on for large positions relative to ADV.

    If position > threshold × ADV, extra days are needed to liquidate.
    The add-on scales the IM for the extra liquidation time.

    Liquidation days = max(horizon, ceil(notional / (threshold * ADV)))
    Add-on = base_IM × (sqrt(liq_days/horizon) - 1) [if sqrt scaling]

    Parameters
    ----------
    notional  : position notional (INR)
    adv       : average daily volume (INR)
    base_im   : base IM before add-on
    adv_threshold : fraction of ADV above which concentration applies
    liquidation_days_assumption: assumed days to exit concentrated position
    horizon_days: base IM horizon
    sqrt_scaling: use √(liq_days/horizon) scaling (else linear)
    """
    if adv <= 0:
        return {"addon_inr": 0.0, "liq_days": horizon_days, "is_concentrated": False}

    # Days to liquidate at threshold% of ADV
    daily_capacity = adv_threshold * adv
    liq_days = int(np.ceil(notional / daily_capacity)) if daily_capacity > 0 else horizon_days
    liq_days = max(liq_days, horizon_days)

    if liq_days <= horizon_days:
        return {"addon_inr": 0.0, "liq_days": liq_days, "is_concentrated": False}

    if sqrt_scaling:
        scaling = np.sqrt(liq_days / horizon_days) - 1
    else:
        scaling = (liq_days / horizon_days) - 1

    addon = base_im * scaling

    # Liquidation cost: bid-ask and market impact
    # Simple approximation: 0.1% × notional × extra_days / horizon
    liq_cost = 0.001 * notional * (liq_days - horizon_days) / horizon_days

    return {
        "addon_inr": float(addon),
        "liq_cost_inr": float(liq_cost),
        "liq_days": int(liq_days),
        "is_concentrated": True,
        "adv_participation_pct": float(notional / adv) if adv > 0 else 0.0,
    }


def liquidity_addon(
    notional: float,
    adv: float,
    base_im: float,
    illiquidity_threshold: float = 0.05,
    addon_multiplier: float = 1.50,
) -> dict:
    """
    Liquidity add-on for illiquid positions.

    Triggered when daily turnover as % of position < illiquidity_threshold.

    Parameters
    ----------
    notional              : position size (INR)
    adv                   : average daily volume (INR)
    base_im               : base IM
    illiquidity_threshold : fraction of position that must be tradeable daily
    addon_multiplier      : IM multiplier for illiquid positions
    """
    if adv <= 0:
        return {"addon_inr": base_im * (addon_multiplier - 1), "is_illiquid": True}

    turnover_ratio = adv / notional if notional > 0 else 1.0

    if turnover_ratio >= illiquidity_threshold:
        return {"addon_inr": 0.0, "is_illiquid": False, "turnover_ratio": turnover_ratio}

    addon = base_im * (addon_multiplier - 1)
    return {
        "addon_inr": float(addon),
        "is_illiquid": True,
        "turnover_ratio": float(turnover_ratio),
        "addon_multiplier": addon_multiplier,
    }


def wrong_way_risk_addon(
    base_im: float,
    collateral_correlation: float,
    correlation_threshold: float = 0.40,
    addon_multiplier: float = 1.20,
) -> dict:
    """
    Wrong-way risk (WWR) add-on.

    Triggered when client's collateral is positively correlated with their exposure.
    (If client defaults, collateral also falls in value — WWR.)

    Parameters
    ----------
    base_im               : base IM
    collateral_correlation: correlation between collateral value and client exposure
    correlation_threshold : threshold above which WWR add-on applies
    addon_multiplier      : IM multiplier for WWR situations
    """
    if abs(collateral_correlation) < correlation_threshold:
        return {
            "addon_inr": 0.0,
            "is_wwr": False,
            "collateral_correlation": collateral_correlation,
        }

    addon = base_im * (addon_multiplier - 1)
    return {
        "addon_inr": float(addon),
        "is_wwr": True,
        "collateral_correlation": float(collateral_correlation),
        "addon_multiplier": addon_multiplier,
    }


def compute_all_addons(
    notional: float,
    adv: float,
    base_im: float,
    collateral_correlation: float = 0.0,
    params: dict | None = None,
) -> dict:
    """
    Compute all add-ons and return combined result.

    Total IM = base_im + concentration_addon + liquidity_addon + wwr_addon
    """
    if params is None:
        params = {}

    conc = concentration_addon(
        notional, adv, base_im,
        adv_threshold=params.get("adv_participation_threshold", 0.10),
        liquidation_days_assumption=params.get("liquidation_days_assumption", 10),
    )
    liq = liquidity_addon(
        notional, adv, base_im,
        illiquidity_threshold=params.get("illiquidity_threshold_adv", 0.05),
        addon_multiplier=params.get("illiquidity_addon_multiplier", 1.50),
    )
    wwr = wrong_way_risk_addon(
        base_im,
        collateral_correlation,
        correlation_threshold=params.get("correlation_threshold", 0.40),
        addon_multiplier=params.get("wwr_addon_multiplier", 1.20),
    )

    total_addon = conc["addon_inr"] + liq["addon_inr"] + wwr["addon_inr"]
    total_im = base_im + total_addon

    return {
        "base_im_inr": float(base_im),
        "total_im_inr": float(total_im),
        "addon_concentration_inr": float(conc["addon_inr"]),
        "addon_liquidity_inr": float(liq["addon_inr"]),
        "addon_wwr_inr": float(wwr["addon_inr"]),
        "total_addon_inr": float(total_addon),
        "is_concentrated": conc.get("is_concentrated", False),
        "is_illiquid": liq.get("is_illiquid", False),
        "is_wwr": wwr.get("is_wwr", False),
        "liq_days": conc.get("liq_days"),
        "liq_cost_inr": conc.get("liq_cost_inr", 0.0),
    }
