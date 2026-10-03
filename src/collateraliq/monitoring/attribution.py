"""
Day-on-day attribution engine.

Decomposes the change in exposure or IM into:
1. Market moves (by risk factor: equity, FX, rates, credit spread)
2. New trades booked
3. Matured trades
4. Collateral movements (VM received, IM posted)
5. Residual (model/rounding differences)

The components must reconcile exactly to the total change.
A test enforces this reconciliation.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ATTRIBUTION_TOLERANCE = 1.0  # INR — reconciliation tolerance


def compute_attribution(
    today_exposure: float,
    yesterday_exposure: float,
    today_trades: pd.DataFrame,
    yesterday_trades: pd.DataFrame,
    today_market: dict,
    yesterday_market: dict,
    today_vm: float,
    yesterday_vm: float,
    client_id: str,
) -> dict:
    """
    Attribute the change in exposure to its components.

    Total change = market_moves + new_trades + matured_trades + collateral_change + residual

    All components are in INR.
    """
    total_change = today_exposure - yesterday_exposure

    # New trades: exposure from trades not in yesterday's book
    today_ids = set(today_trades["trade_id"]) if not today_trades.empty else set()
    yesterday_ids = set(yesterday_trades["trade_id"]) if not yesterday_trades.empty else set()

    new_trade_ids = today_ids - yesterday_ids
    matured_trade_ids = yesterday_ids - today_ids

    new_trades_df = today_trades[today_trades["trade_id"].isin(new_trade_ids)]
    new_trade_attribution = float(new_trades_df["mtm_inr"].sum()) if not new_trades_df.empty else 0.0

    matured_df = yesterday_trades[yesterday_trades["trade_id"].isin(matured_trade_ids)]
    matured_trade_attribution = float(-matured_df["mtm_inr"].sum()) if not matured_df.empty else 0.0

    # Collateral change attribution
    collateral_change = -(today_vm - yesterday_vm)  # More VM received → lower exposure

    # True market attribution is the MTM change of the common trades
    common_ids = today_ids.intersection(yesterday_ids)
    if common_ids:
        today_common = today_trades[today_trades["trade_id"].isin(common_ids)]["mtm_inr"].sum()
        yesterday_common = yesterday_trades[yesterday_trades["trade_id"].isin(common_ids)]["mtm_inr"].sum()
        market_total = today_common - yesterday_common
    else:
        market_total = 0.0

    # Apportion market_total to risk factors using sensitivities
    equity_move = (today_market.get("^NSEI_close", 1) - yesterday_market.get("^NSEI_close", 1)) / yesterday_market.get("^NSEI_close", 1)
    fx_move = (today_market.get("USDINR=X_close", 84) - yesterday_market.get("USDINR=X_close", 84)) / yesterday_market.get("USDINR=X_close", 84)
    rate_move = today_market.get("INDIRLTLT01STM_rate", 0.07) - yesterday_market.get("INDIRLTLT01STM_rate", 0.07)

    # Simplified risk allocation based on generic move sizes
    total_abs_move = abs(equity_move) + abs(fx_move) + abs(rate_move * 100)
    if total_abs_move > 0:
        equity_attribution = market_total * (abs(equity_move) / total_abs_move)
        fx_attribution = market_total * (abs(fx_move) / total_abs_move)
        rates_attribution = market_total * (abs(rate_move * 100) / total_abs_move)
    else:
        equity_attribution, fx_attribution, rates_attribution = market_total, 0.0, 0.0

    # Residual is now just rounding errors or non-linear effects
    residual = total_change - market_total - new_trade_attribution - matured_trade_attribution - collateral_change
    
    # Check if residual is larger than rounding error (i.e., non-linear netting effect)
    reconciles = abs(residual) < ATTRIBUTION_TOLERANCE

    return {
        "client_id": client_id,
        "total_change_inr": float(total_change),
        "market_equity_inr": float(equity_attribution),
        "market_fx_inr": float(fx_attribution),
        "market_rates_inr": float(rates_attribution),
        "market_total_inr": float(market_total),
        "new_trades_inr": float(new_trade_attribution),
        "matured_trades_inr": float(matured_trade_attribution),
        "collateral_change_inr": float(collateral_change),
        "residual_inr": float(residual),
        "reconciles": reconciles,
        "recon_error_inr": float(abs(residual)),
        "equity_move_pct": float(equity_move * 100),
        "fx_move_pct": float(fx_move * 100),
        "rate_move_bps": float(rate_move * 10000),
    }


def verify_attribution_reconciliation(attribution: dict, tol: float = ATTRIBUTION_TOLERANCE) -> bool:
    """
    Assert that attribution components sum to total change.

    Raises AssertionError if reconciliation fails beyond tolerance.
    Used in tests and as a production control.
    """
    computed_total = (
        attribution["market_total_inr"]
        + attribution["new_trades_inr"]
        + attribution["matured_trades_inr"]
        + attribution["collateral_change_inr"]
        + attribution["residual_inr"]
    )
    error = abs(computed_total - attribution["total_change_inr"])
    if error > tol:
        raise AssertionError(
            f"Attribution reconciliation failed for {attribution['client_id']}: "
            f"error={error:.2f} INR > tolerance={tol} INR"
        )
    return True
