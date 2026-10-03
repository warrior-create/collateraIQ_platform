"""
Unified valuation engine: price all products for a given date.

Imports pricers from RiskCore (equity, options, FX, bonds) and CollateralIQ's
own IRS and CDS pricers. Returns MTM and Greeks per trade.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

# RiskCore pricers (installed via pip install -e ../riskcore)
try:
    from marketrisk.pricing.bonds import bond_price, modified_duration, dv01
    from marketrisk.pricing.options import black76_price, black76_greeks, implied_vol_with_skew, nifty_forward
    from marketrisk.pricing.fx import fx_forward_rate, fx_forward_pnl
    RISKCORE_AVAILABLE = True
except ImportError:
    RISKCORE_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "RiskCore pricers not found. Using built-in fallbacks."
    )

# CollateralIQ pricers
from collateraliq.pricing.irs import price_irs, build_inr_curve
from collateraliq.pricing.cds import price_cds, hazard_rate_from_spread
from collateraliq.data.db import get_engine

logger = logging.getLogger(__name__)


def _bond_price_fallback(face, coupon_rate, ytm, maturity_years, frequency=2):
    """Fallback bond pricer if RiskCore not available."""
    n = int(round(maturity_years * frequency))
    if n == 0:
        return face
    coupon = face * coupon_rate / frequency
    y = ytm / frequency
    t = np.arange(1, n + 1)
    return float(np.sum(coupon / (1 + y) ** t) + face / (1 + y) ** n)


def _black76_fallback(F, K, T, r, sigma, option_type="call"):
    """Fallback Black-76 if RiskCore not available."""
    from scipy.stats import norm
    if T <= 0:
        if option_type == "call":
            return max(F - K, 0) * np.exp(-r * T)
        return max(K - F, 0) * np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if option_type == "call":
        return float(disc * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def _fx_forward_fallback(spot, r_inr, r_usd, tenor):
    return float(spot * np.exp((r_inr - r_usd) * tenor))


def value_trade(
    trade: dict,
    market: dict,
) -> dict:
    """
    Value a single trade given current market data.

    Parameters
    ----------
    trade  : dict with trade fields
    market : dict with market data {ticker: price, rates, vols, etc.}

    Returns
    -------
    dict: trade_id, mtm_inr, delta_inr, dv01_inr, vega_inr
    """
    product = trade["product_type"]
    params = json.loads(trade["params"]) if isinstance(trade["params"], str) else trade["params"]
    notional = trade["notional_inr"]
    direction = trade["direction"]

    try:
        if product == "equity_financing":
            return _value_equity_financing(trade, params, market, notional, direction)
        elif product == "gsec_repo":
            return _value_gsec_repo(trade, params, market, notional, direction)
        elif product == "irs":
            return _value_irs(trade, params, market, notional, direction)
        elif product == "cds_index":
            return _value_cds(trade, params, market, notional, direction)
        elif product in ("equity_options", "single_stock_options"):
            return _value_option(trade, params, market, notional, direction, product)
        elif product == "fx_forward":
            return _value_fx_forward(trade, params, market, notional, direction)
        else:
            logger.warning(f"Unknown product: {product}")
            return _zero_val(trade["trade_id"])
    except Exception as e:
        logger.error(f"Valuation error for {trade['trade_id']}: {e}")
        return _zero_val(trade["trade_id"])


def _value_equity_financing(trade, params, market, notional, direction):
    ticker = params.get("ticker", "^NSEI")
    price = market.get(f"{ticker}_close", market.get("^NSEI_close", 22000))
    repo_rate = params.get("repo_rate", 0.07)
    dt = 1 / 252
    # Daily accrual: repo interest on notional
    mtm = direction * notional * repo_rate * dt
    return {
        "trade_id": trade["trade_id"],
        "mtm_inr": float(mtm),
        "delta_inr": float(direction * notional),
        "gamma_inr": 0.0,
        "vega_inr": 0.0,
        "dv01_inr": 0.0,
    }


def _value_gsec_repo(trade, params, market, notional, direction):
    ytm = market.get("INDIRLTLT01STM_rate", 0.07)
    coupon = params.get("coupon_rate", 0.07)
    mat = params.get("bond_maturity_years", 5)

    if RISKCORE_AVAILABLE:
        price = bond_price(notional, coupon, ytm, mat)
        dur = modified_duration(notional, coupon, ytm, mat)
        dv01_val = dv01(notional, coupon, ytm, mat)
    else:
        price = _bond_price_fallback(notional, coupon, ytm, mat)
        dur = mat * 0.9  # approximation
        dv01_val = -dur * price * 0.0001

    mtm = direction * (price - notional)
    return {
        "trade_id": trade["trade_id"],
        "mtm_inr": float(mtm),
        "delta_inr": 0.0,
        "gamma_inr": 0.0,
        "vega_inr": 0.0,
        "dv01_inr": float(direction * dv01_val),
    }


def _value_irs(trade, params, market, notional, direction):
    r_short = market.get("SOFR_rate", 0.05)
    r_10y = market.get("INDIRLTLT01STM_rate", 0.07)
    curve = build_inr_curve(r_short, r_10y)

    fixed_rate = params.get("fixed_rate", 0.07)
    tenor = params.get("tenor_years", 5)
    is_payer = params.get("is_payer", True)

    result = price_irs(notional, fixed_rate, tenor, curve, is_payer)
    npv = result["npv"] * direction

    return {
        "trade_id": trade["trade_id"],
        "mtm_inr": float(npv),
        "delta_inr": 0.0,
        "gamma_inr": 0.0,
        "vega_inr": 0.0,
        "dv01_inr": float(result["dv01"] * direction),
    }


def _value_cds(trade, params, market, notional, direction):
    spread_bps = market.get("BAMLC0A0CM_spread", 0.01) * 10000
    r_free = market.get("SOFR_rate", 0.05)
    hazard = hazard_rate_from_spread(spread_bps)
    tenor = params.get("tenor_years", 5)
    is_buyer = params.get("is_protection_buyer", True)

    result = price_cds(notional, spread_bps, tenor, hazard, r_free,
                       is_protection_buyer=is_buyer)
    npv = result["npv"] * direction

    return {
        "trade_id": trade["trade_id"],
        "mtm_inr": float(npv),
        "delta_inr": 0.0,
        "gamma_inr": 0.0,
        "vega_inr": float(result["cs01"] * direction),
        "dv01_inr": 0.0,
    }


def _value_option(trade, params, market, notional, direction, product):
    ticker = params.get("ticker", "^NSEI")
    spot = market.get(f"{ticker}_close", market.get("^NSEI_close", 22000))
    vix = market.get("^INDIAVIX_close", 20.0)
    r = market.get("INDIRLTLT01STM_rate", 0.07)

    option_type = params.get("option_type", "call")
    moneyness = params.get("moneyness", 1.0)
    T = params.get("tenor_years", 0.083)
    contracts = params.get("contracts", 10)
    lot_size = params.get("lot_size", 50)
    K = spot * moneyness
    sigma = vix / 100

    F = spot * np.exp((r - 0.012) * T)  # Nifty forward

    if RISKCORE_AVAILABLE:
        sigma_adj = implied_vol_with_skew(sigma, F, K)
        greeks = black76_greeks(F, K, T, r, sigma_adj, option_type)
    else:
        sigma_adj = sigma
        price = _black76_fallback(F, K, T, r, sigma_adj, option_type)
        greeks = {"price": price, "delta": 0.5, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    mtm = direction * contracts * lot_size * greeks["price"]
    delta = direction * contracts * lot_size * greeks["delta"] * spot
    vega = direction * contracts * lot_size * greeks.get("vega", 0.0)

    return {
        "trade_id": trade["trade_id"],
        "mtm_inr": float(mtm),
        "delta_inr": float(delta),
        "gamma_inr": float(direction * contracts * lot_size * greeks.get("gamma", 0.0)),
        "vega_inr": float(vega),
        "dv01_inr": 0.0,
    }


def _value_fx_forward(trade, params, market, notional, direction):
    spot = market.get("USDINR=X_close", 84.0)
    r_inr = market.get("INDIRLTLT01STM_rate", 0.07)
    r_usd = market.get("SOFR_rate", 0.05)
    tenor = params.get("tenor_years", 0.25)
    n_usd = params.get("notional_usd", notional / 84)

    if RISKCORE_AVAILABLE:
        fwd_new = fx_forward_rate(spot, r_inr, r_usd, tenor)
    else:
        fwd_new = _fx_forward_fallback(spot, r_inr, r_usd, tenor)

    # MTM = notional_usd × (current_forward - contractual_forward) × direction
    contractual_fwd = params.get("contractual_forward", fwd_new * 1.002)
    mtm = direction * n_usd * (fwd_new - contractual_fwd)

    return {
        "trade_id": trade["trade_id"],
        "mtm_inr": float(mtm),
        "delta_inr": float(direction * n_usd),  # USDINR delta
        "gamma_inr": 0.0,
        "vega_inr": 0.0,
        "dv01_inr": 0.0,
    }


def _zero_val(trade_id: str) -> dict:
    return {
        "trade_id": trade_id,
        "mtm_inr": 0.0,
        "delta_inr": 0.0,
        "gamma_inr": 0.0,
        "vega_inr": 0.0,
        "dv01_inr": 0.0,
    }


def value_all_trades(
    trades_df: pd.DataFrame,
    market: dict,
    as_of: date,
    db_path: str = "data/collateraliq.db",
) -> pd.DataFrame:
    """
    Value all trades in the DataFrame and write to valuations table.

    Returns DataFrame of valuations.
    """
    results = []
    for _, trade in trades_df.iterrows():
        val = value_trade(trade.to_dict(), market)
        val["date"] = str(as_of)
        results.append(val)

    df = pd.DataFrame(results)

    # Write to DB
    engine = get_engine(db_path)
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO valuations
                        (date, trade_id, mtm_inr, delta_inr, gamma_inr, vega_inr, dv01_inr)
                    VALUES
                        (:date, :trade_id, :mtm_inr, :delta_inr, :gamma_inr, :vega_inr, :dv01_inr) ON CONFLICT DO NOTHING
                """),
                row.to_dict(),
            )

    return df
