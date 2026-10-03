"""
SPAN-style scenario grid for equity options IM.

Method: compute P&L for a grid of (spot shock, vol shock) scenarios.
IM = worst scenario loss across the grid.
Short option minimum applied.

References
----------
CME SPAN Methodology, 2020.
NSE SPAN-like exchange margin calculation (Indian market reference).
"""

from __future__ import annotations

import logging
from itertools import product

import numpy as np

logger = logging.getLogger(__name__)

try:
    from marketrisk.pricing.options import black76_price, implied_vol_with_skew, nifty_forward
    _RISKCORE = True
except ImportError:
    _RISKCORE = False


def _black76_local(F, K, T, r, sigma, option_type="call"):
    from scipy.stats import norm
    if T <= 0:
        return max(F - K, 0) if option_type == "call" else max(K - F, 0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if option_type == "call":
        return float(disc * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def compute_scenario_grid_im(
    options: list[dict],
    spot: float,
    sigma_atm: float,
    r: float,
    spot_shocks: list[float] | None = None,
    vol_shocks: list[float] | None = None,
    short_option_minimum_pct: float = 0.03,
) -> dict:
    """
    Compute SPAN-style IM for an options portfolio.

    Parameters
    ----------
    options : list of dicts with keys: option_type, K, T, contracts, lot_size, direction
    spot    : current underlying price
    sigma_atm : ATM implied vol (e.g. VIX/100)
    r       : risk-free rate
    spot_shocks : fractional spot shocks (e.g. [-0.30, ..., 0.30])
    vol_shocks  : fractional vol shocks (e.g. [-0.40, ..., 0.40])
    short_option_minimum_pct: minimum IM as % of notional for short options

    Returns
    -------
    dict: im_inr, worst_scenario, scenario_matrix
    """
    if spot_shocks is None:
        spot_shocks = [-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    if vol_shocks is None:
        vol_shocks = [-0.40, -0.20, 0.0, 0.20, 0.40]

    if not options:
        return {"im_inr": 0.0, "worst_scenario": None, "scenario_matrix": []}

    # Current portfolio value
    base_pv = _portfolio_pv(options, spot, sigma_atm, r)

    # Scenario P&L matrix
    scenario_results = []
    worst_loss = 0.0
    worst_scenario = None

    for ss, vs in product(spot_shocks, vol_shocks):
        spot_s = spot * (1 + ss)
        sigma_s = max(sigma_atm * (1 + vs), 0.01)

        scenario_pv = _portfolio_pv(options, spot_s, sigma_s, r)
        scenario_pnl = scenario_pv - base_pv
        scenario_loss = -scenario_pnl  # Loss to us

        scenario_results.append({
            "spot_shock": ss,
            "vol_shock": vs,
            "pnl": scenario_pnl,
            "loss": scenario_loss,
        })

        if scenario_loss > worst_loss:
            worst_loss = scenario_loss
            worst_scenario = {"spot_shock": ss, "vol_shock": vs, "loss": scenario_loss}

    # Short option minimum: 3% of notional for net short positions
    som = _short_option_minimum(options, spot, short_option_minimum_pct)

    im = max(worst_loss, som)

    return {
        "im_inr": float(im),
        "im_scenario_component": float(worst_loss),
        "im_short_option_minimum": float(som),
        "worst_scenario": worst_scenario,
        "n_scenarios": len(scenario_results),
        "base_pv": float(base_pv),
    }


def _portfolio_pv(options: list[dict], spot: float, sigma_atm: float, r: float) -> float:
    """Compute portfolio value across all options at given market conditions."""
    total = 0.0
    for opt in options:
        K = opt.get("K", spot)
        T = max(opt.get("T", 0.083), 1e-6)
        otype = opt.get("option_type", "call")
        contracts = opt.get("contracts", 1)
        lot_size = opt.get("lot_size", 50)
        direction = opt.get("direction", 1)  # 1=long, -1=short

        # Forward
        div_yield = 0.012
        F = spot * np.exp((r - div_yield) * T)

        # Skew adjustment
        if _RISKCORE:
            sigma = implied_vol_with_skew(sigma_atm, F, K)
            price = black76_price(F, K, T, r, sigma, otype)
        else:
            sigma = sigma_atm
            price = _black76_local(F, K, T, r, sigma, otype)

        total += direction * contracts * lot_size * price

    return total


def _short_option_minimum(
    options: list[dict],
    spot: float,
    minimum_pct: float,
) -> float:
    """
    Short option minimum: applied when portfolio is net short options.

    SOM = minimum_pct × sum of |notional| for net short positions.
    """
    net_short_notional = sum(
        abs(opt.get("contracts", 1) * opt.get("lot_size", 50) * spot)
        for opt in options
        if opt.get("direction", 1) < 0  # net short
    )
    return float(minimum_pct * net_short_notional)


def nsm_span_comparison(
    options: list[dict],
    spot: float,
    sigma_atm: float,
    r: float,
) -> dict:
    """
    Compare CollateralIQ scenario IM vs NSE SPAN-like exchange margin.

    NSE SPAN (proxy): uses a smaller set of scenarios aligned with NSE VaR margin.
    This demonstrates the difference between CEM desk IM and exchange margin.
    """
    # CollateralIQ full grid
    cem_im = compute_scenario_grid_im(options, spot, sigma_atm, r)

    # NSE SPAN proxy: 3-sigma spot shock ±15%, vol shock ±25%
    nse_spot_shocks = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]
    nse_vol_shocks = [-0.25, 0, 0.25]
    nse_im = compute_scenario_grid_im(
        options, spot, sigma_atm, r, nse_spot_shocks, nse_vol_shocks
    )

    return {
        "cem_im_inr": cem_im["im_inr"],
        "nse_span_proxy_im_inr": nse_im["im_inr"],
        "ratio": cem_im["im_inr"] / nse_im["im_inr"] if nse_im["im_inr"] > 0 else np.nan,
        "cem_worst_scenario": cem_im["worst_scenario"],
        "nse_worst_scenario": nse_im["worst_scenario"],
    }
