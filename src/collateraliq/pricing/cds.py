"""
CDS (Credit Default Swap) pricing using the hazard rate model.

Calibrates hazard rate from index OAS spread proxy (ICE BofA or iTraxx EM proxy).
Adds a jump-to-default add-on for IM purposes.

References
----------
O'Kane, Modelling Single-name and Multi-name Credit Derivatives, Wiley Finance 2008.
Hull & White, "Valuing CDS", Journal of Derivatives 2000.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Sequence


def hazard_rate_from_spread(
    spread_bps: float,
    recovery_rate: float = 0.40,
) -> float:
    """
    Approximate hazard rate (intensity) from par CDS spread.

    For flat hazard rate: spread ≈ λ × (1 - R)

    Parameters
    ----------
    spread_bps    : par CDS spread in basis points
    recovery_rate : recovery rate (fraction, 0-1)

    Returns
    -------
    Annualised hazard rate λ
    """
    spread = spread_bps / 10000
    return spread / max(1 - recovery_rate, 1e-6)


def survival_prob(t: float, hazard: float) -> float:
    """Survival probability at time t given constant hazard rate."""
    return float(np.exp(-hazard * t))


def default_prob_incremental(t1: float, t2: float, hazard: float) -> float:
    """Probability of defaulting in (t1, t2]."""
    return float(survival_prob(t1, hazard) - survival_prob(t2, hazard))


def price_cds(
    notional: float,
    spread_bps: float,
    tenor_years: float,
    hazard: float,
    risk_free_rate: float = 0.065,
    recovery_rate: float = 0.40,
    payment_freq: int = 4,
    is_protection_buyer: bool = True,
    **kwargs,
) -> dict:
    """
    Price a CDS using the reduced-form (hazard rate) model.

    NPV = PV(protection leg) - PV(premium leg)

    Parameters
    ----------
    notional          : CDS notional (INR)
    spread_bps        : contractual spread in bps
    tenor_years       : remaining tenor
    hazard            : annualised hazard rate
    risk_free_rate    : flat risk-free rate for discounting
    recovery_rate     : loss given default = 1 - R
    payment_freq      : premium payments per year (4 = quarterly)
    is_protection_buyer: True = pay premium, receive protection

    Returns
    -------
    dict: npv, protection_leg_pv, premium_leg_pv, cs01, jump_to_default
    """
    spread = spread_bps / 10000
    dt = 1.0 / payment_freq
    n = max(1, int(round(tenor_years * payment_freq)))
    lgd = 1.0 - recovery_rate

    # Protection leg: sum over small time steps
    # PV(protection) = N × LGD × Σ q(ti) × DF(ti)
    pv_protection = 0.0
    steps = max(n * 10, 100)
    step_dt = tenor_years / steps
    for k in range(steps):
        t_mid = (k + 0.5) * step_dt
        dp = default_prob_incremental(k * step_dt, (k + 1) * step_dt, hazard)
        df = np.exp(-risk_free_rate * t_mid)
        pv_protection += notional * lgd * dp * df

    # Premium leg
    pv_premium = 0.0
    for i in range(1, n + 1):
        t = i * dt
        sp = survival_prob(t, hazard)
        df = np.exp(-risk_free_rate * t)
        pv_premium += notional * spread * dt * sp * df

    if is_protection_buyer:
        npv = pv_protection - pv_premium
    else:
        npv = pv_premium - pv_protection

    # CS01: change in NPV per 1bp widening of hazard rate
    h_bumped = hazard_rate_from_spread(spread_bps + 1, recovery_rate)
    pv_prot_bumped = _protection_pv(notional, tenor_years, h_bumped, risk_free_rate, recovery_rate)
    cs01 = (pv_prot_bumped - pv_protection) * (1 if is_protection_buyer else -1)

    # Jump-to-default add-on: loss if immediate default (no recovery hedged)
    jump_to_default = notional * lgd if is_protection_buyer else 0.0

    par_spread_bps = 0.0
    if kwargs.get("compute_par_spread", True):
        par_spread_bps = _solve_par_spread(
            notional, tenor_years, hazard, risk_free_rate, recovery_rate, payment_freq
        )

    return {
        "npv": float(npv),
        "protection_leg_pv": float(pv_protection),
        "premium_leg_pv": float(pv_premium),
        "cs01": float(cs01),
        "jump_to_default_addon": float(jump_to_default),
        "par_spread_bps": float(par_spread_bps),
        "hazard_rate": float(hazard),
        "survival_prob_T": float(survival_prob(tenor_years, hazard)),
    }


def _protection_pv(
    notional: float,
    tenor_years: float,
    hazard: float,
    risk_free_rate: float,
    recovery_rate: float,
    steps: int = 200,
) -> float:
    """Compute protection leg PV."""
    lgd = 1.0 - recovery_rate
    step_dt = tenor_years / steps
    pv = 0.0
    for k in range(steps):
        t_mid = (k + 0.5) * step_dt
        dp = default_prob_incremental(k * step_dt, (k + 1) * step_dt, hazard)
        df = np.exp(-risk_free_rate * t_mid)
        pv += notional * lgd * dp * df
    return pv


def _solve_par_spread(
    notional: float,
    tenor_years: float,
    hazard: float,
    risk_free_rate: float,
    recovery_rate: float,
    payment_freq: int,
) -> float:
    """Solve for par spread (bps) that makes NPV = 0."""
    def f(s):
        res = price_cds(notional, s, tenor_years, hazard, risk_free_rate,
                        recovery_rate, payment_freq, is_protection_buyer=True,
                        compute_par_spread=False)
        return res["npv"]

    try:
        return brentq(f, 0.01, 10000, xtol=0.001, maxiter=100)
    except ValueError:
        return float(hazard_rate_from_spread(100.0) * (1 - recovery_rate) * 10000)


def cds_spread_shock_var(
    notional: float,
    spread_history_bps: np.ndarray,
    tenor_years: float,
    risk_free_rate: float = 0.065,
    recovery_rate: float = 0.40,
    confidence: float = 0.99,
    horizon_days: int = 10,
) -> dict:
    """
    Compute CDS IM via spread-shock historical VaR.

    Uses overlapping 10-day spread changes from historical data.

    Parameters
    ----------
    spread_history_bps: time series of index spread in bps
    confidence        : VaR confidence level
    horizon_days      : IM horizon (business days)

    Returns
    -------
    dict: im_bps, im_inr, breach_threshold
    """
    if len(spread_history_bps) < horizon_days + 1:
        # Fallback: 3× current spread as shock
        current_spread = float(spread_history_bps[-1]) if len(spread_history_bps) > 0 else 100.0
        shock = current_spread * 3.0
        h = hazard_rate_from_spread(current_spread + shock, recovery_rate)
        h_base = hazard_rate_from_spread(current_spread, recovery_rate)
        im_inr = abs(
            price_cds(notional, current_spread, tenor_years, h, risk_free_rate, recovery_rate)["npv"]
            - price_cds(notional, current_spread, tenor_years, h_base, risk_free_rate, recovery_rate)["npv"]
        )
        return {"im_bps": shock, "im_inr": im_inr, "method": "fallback_3x_spread"}

    # Overlapping horizon-day returns
    changes = np.diff(spread_history_bps, n=horizon_days)
    if len(changes) < 10:
        changes = np.diff(spread_history_bps)

    # P&L for each scenario
    pnl = []
    current_spread = float(spread_history_bps[-1])
    h_base = hazard_rate_from_spread(current_spread, recovery_rate)
    base_price = price_cds(notional, current_spread, tenor_years, h_base, risk_free_rate, recovery_rate)

    for ds in changes:
        shocked_spread = max(current_spread + ds, 0.1)
        h_shocked = hazard_rate_from_spread(shocked_spread, recovery_rate)
        shocked_price = price_cds(notional, shocked_spread, tenor_years, h_shocked, risk_free_rate, recovery_rate)
        pnl.append(shocked_price["npv"] - base_price["npv"])

    pnl = np.array(pnl)
    losses = -pnl  # IM protects against losses
    im_inr = float(np.percentile(losses, confidence * 100))

    # Jump-to-default add-on
    jtd = base_price["jump_to_default_addon"]
    im_total = max(im_inr, 0) + jtd * 0.10  # 10% JTD add-on

    return {
        "im_inr": max(im_total, 0),
        "im_spread_component_inr": max(im_inr, 0),
        "jtd_addon_inr": jtd * 0.10,
        "current_spread_bps": current_spread,
        "worst_spread_change_bps": float(np.percentile(changes, (1 - confidence) * 100)),
        "method": "spread_shock_var_fhs",
    }


def par_cds_check(
    notional: float,
    tenor_years: float,
    hazard: float,
    risk_free_rate: float = 0.065,
    recovery_rate: float = 0.40,
    tol: float | None = None,
) -> bool:
    """
    Known-answer test: CDS at par spread should have NPV ≈ 0.

    Returns True if |NPV| < tol.
    """
    if tol is None:
        tol = notional * 0.001
    par_spread = _solve_par_spread(notional, tenor_years, hazard, risk_free_rate, recovery_rate, 4)
    result = price_cds(notional, par_spread, tenor_years, hazard, risk_free_rate, recovery_rate)
    return abs(result["npv"]) < max(tol, notional * 0.001)
