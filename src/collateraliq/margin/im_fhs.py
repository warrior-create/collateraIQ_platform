"""
Filtered Historical Simulation (FHS) for Initial Margin.

Method: Historical/FHS at 99% confidence, 10-day horizon.
MPOR scaling: compare √time vs direct overlapping 10-day returns.

Used for: equity financing, IRS (curve-factor P&L), FX forwards.

References
----------
Barone-Adesi, Giannopoulos, Vosper, "VaR without Correlations for Nonlinear Portfolios"
J. of Futures Markets 1999.
EMIR Article 24 — Confidence level and horizon for initial margin.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError:
    # Fallback if numba is not installed
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper

logger = logging.getLogger(__name__)


def compute_fhs_im(
    pnl_series: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 10,
    lookback_days: int = 250,
    method: str = "overlapping",
) -> dict:
    """
    Compute IM via Historical/FHS at given confidence and horizon.

    Parameters
    ----------
    pnl_series    : 1D array of daily P&L (in INR)
    confidence    : VaR confidence level (e.g. 0.99)
    horizon_days  : IM horizon in business days
    lookback_days : historical window
    method        : 'overlapping' (direct 10-day) or 'sqrt_time' (1-day scaled)

    Returns
    -------
    dict: im_inr, method, horizon_days, confidence, n_observations, worst_loss
    """
    if len(pnl_series) < 2:
        return {"im_inr": 0.0, "method": method, "n_observations": 0, "worst_loss": 0.0}

    # Use most recent lookback_days
    series = np.array(pnl_series[-lookback_days:])

    if method == "overlapping":
        return _fhs_overlapping(series, confidence, horizon_days)
    elif method == "sqrt_time":
        return _fhs_sqrt_time(series, confidence, horizon_days)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'overlapping' or 'sqrt_time'.")


@njit
def fast_overlapping_sum(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n - window + 1)
    # Moving window sum
    s = np.sum(arr[:window])
    out[0] = s
    for i in range(1, n - window + 1):
        s += arr[i + window - 1] - arr[i - 1]
        out[i] = s
    return out

def _fhs_overlapping(
    daily_pnl: np.ndarray,
    confidence: float,
    horizon: int,
) -> dict:
    """
    Direct overlapping horizon-day returns.
    """
    if len(daily_pnl) < horizon:
        # Fall back to square-root time scaling
        return _fhs_sqrt_time(daily_pnl, confidence, horizon)

    # Fast JIT overlapping sums (Performance/Numba Addition)
    horizon_pnl = fast_overlapping_sum(daily_pnl, horizon)

    losses = -horizon_pnl  # Positive loss = bad for us
    im = float(np.percentile(losses, confidence * 100))

    return {
        "im_inr": max(im, 0.0),
        "method": "overlapping",
        "horizon_days": horizon,
        "confidence": confidence,
        "n_observations": len(horizon_pnl),
        "worst_loss": float(losses.max()),
        "mean_loss": float(losses.mean()),
        "percentile_99": float(np.percentile(losses, 99)),
        "percentile_95": float(np.percentile(losses, 95)),
    }


def _fhs_sqrt_time(
    daily_pnl: np.ndarray,
    confidence: float,
    horizon: int,
) -> dict:
    """
    Square-root-of-time scaling from 1-day to horizon-day VaR.

    Assumes i.i.d. daily P&L (known limitation: understates IM in trending markets).
    """
    losses_1d = -daily_pnl
    var_1d = float(np.percentile(losses_1d, confidence * 100))
    im = var_1d * np.sqrt(horizon)

    return {
        "im_inr": max(im, 0.0),
        "method": "sqrt_time",
        "horizon_days": horizon,
        "confidence": confidence,
        "n_observations": len(daily_pnl),
        "worst_loss": float(losses_1d.max()),
        "var_1d": float(var_1d),
    }


def compare_mpor_scaling(
    daily_pnl: np.ndarray,
    horizons: list[int] | None = None,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """
    Compare √time scaling vs direct overlapping returns across horizons.

    Returns DataFrame with columns: horizon, im_overlapping, im_sqrt_time, ratio
    """
    if horizons is None:
        horizons = [1, 2, 5, 10, 20]

    results = []
    for h in horizons:
        overlap = _fhs_overlapping(daily_pnl, confidence, h)
        sqrt_t = _fhs_sqrt_time(daily_pnl, confidence, h)
        results.append({
            "horizon_days": h,
            "im_overlapping": overlap["im_inr"],
            "im_sqrt_time": sqrt_t["im_inr"],
            "ratio": overlap["im_inr"] / sqrt_t["im_inr"] if sqrt_t["im_inr"] > 0 else np.nan,
        })

    return pd.DataFrame(results)


def compute_irs_im_fhs(
    dv01_history: np.ndarray,
    rate_changes_history: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 10,
    lookback_days: int = 250,
) -> dict:
    """
    IRS IM via FHS on curve-factor P&L: P&L = DV01 × Δy.

    Parameters
    ----------
    dv01_history        : daily DV01 values (INR per bp)
    rate_changes_history: daily yield changes in bps
    """
    pnl = dv01_history * rate_changes_history  # INR P&L
    return compute_fhs_im(pnl, confidence, horizon_days, lookback_days)


def blended_stressed_im(
    current_im: float,
    stressed_im: float,
    stress_weight: float = 0.25,
) -> float:
    """
    Blended IM = (1-w) × current_IM + w × stressed_IM.

    Per EMIR-style procyclicality controls:
    stressed_IM is calibrated over a stress period (e.g. 2020 March–June).
    """
    return float((1 - stress_weight) * current_im + stress_weight * stressed_im)
