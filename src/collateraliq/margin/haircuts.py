"""
Bond repo haircut-based IM.

Haircuts are calibrated from yield-change history (99th percentile 10-day loss).
Applied to repo positions by bond maturity bucket.

Reference: Basel III LCR haircut schedule (proxy). 
Actual haircuts in bilateral repos are CSA-negotiated; these are calibrated values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


HAIRCUT_SCHEDULE = {
    "0-1yr":   0.005,   # 0.5%
    "1-3yr":   0.010,   # 1.0%
    "3-7yr":   0.020,   # 2.0%
    "7-10yr":  0.030,   # 3.0%
    "10yr+":   0.050,   # 5.0%
}


def get_haircut_bucket(maturity_years: float) -> str:
    """Return the maturity bucket label for a given tenor."""
    if maturity_years <= 1:
        return "0-1yr"
    elif maturity_years <= 3:
        return "1-3yr"
    elif maturity_years <= 7:
        return "3-7yr"
    elif maturity_years <= 10:
        return "7-10yr"
    else:
        return "10yr+"


def haircut_im(
    notional: float,
    maturity_years: float,
    custom_schedule: dict | None = None,
) -> dict:
    """
    Compute haircut-based IM for a repo position.

    IM = notional × haircut_for_maturity_bucket

    Parameters
    ----------
    notional        : repo notional (INR)
    maturity_years  : underlying bond maturity in years
    custom_schedule : override default haircut schedule
    """
    schedule = custom_schedule if custom_schedule is not None else HAIRCUT_SCHEDULE
    bucket = get_haircut_bucket(maturity_years)
    haircut = schedule.get(bucket, 0.03)
    im = notional * haircut

    return {
        "im_inr": float(im),
        "haircut": float(haircut),
        "bucket": bucket,
        "notional_inr": float(notional),
        "method": "haircut",
    }


def calibrate_haircuts_from_yield_history(
    yield_history: pd.DataFrame,
    confidence: float = 0.99,
    horizon_days: int = 10,
) -> dict:
    """
    Calibrate haircuts from yield change history.

    For each maturity bucket, computes the 99th percentile 10-day yield change
    and converts to price impact using modified duration approximation.

    Parameters
    ----------
    yield_history : DataFrame with columns as maturity tenors (e.g., '2Y', '5Y', '10Y')
                    and index as dates
    Returns dict of {bucket: calibrated_haircut}
    """
    calibrated = {}

    tenor_to_bucket = {
        "2Y": ("1-3yr", 2.0),
        "5Y": ("3-7yr", 5.0),
        "10Y": ("7-10yr", 10.0),
        "30Y": ("10yr+", 30.0),
    }

    for col in yield_history.columns:
        if col not in tenor_to_bucket:
            continue
        bucket, dur_approx = tenor_to_bucket[col]

        # Overlapping horizon-day yield changes
        changes = np.array(yield_history[col].dropna())
        horizon_changes = np.array([
            changes[i:i + horizon_days].sum()
            for i in range(len(changes) - horizon_days + 1)
        ])

        # 99th percentile yield rise (loss for long bond)
        worst_yield_rise = float(np.percentile(horizon_changes, confidence * 100))
        # Duration-approximated price loss
        haircut = dur_approx * worst_yield_rise * 0.01  # convert bps to decimal if needed

        calibrated[bucket] = max(float(abs(haircut)), HAIRCUT_SCHEDULE.get(bucket, 0.01))

    return calibrated
