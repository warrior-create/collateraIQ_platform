"""
SIMM-lite (simplified SIMM-style delta margin) for IRS.

Used as a challenger to FHS IM for IRS.
Blended weight: 30% SIMM-lite, 70% FHS (configurable).

Reference: ISDA SIMM v2.6 methodology (simplified implementation).
This is not a full SIMM implementation; it uses SIMM-style risk bucketing
and risk weights as a challenger model only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# SIMM v2.6 risk weights for interest rates (INR, simplified)
# Source: ISDA SIMM Methodology v2.6 (2023)
# Units: basis points per year of DV01
SIMM_RISK_WEIGHTS_IR = {
    "0.25Y": 16,   # bps
    "0.5Y":  15,
    "1Y":    14,
    "2Y":    13,
    "3Y":    12,
    "5Y":    11,
    "7Y":    11,
    "10Y":   12,
    "15Y":   12,
    "20Y":   13,
    "30Y":   13,
}

# SIMM historical vol correlation matrix (simplified, symmetric)
# Between adjacent tenor buckets
SIMM_TENOR_CORRELATION = 0.98  # Adjacent tenors highly correlated


def compute_simm_lite_im(
    dv01_buckets: dict[str, float],
    currency: str = "INR",
    confidence_scalar: float = 1.65,  # 1.65σ ≈ 95%; SIMM targets specific quantile
) -> dict:
    """
    Compute SIMM-lite delta margin for IRS.

    Risk = Σ_i (DV01_i × RW_i) [per bucket]
    Margin = sqrt(Σ_i Σ_j ρ_ij × Risk_i × Risk_j)

    Parameters
    ----------
    dv01_buckets : dict of {tenor: DV01_INR}
    currency     : currency for risk weights
    confidence_scalar: scalar to convert 1σ to target confidence

    Returns
    -------
    dict: im_inr, risk_by_bucket, delta_margin
    """
    tenors = list(SIMM_RISK_WEIGHTS_IR.keys())
    n = len(tenors)

    # Map dv01_buckets to SIMM tenors
    risk = np.zeros(n)
    for i, tenor in enumerate(tenors):
        tenor_key = f"DV01_{tenor.replace('Y', 'Y')}"  # e.g. DV01_5Y
        # Try to match
        for key, val in dv01_buckets.items():
            if tenor.replace("Y", "") in key:
                rw = SIMM_RISK_WEIGHTS_IR[tenor] / 10000  # bps → decimal
                risk[i] = val * rw  # INR risk
                break

    # Correlation matrix: ρ_ij = max(correlation^|i-j|, 0.40)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = max(SIMM_TENOR_CORRELATION ** abs(i - j), 0.40)
    np.fill_diagonal(corr, 1.0)

    # Portfolio variance
    portfolio_variance = risk @ corr @ risk
    portfolio_sigma = np.sqrt(max(portfolio_variance, 0))

    # Delta margin
    delta_margin = confidence_scalar * portfolio_sigma

    return {
        "im_inr": float(delta_margin),
        "risk_by_bucket": {t: float(r) for t, r in zip(tenors, risk)},
        "portfolio_sigma": float(portfolio_sigma),
        "confidence_scalar": confidence_scalar,
        "method": "simm_lite",
        "currency": currency,
    }


def blend_fhs_simm(
    fhs_im: float,
    simm_im: float,
    simm_weight: float = 0.30,
) -> float:
    """
    Blended IRS IM: (1-w) × FHS + w × SIMM-lite.

    This provides a challenger that combines two orthogonal approaches.
    """
    return float((1 - simm_weight) * fhs_im + simm_weight * simm_im)
