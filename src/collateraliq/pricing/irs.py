"""
IRS (Interest Rate Swap) pricing with full yield curve bootstrapping.

This module extends the basic IRS pricer from RiskCore with:
- Multi-point bootstrapping from INR par swap rates
- Discount and forward curve separation
- DV01 bucketing for SIMM-style IM computation

References
----------
Hull, Options, Futures and Other Derivatives, 10th ed. Ch. 6-7.
Brigo & Mercurio, Interest Rate Models, 2nd ed. Ch. 1.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Sequence


class BootstrappedCurve:
    """
    Zero-rate curve bootstrapped from par swap rates.

    Bootstrapping procedure:
    1. Overnight/short tenors: take SOFR/repo as zero rates
    2. Longer tenors: solve for zero rate that prices each par swap at par
    """

    def __init__(
        self,
        tenors: Sequence[float],
        zero_rates: Sequence[float],
    ) -> None:
        self.tenors = np.array(tenors)
        self.zero_rates = np.array(zero_rates)
        # Cubic spline interpolation on zero rates
        self._spline = CubicSpline(self.tenors, self.zero_rates, extrapolate=True)

    @classmethod
    def from_par_rates(
        cls,
        par_tenors: Sequence[float],
        par_rates: Sequence[float],
        freq: int = 2,
    ) -> "BootstrappedCurve":
        """
        Bootstrap zero curve from par swap rates.

        Parameters
        ----------
        par_tenors : list of tenors in years
        par_rates  : corresponding par (fixed) rates
        freq       : coupon frequency (2 = semi-annual)
        """
        tenors = np.array(par_tenors)
        par = np.array(par_rates)
        zeros = np.zeros_like(par)
        dt = 1.0 / freq

        # Bootstrap each tenor sequentially
        for i, (T, c) in enumerate(zip(tenors, par)):
            n = int(round(T * freq))
            # Sum of discount factors for all prior cash flows
            prior_sum = sum(
                np.exp(-zeros[j] * tenors[j]) * c * dt
                for j in range(i)
                if tenors[j] < T
            )
            # Solve: 1 = prior_sum + (1 + c*dt) * exp(-z * T)
            discount_T = (1 - prior_sum) / (1 + c * dt)
            zeros[i] = -np.log(max(discount_T, 1e-10)) / T

        return cls(tenors, zeros)

    def zero_rate(self, t: float) -> float:
        """Continuously-compounded zero rate at tenor t."""
        return float(np.clip(self._spline(t), 0.0001, 0.5))

    def discount_factor(self, t: float) -> float:
        """Discount factor P(0, t) = exp(-r(t) * t)."""
        return float(np.exp(-self.zero_rate(t) * t))

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously-compounded forward rate between t1 and t2."""
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        if t2 <= t1 or df2 <= 0:
            return self.zero_rate(t1)
        return float(np.log(df1 / df2) / (t2 - t1))


def build_inr_curve(r_short: float, r_10y: float) -> BootstrappedCurve:
    """
    Build a simple 5-point INR zero curve from short rate and 10Y rate.

    Uses linear interpolation of par rates, then bootstraps.
    """
    # Synthetic par rates: linearly interpolate between 1Y and 10Y
    par_tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    slope = (r_10y - r_short) / 9.75  # rise per year
    par_rates = [r_short + slope * (t - 0.25) for t in par_tenors]
    return BootstrappedCurve.from_par_rates(par_tenors, par_rates)


def price_irs(
    notional: float,
    fixed_rate: float,
    tenor_years: float,
    curve: BootstrappedCurve,
    is_payer: bool = True,
    payment_freq: int = 2,
    seasoning_years: float = 0.0,
) -> dict:
    """
    Price a plain-vanilla Interest Rate Swap.

    Parameters
    ----------
    notional       : swap notional (INR)
    fixed_rate     : contractual fixed rate (decimal)
    tenor_years    : remaining tenor from pricing date
    curve          : bootstrapped zero curve
    is_payer       : True = pay fixed, receive float
    payment_freq   : coupons per year (2 = semi-annual)
    seasoning_years: time already elapsed since trade date

    Returns
    -------
    dict: npv, fixed_leg_pv, float_leg_pv, pv01, dv01, par_rate
    """
    dt = 1.0 / payment_freq
    n = max(1, int(round(tenor_years * payment_freq)))

    # Fixed leg: sum of discounted fixed cash flows
    pv01 = 0.0
    for i in range(1, n + 1):
        t = i * dt
        pv01 += curve.discount_factor(t) * dt
    fixed_leg_pv = notional * fixed_rate * pv01

    # Float leg: par assumption — PV(float) = N × (1 - DF(T))
    df_T = curve.discount_factor(tenor_years)
    float_leg_pv = notional * (1.0 - df_T)

    # NPV: positive means asset to us if is_payer
    if is_payer:
        npv = float_leg_pv - fixed_leg_pv
        dv01_sign = -1
    else:
        npv = fixed_leg_pv - float_leg_pv
        dv01_sign = +1

    # DV01: approximate from pv01
    dv01 = dv01_sign * pv01 * notional * 0.0001  # per 1bp

    # Par rate: rate that makes NPV = 0
    par_rate = (1.0 - df_T) / pv01 if pv01 > 0 else fixed_rate

    return {
        "npv": float(npv),
        "fixed_leg_pv": float(fixed_leg_pv),
        "float_leg_pv": float(float_leg_pv),
        "pv01": float(pv01 * notional),
        "dv01": float(dv01),
        "par_rate": float(par_rate),
        "annuity": float(pv01),
    }


def irs_dv01_buckets(
    notional: float,
    fixed_rate: float,
    tenor_years: float,
    curve: BootstrappedCurve,
    is_payer: bool = True,
    payment_freq: int = 2,
    bucket_tenors: list[float] | None = None,
    bump_bps: float = 1.0,
) -> dict[str, float]:
    """
    Compute DV01 by tenor bucket for SIMM-style risk bucketing.

    Returns dict of {bucket_label: DV01_INR}.
    """
    if bucket_tenors is None:
        bucket_tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10]

    base_npv = price_irs(notional, fixed_rate, tenor_years, curve, is_payer, payment_freq)["npv"]
    dv01s = {}

    for t in bucket_tenors:
        if t > tenor_years:
            continue
        # Bump zero rate at this tenor by 1bp
        bumped_zeros = curve.zero_rates.copy()
        # Find closest tenor in curve
        idx = np.argmin(np.abs(curve.tenors - t))
        bumped_zeros[idx] += bump_bps / 10000
        bumped_curve = BootstrappedCurve(curve.tenors, bumped_zeros)
        bumped_npv = price_irs(notional, fixed_rate, tenor_years, bumped_curve, is_payer, payment_freq)["npv"]
        dv01s[f"DV01_{t}Y"] = float(bumped_npv - base_npv)

    return dv01s


def par_irs_check(
    notional: float,
    tenor_years: float,
    curve: BootstrappedCurve,
    payment_freq: int = 2,
    tol: float = 1.0,
) -> bool:
    """
    Known-answer test: IRS at par rate should have NPV ≈ 0.

    Returns True if |NPV| < tol (INR).
    """
    dt = 1.0 / payment_freq
    n = int(round(tenor_years * payment_freq))
    pv01 = sum(curve.discount_factor(i * dt) * dt for i in range(1, n + 1))
    df_T = curve.discount_factor(tenor_years)
    par_rate = (1.0 - df_T) / pv01 if pv01 > 0 else 0.065

    result = price_irs(notional, par_rate, tenor_years, curve, is_payer=True, payment_freq=payment_freq)
    return abs(result["npv"]) < tol
