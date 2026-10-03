"""
Procyclicality buffer and stress floor for IM.

EMIR-style procyclicality controls:
- Buffer: add 25% buffer on top of base IM (always held, released when stressed)
- Floor: IM cannot fall below 75% of the stressed-period IM
- Blend: 75% current calibration + 25% stressed calibration

Reference: EMIR RTS on CCPs (Commission Delegated Regulation EU 153/2013),
Article 28 — Procyclical margin requirements.
Exact parameter values used here are illustrative proxies of the EMIR standard.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProcyclicalityBuffer:
    """
    Manages IM procyclicality controls.

    Three approaches (can be run in parallel and compared):
    1. Buffer add-on: IM_total = IM_base × (1 + buffer_fraction)
    2. Floor: IM_total = max(IM_base, floor_fraction × IM_stressed)
    3. Blended: IM_total = (1-w) × IM_current + w × IM_stressed
    """

    def __init__(
        self,
        buffer_fraction: float = 0.25,
        floor_fraction: float = 0.75,
        stress_weight: float = 0.25,
    ) -> None:
        self.buffer_fraction = buffer_fraction
        self.floor_fraction = floor_fraction
        self.stress_weight = stress_weight

    def apply_buffer(self, base_im: float) -> float:
        """Add buffer on top of base IM."""
        return float(base_im * (1 + self.buffer_fraction))

    def apply_floor(self, base_im: float, stressed_im: float) -> float:
        """Apply floor: IM ≥ floor_fraction × stressed_IM."""
        return float(max(base_im, self.floor_fraction * stressed_im))

    def apply_blend(self, current_im: float, stressed_im: float) -> float:
        """Blended calibration."""
        return float((1 - self.stress_weight) * current_im + self.stress_weight * stressed_im)

    def compare_approaches(
        self, base_im: float, stressed_im: float
    ) -> dict:
        """Return all three approaches for comparison."""
        return {
            "base_im": base_im,
            "stressed_im": stressed_im,
            "with_buffer": self.apply_buffer(base_im),
            "with_floor": self.apply_floor(base_im, stressed_im),
            "blended": self.apply_blend(base_im, stressed_im),
            "buffer_fraction": self.buffer_fraction,
            "floor_fraction": self.floor_fraction,
            "stress_weight": self.stress_weight,
        }


def compute_stress_period_im(
    pnl_series: np.ndarray,
    dates: pd.DatetimeIndex,
    confidence: float = 0.99,
    horizon_days: int = 10,
    stress_periods: list[tuple] | None = None,
) -> dict:
    """
    Compute IM calibrated over historical stress periods.

    Default stress periods: March 2020 (COVID), Feb-March 2022 (rate shock).

    Returns stressed IM for use as the floor/blend component.
    """
    if stress_periods is None:
        stress_periods = [
            ("2020-02-01", "2020-05-31", "COVID_2020"),
            ("2022-01-01", "2022-06-30", "Rate_Shock_2022"),
        ]

    stressed_results = {}

    for start_str, end_str, label in stress_periods:
        try:
            mask = (dates >= pd.Timestamp(start_str)) & (dates <= pd.Timestamp(end_str))
            stress_pnl = pnl_series[mask.values] if hasattr(mask, "values") else pnl_series[mask]

            if len(stress_pnl) < 5:
                continue

            # IM from stress period
            losses = -stress_pnl
            im = float(np.percentile(losses, confidence * 100))

            stressed_results[label] = {
                "im_inr": max(im, 0),
                "n_days": int(len(stress_pnl)),
                "max_loss": float(losses.max()),
                "mean_loss": float(losses.mean()),
            }
        except Exception as e:
            logger.warning(f"Stress period {label} failed: {e}")

    # Use the worst stressed IM
    if stressed_results:
        worst_label = max(stressed_results, key=lambda k: stressed_results[k]["im_inr"])
        stressed_im = stressed_results[worst_label]["im_inr"]
    else:
        stressed_im = 0.0

    return {
        "stressed_im_inr": stressed_im,
        "stress_periods": stressed_results,
        "worst_stress_period": worst_label if stressed_results else None,
    }


def procyclicality_study(
    im_time_series: pd.Series,
    dates: pd.DatetimeIndex,
    stressed_im_time_series: pd.Series | None = None,
    buffer_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    Study IM time series around stress events.

    Returns DataFrame with columns:
    date, im_no_buffer, im_with_buffer, im_jump_pct_no_buffer, im_jump_pct_with_buffer
    """
    df = pd.DataFrame({
        "date": dates,
        "im_no_buffer": im_time_series.values,
    })
    df["im_with_buffer"] = df["im_no_buffer"] * (1 + buffer_fraction)

    if stressed_im_time_series is not None:
        df["im_stressed"] = stressed_im_time_series.values
        buf = ProcyclicalityBuffer(buffer_fraction=buffer_fraction)
        df["im_blended"] = df.apply(
            lambda r: buf.apply_blend(r["im_no_buffer"], r["im_stressed"]), axis=1
        )

    # Compute day-on-day jumps
    df["im_jump_pct"] = df["im_no_buffer"].pct_change() * 100
    df["im_buffered_jump_pct"] = df["im_with_buffer"].pct_change() * 100

    # Flag stress events (>20% jump)
    df["is_stress_event"] = df["im_jump_pct"].abs() > 20

    return df.set_index("date")
