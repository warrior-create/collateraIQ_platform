"""
IM adequacy backtesting.

Compares realised MPOR losses against IM held.
Reports:
- Breach rate (% of days IM was insufficient)
- Kupiec test (binomial test of breach rate)
- Breach severity (average shortfall / IM)

Reference for Kupiec test: Kupiec (1995), "Techniques for Verifying the 
Accuracy of Risk Measurement Models", Journal of Derivatives.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

from collateraliq.data.db import get_engine

logger = logging.getLogger(__name__)


def kupiec_test(
    n_obs: int,
    n_breaches: int,
    confidence: float = 0.99,
    alpha: float = 0.05,
) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.

    H0: breach rate p0 = 1 - confidence
    Reject H0 if the observed breach rate is significantly different.

    Test statistic: LR = -2 × ln[(1-p0)^(n-x) × p0^x / ((1-x/n)^(n-x) × (x/n)^x)]
    Under H0: LR ~ χ²(1)

    Parameters
    ----------
    n_obs       : total observations
    n_breaches  : number of breaches
    confidence  : IM confidence level
    alpha       : significance level for rejection

    Returns
    -------
    dict: test_stat, p_value, reject_h0, expected_breaches, observed_rate
    """
    p0 = 1 - confidence
    if n_obs == 0:
        return {"test_stat": np.nan, "p_value": np.nan, "reject_h0": False}

    x = n_breaches
    n = n_obs
    p_hat = x / n

    expected_breaches = p0 * n

    # LR test statistic
    if x == 0:
        lr = -2 * n * np.log(1 - p0)
    elif x == n:
        lr = -2 * n * np.log(p0)
    else:
        try:
            lr = -2 * (
                (n - x) * np.log((1 - p0) / (1 - p_hat))
                + x * np.log(p0 / p_hat)
            )
        except (ValueError, ZeroDivisionError):
            lr = np.nan

    p_value = float(1 - stats.chi2.cdf(abs(lr), df=1)) if not np.isnan(lr) else np.nan
    reject = p_value < alpha if not np.isnan(p_value) else False

    return {
        "test_stat": float(lr) if not np.isnan(lr) else None,
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "reject_h0": bool(reject),
        "n_obs": n_obs,
        "n_breaches": n_breaches,
        "expected_breaches": float(expected_breaches),
        "observed_rate": float(p_hat),
        "expected_rate": float(p0),
    }


def run_im_backtest(
    as_of_start: date,
    as_of_end: date,
    db_path: str = "data/collateraliq.db",
    confidence: float = 0.99,
) -> pd.DataFrame:
    """
    Run IM adequacy backtest over a date range.

    For each (date, client): compare realised MPOR loss against IM held.
    A breach occurs when realised loss > IM.

    Returns DataFrame with backtest results.
    """
    engine = get_engine(db_path)

    # Load exposures and IM results
    with engine.connect() as conn:
        exposures = pd.read_sql(
            text("""
                SELECT e.date, e.client_id, e.collateralised_exposure_inr,
                       e.exposure_inr, e.net_mtm_inr, e.pfe_99_inr
                FROM exposures e
                WHERE e.date BETWEEN :d1 AND :d2
                ORDER BY e.date, e.client_id
            """),
            conn,
            params={"d1": str(as_of_start), "d2": str(as_of_end)},
        )

        im_results = pd.read_sql(
            text("""
                SELECT date, client_id, im_total_inr, method
                FROM im_results
                WHERE product_type = 'Portfolio'
                  AND date BETWEEN :d1 AND :d2
            """),
            conn,
            params={"d1": str(as_of_start), "d2": str(as_of_end)},
        )

    if exposures.empty or im_results.empty:
        logger.warning("No data for backtest range")
        return pd.DataFrame()

    # Merge
    bt = exposures.merge(im_results, on=["date", "client_id"], how="inner")
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt.sort_values(["client_id", "date"])

    # MPOR loss: use actual net MTM change over a standard 10-day MPOR window
    # If 10 days not available, fallback to max available
    bt["mpor_loss_inr"] = bt.groupby("client_id")["net_mtm_inr"].shift(-10) - bt["net_mtm_inr"]
    bt["mpor_loss_inr"] = bt["mpor_loss_inr"].fillna(0).clip(lower=0)

    # Breach flag
    bt["is_breach"] = bt["mpor_loss_inr"] > bt["im_total_inr"]
    bt["shortfall_inr"] = (bt["mpor_loss_inr"] - bt["im_total_inr"]).clip(lower=0)
    bt["coverage_ratio"] = bt["im_total_inr"] / bt["mpor_loss_inr"].replace(0, np.nan)

    # Write to DB
    _write_backtest_results(bt, engine)

    return bt


def compute_backtest_summary(
    bt_df: pd.DataFrame,
    confidence: float = 0.99,
) -> dict:
    """
    Compute summary statistics for the backtest.

    Returns overall and per-client/per-product summary.
    """
    if bt_df.empty:
        return {}

    # Remove rows with no loss (not informative)
    bt_active = bt_df[bt_df["mpor_loss_inr"] > 0].copy()
    if bt_active.empty:
        return {"n_obs": 0, "n_breaches": 0}

    n_obs = len(bt_active)
    n_breaches = int(bt_active["is_breach"].sum())
    breach_rate = n_breaches / n_obs

    # Kupiec test
    kupiec = kupiec_test(n_obs, n_breaches, confidence)

    # Severity
    shortfalls = bt_active[bt_active["is_breach"]]["shortfall_inr"]
    avg_shortfall = float(shortfalls.mean()) if len(shortfalls) > 0 else 0.0
    max_shortfall = float(shortfalls.max()) if len(shortfalls) > 0 else 0.0

    # Per-client summary
    client_summary = bt_active.groupby("client_id").agg(
        n_obs=("is_breach", "count"),
        n_breaches=("is_breach", "sum"),
        avg_shortfall=("shortfall_inr", "mean"),
        max_shortfall=("shortfall_inr", "max"),
        avg_coverage=("coverage_ratio", "mean"),
    ).reset_index()
    client_summary["breach_rate"] = client_summary["n_breaches"] / client_summary["n_obs"]

    return {
        "overall": {
            "n_obs": n_obs,
            "n_breaches": n_breaches,
            "breach_rate": breach_rate,
            "expected_breach_rate": 1 - confidence,
            "avg_shortfall_inr": avg_shortfall,
            "max_shortfall_inr": max_shortfall,
            "kupiec_test": kupiec,
        },
        "by_client": client_summary.to_dict(orient="records"),
    }


def _write_backtest_results(bt_df: pd.DataFrame, engine) -> None:
    """Write backtest results to im_backtest table."""
    with engine.begin() as conn:
        for _, row in bt_df.iterrows():
            try:
                conn.execute(
                    text("""
                        INSERT INTO im_backtest
                            (date, client_id, mpor_loss_inr, im_held_inr, is_breach, shortfall_inr, method)
                        VALUES
                            (:date, :client_id, :mpor_loss_inr, :im_held_inr, :is_breach, :shortfall_inr, :method) ON CONFLICT DO NOTHING
                    """),
                    {
                        "date": str(row["date"])[:10],
                        "client_id": row["client_id"],
                        "mpor_loss_inr": float(row["mpor_loss_inr"]),
                        "im_held_inr": float(row["im_total_inr"]),
                        "is_breach": int(row["is_breach"]),
                        "shortfall_inr": float(row["shortfall_inr"]),
                        "method": str(row.get("method", "")),
                    },
                )
            except Exception:
                pass
