"""
Monte Carlo simulation for Potential Future Exposure (PFE) and Expected Exposure (EE).

Uses:
- Geometric Brownian Motion (GBM) for equity and FX
- Hull-White (one-factor) for interest rates

Output: EE, EPE, PFE at 95% and 99% for each client netting set.

References
----------
Gregory, Counterparty Credit Risk and CVA, 2nd ed. Wiley Finance.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate GBM paths: dS = μS dt + σS dW.

    Returns array of shape (n_paths, n_steps+1).
    """
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def hull_white_paths(
    r0: float,
    theta: float,
    kappa: float,
    sigma_r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Hull-White one-factor rate model: dr = κ(θ - r)dt + σ dW.

    Parameters
    ----------
    r0      : initial short rate
    theta   : long-run mean rate
    kappa   : mean-reversion speed
    sigma_r : short-rate volatility

    Returns array of shape (n_paths, n_steps+1).
    """
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0
    Z = rng.standard_normal((n_paths, n_steps))

    for i in range(n_steps):
        r = paths[:, i]
        drift = kappa * (theta - r) * dt
        shock = sigma_r * np.sqrt(dt) * Z[:, i]
        paths[:, i + 1] = np.maximum(r + drift + shock, -0.01)  # floor at -1%

    return paths


def compute_ee_pfe(
    client_id: str,
    trades_df,
    market: dict,
    n_paths: int = 5000,
    n_steps: int = 50,
    T_years: float = 5.0,
    seed: int = 42,
) -> dict:
    """
    Compute EE, EPE, PFE(95%), PFE(99%) for a client netting set.

    Returns
    -------
    dict: ee_profile, epe, pfe_95, pfe_99 (in INR), time_grid
    """
    rng = np.random.default_rng(seed)
    time_grid = np.linspace(0, T_years, n_steps + 1)

    # Market parameters
    nifty_spot = market.get("^NSEI_close", 22000)
    nifty_vol = market.get("^INDIAVIX_close", 20.0) / 100
    fx_spot = market.get("USDINR=X_close", 84.0)
    r_inr = market.get("INDIRLTLT01STM_rate", 0.07)
    r_usd = market.get("SOFR_rate", 0.05)

    # Simulate paths for each risk factor
    equity_paths = gbm_paths(
        nifty_spot, r_inr - 0.012, nifty_vol, T_years, n_steps, n_paths, rng
    )
    fx_paths = gbm_paths(
        fx_spot, r_inr - r_usd, 0.05, T_years, n_steps, n_paths, rng
    )
    rate_paths = hull_white_paths(
        r_inr, r_inr, 0.10, 0.01, T_years, n_steps, n_paths, rng
    )

    # Compute portfolio MTM at each time step and path
    exposure_matrix = np.zeros((n_paths, n_steps + 1))

    for i, t in enumerate(time_grid):
        # Simple exposure model: scale current exposure by scenario
        # In production: full revaluation per path
        equity_ratio = equity_paths[:, i] / nifty_spot
        fx_ratio = fx_paths[:, i] / fx_spot
        rate_shock = rate_paths[:, i] - r_inr

        # Proxy MTM: sum of risk-factor sensitivities × scenario
        total_delta = trades_df["delta_inr"].sum() if "delta_inr" in trades_df.columns else 0
        total_dv01 = trades_df["dv01_inr"].sum() if "dv01_inr" in trades_df.columns else 0
        total_mtm_base = trades_df["mtm_inr"].sum() if "mtm_inr" in trades_df.columns else 0

        scenario_mtm = (
            total_mtm_base
            + total_delta * (equity_ratio - 1)
            + total_dv01 * rate_shock * 10000
        )

        exposure_matrix[:, i] = np.maximum(scenario_mtm, 0)

    # EE profile: mean positive exposure at each time step
    ee_profile = exposure_matrix.mean(axis=0)

    # EPE: average of EE over the life
    epe = float(ee_profile.mean())

    # PFE: percentiles of exposure
    pfe_95 = float(np.percentile(exposure_matrix, 95, axis=0).max())
    pfe_99 = float(np.percentile(exposure_matrix, 99, axis=0).max())

    return {
        "client_id": client_id,
        "ee_profile": ee_profile.tolist(),
        "epe": epe,
        "pfe_95": pfe_95,
        "pfe_99": pfe_99,
        "time_grid": time_grid.tolist(),
        "n_paths": n_paths,
    }
