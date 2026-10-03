"""
Client default loss simulation.

For each client and date, simulate default and close-out over the MPOR.
Loss = max(V(d+MPOR) - VM - IM + liquidation_cost, 0)

Reports the distribution of shortfalls and the fraction of default scenarios
where IM+VM would have covered the loss.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from collateraliq.data.db import get_engine
from collateraliq.exposure.netting import collateralised_exposure_at_default

logger = logging.getLogger(__name__)


def simulate_default_scenarios(
    as_of: date,
    clients_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    im_df: pd.DataFrame,
    vm_df: pd.DataFrame,
    market: dict,
    mpor_days: int = 5,
    n_scenarios: int = 1000,
    seed: int = 42,
    db_path: str = "data/collateraliq.db",
) -> pd.DataFrame:
    """
    Simulate default close-out scenarios for each client.

    For each client, simulate n_scenarios market moves over MPOR days.
    Compute loss under each scenario.

    Returns DataFrame of results.
    """
    rng = np.random.default_rng(seed)
    engine = get_engine(db_path)
    results = []

    nifty_vol = market.get("^INDIAVIX_close", 20.0) / 100
    fx_vol = 0.08
    rate_vol = 0.01

    for _, client in clients_df.iterrows():
        client_id = client["client_id"]

        # Get current exposure
        exp_row = exposures_df[exposures_df["client_id"] == client_id]
        if exp_row.empty:
            continue
        current_exposure = float(exp_row["exposure_inr"].iloc[0])

        # Get VM and IM held
        vm_row = vm_df[vm_df["client_id"] == client_id] if not vm_df.empty else pd.DataFrame()
        vm_held = float(vm_row["vm_held_inr"].sum()) if not vm_row.empty else 0.0

        im_row = im_df[
            (im_df["client_id"] == client_id) & im_df["product_type"].isna()
        ]
        im_held = float(im_row["im_total_inr"].sum()) if not im_row.empty else 0.0

        ia = float(client.get("independent_amount_inr", 0.0))

        # Simulate MPOR market moves
        # Equity shock (main driver for most clients)
        equity_shocks = rng.standard_normal(n_scenarios) * nifty_vol * np.sqrt(mpor_days / 252)
        fx_shocks = rng.standard_normal(n_scenarios) * fx_vol * np.sqrt(mpor_days / 252)

        # Proxy: exposure changes proportionally to equity shock
        # In production: full revaluation per scenario
        delta_exposure = current_exposure * equity_shocks * 0.5
        exposure_at_mpor = np.maximum(current_exposure + delta_exposure, 0)

        # Liquidation cost: 0.5% of notional for most products
        liq_cost = current_exposure * 0.005

        # Close-out loss per scenario
        losses = np.maximum(
            exposure_at_mpor - vm_held - im_held - ia + liq_cost, 0
        )

        # Summary
        n_covered = int(np.sum(losses <= 0))
        coverage_pct = n_covered / n_scenarios * 100

        results.append({
            "sim_date": str(as_of),
            "client_id": client_id,
            "archetype": client.get("archetype", ""),
            "current_exposure_inr": current_exposure,
            "vm_held_inr": vm_held,
            "im_held_inr": im_held,
            "liq_cost_inr": liq_cost,
            "n_scenarios": n_scenarios,
            "n_covered": n_covered,
            "coverage_pct": coverage_pct,
            "mean_loss_inr": float(losses.mean()),
            "p95_loss_inr": float(np.percentile(losses, 95)),
            "p99_loss_inr": float(np.percentile(losses, 99)),
            "max_loss_inr": float(losses.max()),
            "expected_shortfall_inr": float(
                losses[losses > np.percentile(losses, 95)].mean()
            ) if len(losses) > 0 else 0.0,
        })

    df = pd.DataFrame(results)
    _write_default_loss(df, engine)
    return df


def _write_default_loss(df: pd.DataFrame, engine) -> None:
    """Write default loss simulation results to DB."""
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO default_loss
                        (sim_date, client_id, close_out_loss_inr, vm_held_inr, im_held_inr,
                         liquidation_cost_inr, net_shortfall_inr, is_covered, scenario_label)
                    VALUES
                        (:sim_date, :client_id, :mean_loss, :vm_held, :im_held,
                         :liq_cost, :shortfall, :is_covered, :label) ON CONFLICT DO NOTHING
                """),
                {
                    "sim_date": row["sim_date"],
                    "client_id": row["client_id"],
                    "mean_loss": row["mean_loss_inr"],
                    "vm_held": row["vm_held_inr"],
                    "im_held": row["im_held_inr"],
                    "liq_cost": row["liq_cost_inr"],
                    "shortfall": max(row["mean_loss_inr"] - row["vm_held_inr"] - row["im_held_inr"], 0),
                    "is_covered": int(row["coverage_pct"] >= 99),
                    "label": "fhs_mc_sim",
                },
            )


def default_scenario_summary(df: pd.DataFrame) -> dict:
    """Compute summary statistics across all default simulations."""
    if df.empty:
        return {}

    overall_coverage = float(df["coverage_pct"].mean())
    by_archetype = df.groupby("archetype")["coverage_pct"].mean().to_dict()

    return {
        "overall_coverage_pct": overall_coverage,
        "coverage_by_archetype": by_archetype,
        "n_clients": len(df),
        "worst_coverage_client": df.loc[df["coverage_pct"].idxmin(), "client_id"],
        "worst_coverage_pct": float(df["coverage_pct"].min()),
        "total_potential_shortfall_inr": float(
            (df["p99_loss_inr"] - df["im_held_inr"] - df["vm_held_inr"]).clip(lower=0).sum()
        ),
    }
