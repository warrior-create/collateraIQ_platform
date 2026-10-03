"""
Netting and collateralised exposure calculation.

Key calculations:
1. Net MTM per netting set: E = max(Σ MTM, 0)
2. VM call: max(E - threshold - VM_held, MTA), rounded
3. Collateralised exposure at default: max(V(t+MPOR) - VM - IM, 0)
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_netting_set_exposure(
    valuations_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute exposure per netting set.

    Parameters
    ----------
    valuations_df : trade-level MTM, columns: trade_id, mtm_inr
    trades_df     : trade metadata, columns: trade_id, netting_set_id, client_id

    Returns
    -------
    DataFrame: netting_set_id, client_id, gross_mtm_inr, net_mtm_inr, exposure_inr
    """
    merged = valuations_df.merge(
        trades_df[["trade_id", "netting_set_id", "client_id"]],
        on="trade_id",
        how="left",
    )

    agg = merged.groupby(["netting_set_id", "client_id"]).agg(
        gross_mtm_inr=("mtm_inr", lambda x: x[x > 0].sum()),
        net_mtm_inr=("mtm_inr", "sum"),
    ).reset_index()

    agg["exposure_inr"] = np.maximum(agg["net_mtm_inr"], 0)
    return agg


def compute_vm_call(
    exposure: float,
    threshold: float,
    mta: float,
    rounding: float,
    vm_held: float,
    ia_received: float = 0.0,
) -> float:
    """
    Compute VM (Variation Margin) call amount.

    VM call = max(Exposure - Threshold - VM_held - IA, MTA), rounded to rounding unit.
    Return < 0 means return collateral to client.

    Parameters
    ----------
    exposure   : current net exposure (INR)
    threshold  : CSA threshold (INR)
    mta        : minimum transfer amount (INR)
    rounding   : rounding unit (INR)
    vm_held    : collateral already held (INR)
    ia_received: independent amount received from client

    Returns
    -------
    VM call amount (positive = call from client, negative = return to client)
    """
    required = max(exposure - threshold - ia_received, 0)
    net_call = required - vm_held

    if abs(net_call) < mta:
        return 0.0

    # Round to nearest rounding unit
    if rounding > 0:
        net_call = np.round(net_call / rounding) * rounding

    return float(net_call)


def collateralised_exposure_at_default(
    mtm_at_default: float,
    vm_held: float,
    im_held: float,
    liquidation_cost: float = 0.0,
) -> float:
    """
    Exposure at default after collateral and IM.

    EAD = max(MtM(t+MPOR) - VM_held - IM_held + liquidation_cost, 0)

    This is the loss we face if the client defaults at t+MPOR.
    """
    return float(max(mtm_at_default - vm_held - im_held + liquidation_cost, 0.0))


def compute_client_exposures(
    netting_exposures: pd.DataFrame,
    trades_df: pd.DataFrame,
    csa_df: pd.DataFrame,
    vm_ledger_df: pd.DataFrame,
    im_results_df: pd.DataFrame,
    collateral_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full exposure calculation per client.

    Merges exposure, VM, IM and collateral to produce collateralised exposure.
    """
    df = netting_exposures.copy()

    # Merge CSA terms
    df = df.merge(
        csa_df[["client_id", "threshold_inr", "mta_inr", "independent_amount_inr",
                "rounding_inr", "mpor_days"]],
        on="client_id",
        how="left",
    )

    # Merge VM held
    if not vm_ledger_df.empty:
        vm = vm_ledger_df.groupby("client_id")["vm_held_inr"].sum().reset_index()
        df = df.merge(vm, on="client_id", how="left")
        df["vm_held_inr"] = df["vm_held_inr"].fillna(0)
    else:
        df["vm_held_inr"] = 0.0

    # Merge IM held (portfolio-level)
    if not im_results_df.empty:
        im = im_results_df[im_results_df["product_type"] == "Portfolio"].groupby("client_id")["im_total_inr"].sum().reset_index()
        df = df.merge(im.rename(columns={"im_total_inr": "im_held_inr"}), on="client_id", how="left")
        df["im_held_inr"] = df["im_held_inr"].fillna(0)
    else:
        df["im_held_inr"] = 0.0

    # SA-CCR Implementation
    # 1. Replacement Cost (RC) = max(V - C, 0)
    # V = net_mtm, C = VM + IM (simplified, not including threshold/MTA mechanics for V-C)
    # 2. PFE = multiplier * AddOn
    # EAD = 1.4 * (RC + PFE)

    # First calculate the AddOn for each client based on their trades
    addons = []
    if not trades_df.empty:
        for cid in df["client_id"]:
            c_trades = trades_df[trades_df["client_id"] == cid]
            addon = 0.0
            for _, t in c_trades.iterrows():
                # Simplified Supervisory Factors
                sf = 0.32 if t["product_type"] == "equity_swap" else \
                     0.04 if t["product_type"] == "fx_forward" else \
                     0.005 # rates
                addon += float(t["notional_inr"]) * sf
            addons.append(addon)
    else:
        addons = [0.0] * len(df)
        
    df["addon"] = addons
    
    # Calculate RC and V_minus_C
    df["v_minus_c"] = df["net_mtm_inr"] - df["vm_held_inr"] - df["im_held_inr"]
    df["rc"] = df["v_minus_c"].clip(lower=0)
    
    # Multiplier: min(1, 0.05 + 0.95 * exp((V - C) / (1.9 * AddOn)))
    def _calc_mult(row):
        addon = row["addon"]
        if addon <= 0:
            return 1.0
        exponent = row["v_minus_c"] / (1.9 * addon)
        # Cap exponent to avoid overflow
        exponent = max(min(exponent, 50), -50)
        import math
        return min(1.0, 0.05 + 0.95 * math.exp(exponent))
        
    df["multiplier"] = df.apply(_calc_mult, axis=1)
    df["pfe_saccr"] = df["multiplier"] * df["addon"]
    
    # SA-CCR EAD
    df["collateralised_exposure_inr"] = 1.4 * (df["rc"] + df["pfe_saccr"])

    # VM calls
    df["vm_call_inr"] = df.apply(
        lambda r: compute_vm_call(
            r["exposure_inr"],
            r.get("threshold_inr", 0),
            r.get("mta_inr", 0),
            r.get("rounding_inr", 100000),
            r.get("vm_held_inr", 0),
            r.get("independent_amount_inr", 0),
        ),
        axis=1,
    )

    return df
