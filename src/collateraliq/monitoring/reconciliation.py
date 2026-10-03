"""
Portfolio Reconciliation Engine (Python vs VBA/External).

Compares our internal valuations against an external file (e.g., from a client or a legacy VBA system).
Identifies MTM breaks at the trade level.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def reconcile_portfolios(
    internal_df: pd.DataFrame, 
    external_df: pd.DataFrame, 
    match_keys: list[str] = ["trade_id"],
    value_col: str = "npv",
    ext_value_col: str = "npv",
    tolerance_pct: float = 0.05,
    tolerance_abs: float = 1000.0
) -> dict:
    """
    Reconcile internal trade valuations against external system valuations.
    
    Returns a dictionary containing summary statistics and the detailed break report.
    """
    # Merge the dataframes on match keys
    merged = pd.merge(
        internal_df, 
        external_df, 
        on=match_keys, 
        how="outer", 
        suffixes=("_int", "_ext"),
        indicator=True
    )
    
    # Missing trades
    missing_in_ext = merged[merged["_merge"] == "left_only"].copy()
    missing_in_int = merged[merged["_merge"] == "right_only"].copy()
    
    # Matched trades
    matched = merged[merged["_merge"] == "both"].copy()
    
    # Calculate differences
    matched["diff_abs"] = (matched[f"{value_col}_int"] - matched[f"{ext_value_col}_ext"]).abs()
    
    # Handle division by zero
    ext_val_safe = matched[f"{ext_value_col}_ext"].replace(0, 1e-9)
    matched["diff_pct"] = (matched["diff_abs"] / ext_val_safe.abs())
    
    # Identify breaks
    matched["is_break"] = (matched["diff_pct"] > tolerance_pct) & (matched["diff_abs"] > tolerance_abs)
    
    breaks_df = matched[matched["is_break"]].copy()
    
    summary = {
        "total_internal_trades": len(internal_df),
        "total_external_trades": len(external_df),
        "matched_trades": len(matched),
        "missing_in_external": len(missing_in_ext),
        "missing_in_internal": len(missing_in_int),
        "valuation_breaks": len(breaks_df),
        "match_rate": len(matched) / max(len(internal_df), 1) * 100,
        "clean_match_rate": (len(matched) - len(breaks_df)) / max(len(internal_df), 1) * 100
    }
    
    return {
        "summary": summary,
        "breaks": breaks_df,
        "missing_internal": missing_in_int,
        "missing_external": missing_in_ext
    }
