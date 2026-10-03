"""
Generate Impact Report for What-If Scenarios.

Produces detailed statistical breakdown of margin changes.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def generate_impact_report(baseline_im: dict[str, float], new_im_df: pd.DataFrame) -> dict:
    """
    Generate comprehensive impact report of a model change.
    
    Args:
        baseline_im: Dictionary of {client_id: current_im_inr}
        new_im_df: DataFrame of newly computed IM
        
    Returns:
        Dict of impact statistics.
    """
    new_im = new_im_df[new_im_df["product_type"].isna()].set_index("client_id")["im_total_inr"].to_dict()
    
    impacts = []
    for cid, base_val in baseline_im.items():
        new_val = new_im.get(cid, base_val)
        diff = new_val - base_val
        pct_change = (diff / max(base_val, 1.0)) * 100
        impacts.append({
            "client_id": cid,
            "baseline_im": base_val,
            "new_im": new_val,
            "diff_abs": diff,
            "diff_pct": pct_change
        })
        
    df = pd.DataFrame(impacts)
    
    total_baseline = df["baseline_im"].sum()
    total_new = df["new_im"].sum()
    
    # Find most impacted clients
    top_increases = df.sort_values("diff_abs", ascending=False).head(3)
    top_decreases = df.sort_values("diff_abs", ascending=True).head(3)
    
    return {
        "total_baseline_im": total_baseline,
        "total_new_im": total_new,
        "total_change_abs": total_new - total_baseline,
        "total_change_pct": ((total_new / total_baseline) - 1) * 100 if total_baseline > 0 else 0,
        "clients_with_increase": len(df[df["diff_abs"] > 0]),
        "clients_with_decrease": len(df[df["diff_abs"] < 0]),
        "max_increase_pct": df["diff_pct"].max(),
        "max_decrease_pct": df["diff_pct"].min(),
        "top_3_increases": top_increases[["client_id", "diff_abs", "diff_pct"]].to_dict(orient="records"),
        "detailed_impacts": impacts
    }
