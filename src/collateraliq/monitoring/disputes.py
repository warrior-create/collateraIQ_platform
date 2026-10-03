"""
Margin Call Dispute Management System.

Handles the lifecycle of a margin call dispute:
1. Identify discrepancies between proposed call and client calculation
2. Check against dispute tolerance levels
3. Log the dispute and update call status
4. Suggest resolution (waive, partial fund, manual escalation)
"""
from __future__ import annotations

import pandas as pd
import logging
from datetime import date
from sqlalchemy import text
from collateraliq.data.db import get_engine

logger = logging.getLogger(__name__)

def evaluate_disputes(
    calls_df: pd.DataFrame, 
    client_disputes: dict[str, float], 
    tolerance_inr: float = 50000.0,
    tolerance_pct: float = 0.05
) -> pd.DataFrame:
    """
    Evaluate margin calls against client's disputed amounts.
    
    Args:
        calls_df: DataFrame of issued margin calls (client_id, call_amount, etc.)
        client_disputes: Dictionary mapping client_id to their proposed margin call amount.
        tolerance_inr: Absolute tolerance in INR.
        tolerance_pct: Relative tolerance as percentage of total call.
        
    Returns:
        DataFrame of updated margin calls with dispute status.
    """
    updated_calls = calls_df.copy()
    
    status_list = []
    action_list = []
    diff_list = []
    
    for _, row in updated_calls.iterrows():
        cid = row["client_id"]
        call_amt = row["call_amount"]
        
        if cid in client_disputes:
            client_amt = client_disputes[cid]
            diff = abs(call_amt - client_amt)
            diff_list.append(diff)
            
            # Check tolerances
            if diff <= tolerance_inr or diff / max(call_amt, 1.0) <= tolerance_pct:
                status_list.append("dispute_resolved")
                action_list.append(f"Waive difference of {diff:,.2f} INR (within tolerance). Fund {client_amt:,.2f}")
            else:
                status_list.append("dispute_escalated")
                action_list.append(f"Manual reconciliation required. Difference {diff:,.2f} INR exceeds tolerance.")
        else:
            diff_list.append(0.0)
            status_list.append(row.get("status", "issued"))
            action_list.append("Awaiting client response")
            
    updated_calls["dispute_diff"] = diff_list
    updated_calls["status"] = status_list
    updated_calls["recommended_action"] = action_list
    
    return updated_calls

def log_disputes_to_db(disputes_df: pd.DataFrame, db_path: str):
    """Save the dispute status to the database."""
    engine = get_engine(db_path)
    with engine.begin() as conn:
        for _, row in disputes_df.iterrows():
            if "dispute" in row["status"]:
                conn.execute(
                    text("""
                        INSERT INTO disputes (client_id, date, original_call, disputed_diff, status, action)
                        VALUES (:cid, :d, :call, :diff, :status, :action) ON CONFLICT DO NOTHING
                    """),
                    {
                        "cid": row["client_id"],
                        "d": str(row["date"]),
                        "call": row["call_amount"],
                        "diff": row["dispute_diff"],
                        "status": row["status"],
                        "action": row["recommended_action"]
                    }
                )
    logger.info(f"Logged {len(disputes_df[disputes_df['status'].str.contains('dispute')])} disputes to database.")
