"""
Margin Call Workflow System.
Manages Call -> Dispute -> Settlement -> Escalation.
"""

from __future__ import annotations
import pandas as pd
from datetime import date

def generate_margin_calls(as_of: date, db_path: str) -> pd.DataFrame:
    """
    Generate margin calls for clients where Exposure > Collateral + Threshold.
    Writes to margin_calls and simulates next-day settlement in collateral_inventory and vm_ledger.
    """
    from collateraliq.data.db import get_engine
    import uuid
    from sqlalchemy import text
    engine = get_engine(db_path)
    with engine.begin() as conn:
        df = pd.read_sql(f"""
            SELECT e.client_id, e.exposure_inr, c.threshold_inr, c.mta_inr,
                   COALESCE(v.vm_held_inr, 0) as vm_held_inr
            FROM exposures e
            JOIN csa_terms c ON e.client_id = c.client_id
            LEFT JOIN (
                SELECT client_id, vm_held_inr FROM vm_ledger WHERE date = (SELECT MAX(date) FROM vm_ledger)
            ) v ON e.client_id = v.client_id
            WHERE e.date = '{as_of}'
        """, conn)
        
    calls = []
    with engine.begin() as conn:
        for _, row in df.iterrows():
            deficit = row["exposure_inr"] - row["vm_held_inr"] - row["threshold_inr"]
            if deficit > row["mta_inr"]:
                call_id = f"CALL-{uuid.uuid4().hex[:8].upper()}"
                calls.append({
                    "call_id": call_id,
                    "client_id": row["client_id"],
                    "call_amount": deficit,
                    "status": "issued",
                    "date": str(as_of)
                })
                conn.execute(text("""
                    INSERT INTO margin_calls (call_id, date, client_id, call_type, amount_inr, status)
                    VALUES (:call_id, :date, :client_id, 'vm', :call_amount, 'issued')
                """), calls[-1])
                
                # Auto-simulate settlement in VM ledger for pipeline completeness
                new_vm = row["vm_held_inr"] + deficit
                conn.execute(text("""
                    INSERT INTO vm_ledger (date, client_id, netting_set_id, vm_required_inr, vm_held_inr, vm_call_inr, vm_return_inr)
                    VALUES (:date, :client_id, :ns, :vm_req, :vm_held, :vm_call, 0)
                """), {
                    "date": str(as_of), "client_id": row["client_id"], "ns": f"NS_{row['client_id']}_01",
                    "vm_req": row["exposure_inr"], "vm_held": new_vm, "vm_call": deficit
                })
                
                # Also update collateral_inventory for this cash call
                conn.execute(text("""
                    INSERT INTO collateral_inventory (date, client_id, asset_type, notional_inr, haircut, eligible_value_inr)
                    VALUES (:date, :client_id, 'cash_inr', :amt, 0, :amt)
                """), {"date": str(as_of), "client_id": row["client_id"], "amt": deficit})
                
    return pd.DataFrame(calls)
