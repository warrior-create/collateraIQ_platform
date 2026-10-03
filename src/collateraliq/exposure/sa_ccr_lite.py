"""
SA-CCR (Standardised Approach for Counterparty Credit Risk) - Lite version.
Computes Exposure at Default (EAD) for CEM reporting.

EAD = alpha * (RC + PFE)
alpha = 1.4
RC = max(V - C, 0)
PFE = multiplier * AddOn_aggregate
"""

from __future__ import annotations
import math

def compute_sa_ccr_ead(
    mtm: float, 
    vm_held: float, 
    im_held: float, 
    notional: float,
    asset_class: str = "equity"
) -> float:
    """
    Compute SA-CCR EAD for a single netting set (simplified).
    """
    alpha = 1.4
    
    # Replacement Cost (RC)
    rc = max(mtm - vm_held - im_held, 0.0)
    
    # PFE Addon based on asset class supervisory factors (simplified)
    # Basel III factors (approx): Equity=32%, FX=4%, Rates=0.5%, Credit=5.4%
    supervisory_factors = {
        "equity": 0.32,
        "fx": 0.04,
        "rates": 0.005,
        "credit": 0.054
    }
    sf = supervisory_factors.get(asset_class, 0.10)
    addon_aggregate = notional * sf
    
    # Multiplier
    # multiplier = min(1, 0.05 + 0.95 * exp( (V-C) / (2 * 0.95 * Addon) ))
    margin_uncollateralised = mtm - vm_held - im_held
    if addon_aggregate > 0:
        exp_term = math.exp(margin_uncollateralised / (2 * 0.95 * addon_aggregate))
        multiplier = min(1.0, 0.05 + 0.95 * exp_term)
    else:
        multiplier = 1.0
        
    pfe = multiplier * addon_aggregate
    
    ead = alpha * (rc + pfe)
    return ead
