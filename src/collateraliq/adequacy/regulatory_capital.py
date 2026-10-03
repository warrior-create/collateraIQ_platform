"""
Regulatory Capital Comparison: SA-CCR vs CEM.

Implements the SA-CCR EAD calculation defined in BCBS d279 and compares it 
to the older Current Exposure Method (CEM) for impact analysis.
"""
from __future__ import annotations
import math

def compare_capital_approaches(
    notional: float, 
    mtm: float, 
    asset_class: str,
    maturity_years: float = 1.0,
    is_collateralised: bool = True
) -> dict:
    """
    Compare SA-CCR EAD with old CEM approach.
    
    Args:
        notional: Trade notional amount
        mtm: Mark-to-market value
        asset_class: One of 'rates', 'fx', 'equity', 'credit_ig', 'credit_hy'
        maturity_years: Time to maturity
        is_collateralised: Whether trade is under a daily-margined CSA
        
    Returns:
        Dictionary of EADs and the capital benefit/punishment.
    """
    # 1. CEM (Current Exposure Method)
    # EAD = max(MTM, 0) + Notional * AddonFactor
    cem_addons = {
        "rates": 0.015 if maturity_years > 5 else (0.005 if maturity_years > 1 else 0.0),
        "fx": 0.075 if maturity_years > 5 else (0.05 if maturity_years > 1 else 0.01),
        "equity": 0.10,
        "credit_ig": 0.05,
        "credit_hy": 0.10
    }
    cem_addon_factor = cem_addons.get(asset_class, 0.10)
    cem_ead = max(mtm, 0) + notional * cem_addon_factor
    
    if is_collateralised:
        cem_ead = max(0, cem_ead) # Simplified VM netting for CEM
        
    # 2. SA-CCR (Standardised Approach for Counterparty Credit Risk)
    # EAD = alpha * (ReplacementCost + PotentialFutureExposure)
    alpha = 1.4
    
    # Replacement Cost (RC)
    if is_collateralised:
        # Assuming VM covers MTM perfectly for simplified comparison
        rc = 0.0 
    else:
        rc = max(mtm, 0)
        
    # Potential Future Exposure (PFE)
    supervisory_factors = {
        "rates": 0.005,
        "fx": 0.04,
        "equity": 0.32,
        "credit_ig": 0.0038,
        "credit_hy": 0.0106
    }
    sf = supervisory_factors.get(asset_class, 0.32)
    
    # Maturity Factor (MF)
    if is_collateralised:
        # MF for margined trades = 1.5 * sqrt(MPOR/250) (Assuming MPOR=10)
        mf = 1.5 * math.sqrt(10 / 250)
    else:
        # MF for unmargined trades = sqrt(min(maturity, 1) / 1) -> Simplified
        mf = math.sqrt(min(maturity_years, 1.0))
        
    pfe = sf * notional * mf
    
    saccr_ead = alpha * (rc + pfe)
    
    return {
        "CEM_EAD": cem_ead,
        "SA_CCR_EAD": saccr_ead,
        "Benefit_vs_CEM": cem_ead - saccr_ead,
        "percentage_change": ((saccr_ead / max(cem_ead, 1.0)) - 1) * 100
    }
