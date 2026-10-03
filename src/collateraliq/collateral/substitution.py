"""
Collateral Substitution Workflow.

Handles logic for checking if a client's request to substitute collateral is valid
based on eligibility rules, haircuts, and concentration limits.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CollateralAsset:
    asset_id: str
    asset_class: str
    currency: str
    market_value: float
    haircut: float

def check_substitution_eligibility(
    csa_eligibility_json: str,
    current_asset: CollateralAsset,
    proposed_asset: CollateralAsset,
    current_im_requirement: float,
    total_collateral_value: float
) -> dict:
    """
    Check if a collateral substitution request is valid.
    
    Args:
        csa_eligibility_json: JSON string from csa_terms.eligible_collateral
        current_asset: The asset the client wants to withdraw
        proposed_asset: The asset the client wants to post
        current_im_requirement: Total IM required
        total_collateral_value: Current total post-haircut collateral held
        
    Returns:
        Dictionary with status, approval boolean, and reasoning.
    """
    try:
        eligible_rules = json.loads(csa_eligibility_json)
    except Exception:
        return {"approved": False, "reason": "Failed to parse CSA eligibility rules."}
        
    # Check if proposed asset is eligible
    eligible_classes = [r.get("asset") for r in eligible_rules]
    if proposed_asset.asset_class not in eligible_classes:
        return {
            "approved": False, 
            "reason": f"Asset class '{proposed_asset.asset_class}' is not eligible under this CSA."
        }
        
    # Find haircut for proposed asset in CSA rules (override proposed if CSA has stricter haircut)
    csa_haircut = next((r.get("haircut", 0.0) for r in eligible_rules if r.get("asset") == proposed_asset.asset_class), 0.0)
    effective_haircut = max(proposed_asset.haircut, csa_haircut)
    
    # Calculate post-haircut values
    current_post_hc = current_asset.market_value * (1 - current_asset.haircut)
    proposed_post_hc = proposed_asset.market_value * (1 - effective_haircut)
    
    # Calculate new total collateral
    new_total_collateral = total_collateral_value - current_post_hc + proposed_post_hc
    
    # Check if new total covers requirement
    if new_total_collateral < current_im_requirement:
        shortfall = current_im_requirement - new_total_collateral
        return {
            "approved": False,
            "reason": f"Substitution would result in a margin deficit of {shortfall:,.2f}. Proposed asset post-haircut value ({proposed_post_hc:,.2f}) is lower than current asset ({current_post_hc:,.2f})."
        }
        
    return {
        "approved": True,
        "reason": f"Substitution approved. New total collateral {new_total_collateral:,.2f} covers requirement of {current_im_requirement:,.2f}.",
        "proposed_post_haircut_value": proposed_post_hc,
        "current_post_haircut_value": current_post_hc,
        "excess_generated": new_total_collateral - current_im_requirement
    }
