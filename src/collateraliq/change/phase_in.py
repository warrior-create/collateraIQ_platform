"""
Phase-in scheduling for margin increases.

Generates a step-up schedule to prevent margin shock for clients.
"""
from __future__ import annotations
import pandas as pd

def calculate_phase_in(
    client_id: str, 
    current_im: float, 
    target_im: float, 
    months: int = 6,
    min_increment: float = 100000.0
) -> pd.DataFrame:
    """
    Calculate a phased margin increase schedule.
    
    Args:
        client_id: Client identifier
        current_im: Baseline IM
        target_im: New target IM from rule change
        months: Phase-in period
        min_increment: Minimum bump size to avoid nuisance calls
        
    Returns:
        DataFrame schedule of expected IM requirements per month.
    """
    if target_im <= current_im:
        # No phase-in needed for decreases
        return pd.DataFrame([{
            "month": 1,
            "target_im": target_im,
            "increment": target_im - current_im
        }])
        
    total_increase = target_im - current_im
    raw_step = total_increase / months
    
    # Adjust step size based on minimum increment
    actual_step = max(raw_step, min_increment)
    
    schedule = []
    accumulated_im = current_im
    
    for month in range(1, months + 1):
        if accumulated_im + actual_step >= target_im or month == months:
            # Final step captures the remainder
            step = target_im - accumulated_im
            accumulated_im = target_im
        else:
            step = actual_step
            accumulated_im += step
            
        schedule.append({
            "client_id": client_id,
            "month": month,
            "target_im": accumulated_im,
            "increment": step
        })
        
        if accumulated_im >= target_im:
            break
            
    return pd.DataFrame(schedule)
