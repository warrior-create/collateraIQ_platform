"""
Liquidation horizon calculations.
"""
def days_to_liquidate(position_size: float, adv: float, max_participation: float = 0.1) -> float:
    daily_capacity = adv * max_participation
    if daily_capacity == 0: return 999.0
    return position_size / daily_capacity
