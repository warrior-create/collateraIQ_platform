"""
Commentary Templates
"""
from datetime import date

def build_template(
    as_of: date,
    client_id: str,
    attribution: dict,
    exposure: float,
    im_held: float,
    vm_held: float,
    vm_call: float,
    coverage_ratio: float,
    alerts: list[dict],
) -> str:
    """Build deterministic commentary text."""
    total_change = attribution.get("total_change_inr", 0)
    change_pct = total_change / (exposure - total_change) * 100 if (exposure - total_change) != 0 else 0
    change_dir = "up" if total_change > 0 else "down"

    equity_pct = attribution.get("equity_move_pct", 0)
    fx_pct = attribution.get("fx_move_pct", 0)
    rate_bps = attribution.get("rate_move_bps", 0)

    drivers = []
    if abs(equity_pct) >= 0.5:
        drivers.append(f"Nifty {'+' if equity_pct > 0 else ''}{equity_pct:.1f}%")
    if abs(fx_pct) >= 0.3:
        drivers.append(f"USDINR {'+' if fx_pct > 0 else ''}{fx_pct:.2f}%")
    if abs(rate_bps) >= 5:
        drivers.append(f"INR rates {'+' if rate_bps > 0 else ''}{rate_bps:.0f}bps")

    driver_str = " and ".join(drivers) if drivers else "small market moves"

    # Call text
    if vm_call > 0.5e6:
        call_text = f" VM call of ₹{vm_call/1e6:.1f}M issued."
    elif vm_call < -0.5e6:
        call_text = f" VM return of ₹{abs(vm_call)/1e6:.1f}M processed."
    else:
        call_text = " No margin call required."

    # New/matured trades
    new_trade_text = ""
    if abs(attribution.get("new_trades_inr", 0)) > 1e6:
        new_trade_text = f" New trades added ₹{attribution['new_trades_inr']/1e6:.1f}M to exposure."

    # Alert text
    alert_texts = []
    for alert in alerts:
        if alert.get("severity") in ("critical", "high"):
            alert_texts.append(f"⚠️ {alert['message']}")

    alert_block = "\n".join(alert_texts) if alert_texts else ""

    text = (
        f"[{as_of}] {client_id} — Daily Margin Commentary\n"
        f"{'='*60}\n"
        f"Exposure: ₹{exposure/1e6:.1f}M ({change_dir} {abs(change_pct):.1f}% DoD), "
        f"driven by {driver_str}.\n"
        f"IM held: ₹{im_held/1e6:.1f}M | VM held: ₹{vm_held/1e6:.1f}M | "
        f"Coverage: {coverage_ratio*100:.0f}%.\n"
        f"{new_trade_text}{call_text}\n"
    )

    if alert_block:
        text += f"\n{alert_block}\n"

    return text.strip()
