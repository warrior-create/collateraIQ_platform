"""
FastAPI pre-trade margin quote service.

POST /quote: given a new proposed trade, compute:
- Extra IM required (marginal IM impact)
- New collateralised exposure
- Whether credit limit would be breached
- Whether any add-ons (concentration, liquidity) would be triggered

This is a live decision support tool for traders.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from collateraliq.data.db import get_engine, init_db
from collateraliq.clients.generator import load_clients
from collateraliq.clients.trade_lifecycle import get_active_trades
from collateraliq.margin.im_engine import IMEngine
from collateraliq.exposure.netting import compute_netting_set_exposure
from collateraliq.margin.addons import compute_all_addons

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CollateralIQ Pre-Trade Quote API",
    description="Real-time IM and exposure impact for proposed trades",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProposedTrade(BaseModel):
    client_id: str
    product_type: str
    direction: int = 1
    notional_inr: float
    params: dict = {}
    as_of: str | None = None


class MarginQuote(BaseModel):
    client_id: str
    product_type: str
    notional_inr: float
    marginal_im_inr: float
    current_im_inr: float
    new_total_im_inr: float
    marginal_exposure_inr: float
    current_exposure_inr: float
    new_exposure_inr: float
    limit_breach_risk: bool
    concentration_warning: bool
    liquidity_warning: bool
    addon_details: dict
    commentary: str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "CollateralIQ Pre-Trade Quote API"}


@app.post("/quote", response_model=MarginQuote)
async def get_margin_quote(trade: ProposedTrade):
    """
    Get pre-trade margin quote.

    Returns marginal IM impact and risk flags for a proposed trade.
    """
    try:
        db_path = "data/collateraliq.db"
        as_of = date.fromisoformat(trade.as_of) if trade.as_of else date.today()

        # Load current state
        clients_df = load_clients(db_path)
        client_row = clients_df[clients_df["client_id"] == trade.client_id]
        if client_row.empty:
            raise HTTPException(status_code=404, detail=f"Client {trade.client_id} not found")

        client = client_row.iloc[0]
        credit_limit = float(client.get("credit_limit_inr", 1e10))

        # Get current IM (from DB)
        engine = get_engine(db_path)
        with engine.connect() as conn:
            from sqlalchemy import text
            current_im_row = conn.execute(
                text("""
                    SELECT im_total_inr FROM im_results
                    WHERE client_id = :cid AND product_type = 'Portfolio'
                    ORDER BY date DESC LIMIT 1
                """),
                {"cid": trade.client_id},
            ).fetchone()
            current_exp_row = conn.execute(
                text("""
                    SELECT exposure_inr FROM exposures
                    WHERE client_id = :cid
                    ORDER BY date DESC LIMIT 1
                """),
                {"cid": trade.client_id},
            ).fetchone()

        current_im = float(current_im_row[0]) if current_im_row else 0.0
        current_exposure = float(current_exp_row[0]) if current_exp_row else 0.0

        # Estimate marginal IM and exposure by simulating the trade
        engine_im = IMEngine(db_path=db_path)
        marginal_im, marginal_exposure = _estimate_marginal_impact(trade, as_of, db_path, engine_im)

        new_im = current_im + marginal_im
        new_exposure = current_exposure + marginal_exposure

        # Check limits
        limit_breach_risk = new_exposure > credit_limit * 0.95  # Within 5% of limit

        # Add-ons check
        adv_proxy = trade.notional_inr * 0.10  # 10% daily turnover proxy
        addons = compute_all_addons(
            trade.notional_inr, adv_proxy, marginal_im,
            collateral_correlation=0.0,
        )

        concentration_warning = addons["is_concentrated"]
        liquidity_warning = addons["is_illiquid"]

        # Commentary
        commentary = (
            f"Proposed {trade.product_type} trade for {trade.client_id}: "
            f"notional ₹{trade.notional_inr/1e6:.1f}M. "
            f"Marginal IM: ₹{marginal_im/1e6:.2f}M. "
            f"New total exposure: ₹{new_exposure/1e6:.1f}M "
            f"({new_exposure/credit_limit*100:.0f}% of limit)."
        )
        if concentration_warning:
            commentary += f" ⚠️ Concentration add-on will apply (+₹{addons['addon_concentration_inr']/1e6:.2f}M)."
        if limit_breach_risk:
            commentary += " ⚠️ WARNING: Near credit limit breach."

        return MarginQuote(
            client_id=trade.client_id,
            product_type=trade.product_type,
            notional_inr=trade.notional_inr,
            marginal_im_inr=marginal_im,
            current_im_inr=current_im,
            new_total_im_inr=new_im,
            marginal_exposure_inr=marginal_exposure,
            current_exposure_inr=current_exposure,
            new_exposure_inr=new_exposure,
            limit_breach_risk=limit_breach_risk,
            concentration_warning=concentration_warning,
            liquidity_warning=liquidity_warning,
            addon_details=addons,
            commentary=commentary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _estimate_marginal_impact(trade: ProposedTrade, as_of: date, db_path: str, engine_im) -> tuple[float, float]:
    """Calculate actual marginal IM and Exposure impact by adding trade to portfolio."""
    from collateraliq.monitoring.daily_run import DailyRunner
    from collateraliq.pricing.engine import value_all_trades
    from collateraliq.exposure.netting import compute_netting_set_exposure
    import pandas as pd
    
    runner = DailyRunner(db_path=db_path)
    market = runner._load_market(as_of)
    
    # 1. Existing trades for this client
    all_trades = get_active_trades(as_of, db_path)
    client_trades = all_trades[all_trades["client_id"] == trade.client_id].copy() if not all_trades.empty else pd.DataFrame()
    
    # Value existing
    if not client_trades.empty:
        base_vals = value_all_trades(client_trades, market, as_of, db_path)
        base_netting = compute_netting_set_exposure(base_vals, client_trades)
        base_netting_row = base_netting[base_netting["netting_set_id"] == f"NS_{trade.client_id}_01"]
        base_exp = float(base_netting_row["exposure_inr"].iloc[0]) if not base_netting_row.empty else 0.0
        # Base IM for this product
        base_im_res = engine_im._compute_product_im(trade.product_type, client_trades[client_trades["product_type"] == trade.product_type], client_trades[client_trades["product_type"] == trade.product_type]["notional_inr"].sum(), market, None, trade.client_id)
        base_im = base_im_res["im_base_inr"]
    else:
        base_exp = 0.0
        base_im = 0.0
        
    # 2. Add proposed trade
    new_trade = pd.DataFrame([{
        "trade_id": "PROPOSED",
        "client_id": trade.client_id,
        "netting_set_id": f"NS_{trade.client_id}_01",
        "product_type": trade.product_type,
        "direction": trade.direction,
        "notional_inr": trade.notional_inr,
        "params": json.dumps(trade.params),
        "status": "live",
        "trade_date": str(as_of)
    }])
    client_trades_new = pd.concat([client_trades, new_trade], ignore_index=True) if not client_trades.empty else new_trade
    
    # Value new
    new_vals = value_all_trades(client_trades_new, market, as_of, db_path)
    new_netting = compute_netting_set_exposure(new_vals, client_trades_new)
    new_netting_row = new_netting[new_netting["netting_set_id"] == f"NS_{trade.client_id}_01"]
    new_exp = float(new_netting_row["exposure_inr"].iloc[0]) if not new_netting_row.empty else 0.0
    
    new_im_res = engine_im._compute_product_im(trade.product_type, client_trades_new[client_trades_new["product_type"] == trade.product_type], client_trades_new[client_trades_new["product_type"] == trade.product_type]["notional_inr"].sum(), market, None, trade.client_id)
    new_im = new_im_res["im_base_inr"]
    
    return max(0.0, new_im - base_im), max(0.0, new_exp - base_exp)


@app.get("/clients")
async def list_clients():
    """List all active clients."""
    clients_df = load_clients("data/collateraliq.db")
    return clients_df[["client_id", "name", "archetype", "credit_limit_inr"]].to_dict(orient="records")


@app.get("/exposure/{client_id}")
async def get_exposure(client_id: str):
    """Get latest exposure for a client."""
    engine = get_engine("data/collateraliq.db")
    with engine.connect() as conn:
        from sqlalchemy import text
        row = conn.execute(
            text("""
                SELECT date, exposure_inr, collateralised_exposure_inr, ead_sa_ccr_inr
                FROM exposures WHERE client_id = :cid ORDER BY date DESC LIMIT 1
            """),
            {"cid": client_id},
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No exposure data found")
    return {"client_id": client_id, "date": row[0], "exposure_inr": row[1],
            "collateralised_exposure_inr": row[2], "ead_sa_ccr_inr": row[3]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
