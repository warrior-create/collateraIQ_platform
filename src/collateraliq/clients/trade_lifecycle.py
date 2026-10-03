"""
Trade lifecycle management: new trades, amendments, rolls and maturities.

Generates the full multi-year trade book replay from 2022-01-01 to 2026-09-30.
Trade events are seeded so every run produces the same book.

Products supported per archetype:
- equity_financing, gsec_repo, irs, cds_index, equity_options,
  single_stock_options, fx_forward
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text

from collateraliq.data.db import get_engine

logger = logging.getLogger(__name__)

PRODUCT_PARAMS = {
    "equity_financing": {
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
        "typical_tenor_days": 90,
        "roll_frequency_days": 90,
    },
    "gsec_repo": {
        "maturities_years": [2, 5, 10],
        "typical_tenor_days": 7,
        "roll_frequency_days": 7,
    },
    "irs": {
        "tenors_years": [1, 2, 3, 5, 7, 10],
        "typical_tenor_days": 365 * 5,
        "roll_frequency_days": None,  # no roll, runs to maturity
    },
    "cds_index": {
        "index": "ITRAXX_EM_PROXY",
        "typical_tenor_days": 365 * 5,
        "roll_frequency_days": 91,  # quarterly roll
    },
    "equity_options": {
        "tickers": ["^NSEI"],
        "typical_tenor_days": 30,
        "roll_frequency_days": 30,
    },
    "single_stock_options": {
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
        "typical_tenor_days": 30,
        "roll_frequency_days": 30,
    },
    "fx_forward": {
        "pair": "USDINR",
        "typical_tenor_days": 90,
        "roll_frequency_days": 30,
    },
}


def generate_trade_book(
    clients: list[dict],
    start_date: date,
    end_date: date,
    seed: int = 42,
    config_path: str = "config/archetypes.yaml",
    db_path: str = "data/collateraliq.db",
) -> pd.DataFrame:
    """
    Generate full trade book with lifecycle events for all clients.

    Returns DataFrame of all trades.
    """
    rng = np.random.default_rng(seed)
    config = yaml.safe_load(Path(config_path).read_text())
    engine = get_engine(db_path)

    all_trades = []
    all_events = []

    for client in clients:
        client_id = client["client_id"]
        archetype = client["archetype"]
        arch_params = config["archetypes"][archetype]
        netting_set_id = client.get("netting_set_id", f"NS_{client_id}_01")

        products = arch_params["products"]
        notional_lo, notional_hi = arch_params["notional_range_inr"]
        seed_offset = client.get("seed_offset", 0)
        client_rng = np.random.default_rng(seed + seed_offset)

        # Number of trades per product
        n_products = len(products)
        trades_per_product = max(1, int(client_rng.integers(2, 6)))

        for product in products:
            for j in range(trades_per_product):
                trade_id = f"T_{client_id}_{product.upper()}_{j+1:02d}"

                # Trade date: random in first 2 years of replay
                days_offset = int(client_rng.integers(0, 365 * 2))
                trade_date = start_date + timedelta(days=days_offset)
                # Snap to business day
                while trade_date.weekday() >= 5:
                    trade_date += timedelta(days=1)

                notional = float(client_rng.uniform(notional_lo, notional_hi))
                direction = int(client_rng.choice([-1, 1]))

                pp = PRODUCT_PARAMS.get(product, {})
                tenor_days = pp.get("typical_tenor_days", 365)
                # Jitter tenor ±20%
                tenor_days = int(tenor_days * client_rng.uniform(0.8, 1.2))
                maturity_date = trade_date + timedelta(days=tenor_days)

                params = _generate_trade_params(product, notional, client_rng)

                trade = {
                    "trade_id": trade_id,
                    "client_id": client_id,
                    "netting_set_id": netting_set_id,
                    "product_type": product,
                    "direction": direction,
                    "notional_inr": notional,
                    "currency": "INR",
                    "trade_date": str(trade_date),
                    "maturity_date": str(maturity_date),
                    "status": "active",
                    "params": json.dumps(params),
                }
                all_trades.append(trade)

                # Generate lifecycle events
                events = _generate_lifecycle_events(
                    trade_id, trade_date, maturity_date, end_date, product, client_rng
                )
                all_events.extend(events)

    _write_trades(all_trades, engine)
    _write_trade_events(all_events, engine)
    logger.info(f"Generated {len(all_trades)} trades, {len(all_events)} lifecycle events")
    return pd.DataFrame(all_trades)


def _generate_trade_params(product: str, notional: float, rng) -> dict:
    """Generate product-specific parameters."""
    if product == "equity_financing":
        tickers = PRODUCT_PARAMS["equity_financing"]["tickers"]
        return {
            "ticker": str(rng.choice(tickers)),
            "repo_rate": float(rng.uniform(0.06, 0.09)),
            "initial_margin_pct": float(rng.uniform(0.10, 0.25)),
        }
    elif product == "gsec_repo":
        mats = PRODUCT_PARAMS["gsec_repo"]["maturities_years"]
        return {
            "bond_maturity_years": float(rng.choice(mats)),
            "coupon_rate": float(rng.uniform(0.06, 0.08)),
            "repo_rate": float(rng.uniform(0.06, 0.075)),
        }
    elif product == "irs":
        tenors = PRODUCT_PARAMS["irs"]["tenors_years"]
        return {
            "tenor_years": float(rng.choice(tenors)),
            "fixed_rate": float(rng.uniform(0.065, 0.085)),
            "payment_frequency": 2,  # semi-annual
            "is_payer": bool(rng.choice([True, False])),
        }
    elif product == "cds_index":
        return {
            "index": "ITRAXX_EM_PROXY",
            "spread_bps": float(rng.uniform(80, 200)),
            "recovery_rate": 0.40,
            "is_protection_buyer": bool(rng.choice([True, False])),
        }
    elif product in ("equity_options", "single_stock_options"):
        is_nifty = product == "equity_options"
        tickers = ["^NSEI"] if is_nifty else PRODUCT_PARAMS["single_stock_options"]["tickers"]
        moneyness = float(rng.uniform(0.85, 1.15))
        return {
            "ticker": str(rng.choice(tickers)),
            "option_type": str(rng.choice(["call", "put"])),
            "moneyness": moneyness,
            "tenor_years": float(rng.uniform(0.05, 0.25)),
            "contracts": int(rng.integers(1, 50)),
            "lot_size": 50,
        }
    elif product == "fx_forward":
        return {
            "pair": "USDINR",
            "tenor_years": float(rng.uniform(0.08, 0.5)),
            "notional_usd": float(notional / 84),  # Approximate USDINR
        }
    return {}


def _generate_lifecycle_events(
    trade_id: str,
    trade_date: date,
    maturity_date: date,
    end_date: date,
    product: str,
    rng,
) -> list[dict]:
    """Generate roll, amendment and maturity events for a trade."""
    events = []

    # New trade event
    events.append({
        "event_id": f"EV_{trade_id}_NEW",
        "trade_id": trade_id,
        "event_type": "new",
        "event_date": str(trade_date),
        "old_params": None,
        "new_params": None,
        "notes": "Initial trade booking",
    })

    # Roll events for short-tenor products
    pp = PRODUCT_PARAMS.get(product, {})
    roll_freq = pp.get("roll_frequency_days")
    if roll_freq and roll_freq < 365:
        roll_date = trade_date + timedelta(days=roll_freq)
        roll_count = 0
        while roll_date < min(maturity_date, end_date) and roll_count < 20:
            events.append({
                "event_id": f"EV_{trade_id}_ROLL_{roll_count+1:02d}",
                "trade_id": trade_id,
                "event_type": "roll",
                "event_date": str(roll_date),
                "old_params": None,
                "new_params": None,
                "notes": f"Scheduled roll #{roll_count+1}",
            })
            roll_date += timedelta(days=roll_freq)
            roll_count += 1

    # Amendment (random, ~30% of trades)
    if rng.random() < 0.30:
        amend_days = int(rng.integers(30, 180))
        amend_date = trade_date + timedelta(days=amend_days)
        if amend_date < min(maturity_date, end_date):
            events.append({
                "event_id": f"EV_{trade_id}_AMEND",
                "trade_id": trade_id,
                "event_type": "amendment",
                "event_date": str(amend_date),
                "old_params": None,
                "new_params": json.dumps({"notional_change_pct": float(rng.uniform(-0.2, 0.2))}),
                "notes": "Notional amendment",
            })

    # Maturity event
    if maturity_date <= end_date:
        events.append({
            "event_id": f"EV_{trade_id}_MAT",
            "trade_id": trade_id,
            "event_type": "maturity",
            "event_date": str(maturity_date),
            "old_params": None,
            "new_params": None,
            "notes": "Natural maturity",
        })

    return events


def _write_trades(trades: list[dict], engine) -> None:
    with engine.begin() as conn:
        for t in trades:
            conn.execute(
                text("""
                    INSERT INTO trades
                        (trade_id, client_id, netting_set_id, product_type, direction,
                         notional_inr, currency, trade_date, maturity_date, status, params)
                    VALUES
                        (:trade_id, :client_id, :netting_set_id, :product_type, :direction,
                         :notional_inr, :currency, :trade_date, :maturity_date, :status, :params) ON CONFLICT DO NOTHING
                """),
                t,
            )


def _write_trade_events(events: list[dict], engine) -> None:
    with engine.begin() as conn:
        for e in events:
            conn.execute(
                text("""
                    INSERT INTO trade_events
                        (event_id, trade_id, event_type, event_date, old_params, new_params, notes)
                    VALUES
                        (:event_id, :trade_id, :event_type, :event_date, :old_params, :new_params, :notes) ON CONFLICT DO NOTHING
                """),
                e,
            )


def get_active_trades(as_of: date, db_path: str = "data/collateraliq.db") -> pd.DataFrame:
    """Return all trades active on a given date by reconstructing from events."""
    engine = get_engine(db_path)
    with engine.connect() as conn:
        # Get all trades that started on or before as_of
        trades_df = pd.read_sql(
            text("""
                SELECT * FROM trades
                WHERE trade_date <= :d
            """),
            conn,
            params={"d": str(as_of)},
        )
        
        if trades_df.empty:
            return trades_df
            
        # Get all events that happened on or before as_of for these trades
        events_df = pd.read_sql(
            text("""
                SELECT * FROM trade_events
                WHERE event_date <= :d
                ORDER BY event_date ASC, event_id ASC
            """),
            conn,
            params={"d": str(as_of)},
        )

    if not events_df.empty:
        # Apply events to reconstruct state
        for _, event in events_df.iterrows():
            t_id = event["trade_id"]
            e_type = event["event_type"]
            
            if t_id not in trades_df["trade_id"].values:
                continue
                
            idx = trades_df[trades_df["trade_id"] == t_id].index[0]
            
            if e_type == "amendment" and event["new_params"]:
                import json
                try:
                    params = json.loads(event["new_params"])
                    if "notional_change_pct" in params:
                        trades_df.at[idx, "notional_inr"] *= (1.0 + params["notional_change_pct"])
                except Exception:
                    pass
            elif e_type == "termination" or e_type == "maturity":
                trades_df.at[idx, "status"] = "terminated"
                
    # Filter out terminated trades
    active_df = trades_df[trades_df["status"] != "terminated"].copy()
    
    # Check natural maturity date as well just in case maturity event was missed
    active_df = active_df[pd.to_datetime(active_df["maturity_date"]).dt.date >= as_of]
    
    return active_df
