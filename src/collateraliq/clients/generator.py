"""
Synthetic client book generator.

Generates 25 clients across 5 archetypes with seeded, reproducible parameters.
Each client gets a full trade book, CSA terms, netting sets and credit limits.

All data is synthetic. Seeded with seed=42 for reproducibility.
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
from collateraliq.data.audit import log_audit

logger = logging.getLogger(__name__)

ARCHETYPE_NAMES = {
    "leveraged_hedge_fund": [
        "Titan Capital LP", "Apex Alpha Fund", "Quantum Macro Master",
        "Vertex Absolute Return", "Meridian Multi-Strat", "Orion Leverage Fund",
    ],
    "long_only_pension": [
        "Bharat Pension Trustees", "National Life Endowment",
        "Provident Growth Fund", "Sovereign Annuity Trust", "Heritage Pension Corp",
    ],
    "corporate_hedger": [
        "Inditech Manufacturing Ltd", "SunExport Trading Co",
        "Cascade Commodities", "IndoGlobal Shipping", "Precision Electronics Ltd",
    ],
    "family_office": [
        "Mehta Family Holdings", "Patel Capital Partners",
        "Agarwal Wealth Management", "Sharma Private Office", "Gupta Dynasty Fund",
    ],
    "macro_fund": [
        "Global Rates Capital", "Emerging Alpha Partners",
        "Crosscurrents Macro", "Pacific EM Strategy",
    ],
}


def generate_clients(
    seed: int = 42,
    config_path: str = "config/archetypes.yaml",
    db_path: str = "data/collateraliq.db",
) -> list[dict]:
    """
    Generate 25 synthetic clients and write to DB.

    Returns list of client dicts.
    """
    rng = np.random.default_rng(seed)
    config = yaml.safe_load(Path(config_path).read_text())
    engine = get_engine(db_path)

    clients = []
    csa_records = []
    netting_sets = []

    for archetype, params in config["archetypes"].items():
        count = params["count"]
        names = ARCHETYPE_NAMES[archetype][:count]

        for i, name in enumerate(names):
            client_id = f"C{archetype[:3].upper()}{i+1:02d}"
            seed_offset = int(rng.integers(1, 1000))

            # Credit limit: scaled by notional range
            notional_lo, notional_hi = params["notional_range_inr"]
            credit_limit = float(rng.uniform(notional_lo * 2, notional_hi * 3))
            ratings = ["AAA", "AA+", "AA", "A+", "A", "BBB+", "BBB"]
            rating_weights = [0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10]
            rating = str(rng.choice(ratings, p=rating_weights))

            # Onboard date: random in 2020-2022
            onboard_days = int(rng.integers(0, 365 * 2))
            onboard_date = date(2020, 1, 1) + timedelta(days=onboard_days)

            client = {
                "client_id": client_id,
                "name": name,
                "archetype": archetype,
                "credit_limit_inr": credit_limit,
                "rating": rating,
                "onboard_date": str(onboard_date),
                "is_active": 1,
                "seed_offset": seed_offset,
            }
            clients.append(client)

            # CSA terms
            threshold_jitter = float(rng.uniform(0.8, 1.2))
            mta_jitter = float(rng.uniform(0.8, 1.2))

            csa_id = f"CSA_{client_id}"
            netting_set_id = f"NS_{client_id}_01"

            csa = {
                "csa_id": csa_id,
                "client_id": client_id,
                "netting_set_id": netting_set_id,
                "threshold_inr": params["threshold"] * threshold_jitter,
                "mta_inr": params["mta"] * mta_jitter,
                "independent_amount_inr": params.get("independent_amount", 0),
                "rounding_inr": params["rounding"],
                "call_frequency": params["call_frequency"],
                "mpor_days": params["mpor"],
                "eligible_collateral": json.dumps(params["eligible_collateral"]),
                "effective_date": str(onboard_date),
                "termination_date": None,
            }
            csa_records.append(csa)

            netting_sets.append({
                "netting_set_id": netting_set_id,
                "client_id": client_id,
                "csa_id": csa_id,
                "description": f"Primary netting set for {name}",
                "currency": "INR",
            })

    # Write to DB
    _write_clients(clients, engine)
    _write_csa_terms(csa_records, engine)
    _write_netting_sets(netting_sets, engine)
    _write_initial_inventory(clients, engine, rng)

    log_audit(engine, "gen_clients", run_id=f"seed_{seed}")
    logger.info(f"Generated {len(clients)} clients, {len(csa_records)} CSAs")
    return clients


def _write_clients(clients: list[dict], engine) -> None:
    with engine.begin() as conn:
        for c in clients:
            conn.execute(
                text("""
                    INSERT INTO clients
                        (client_id, name, archetype, credit_limit_inr, rating,
                         onboard_date, is_active, seed_offset)
                    VALUES
                        (:client_id, :name, :archetype, :credit_limit_inr, :rating,
                         :onboard_date, :is_active, :seed_offset) ON CONFLICT DO NOTHING
                """),
                c,
            )


def _write_csa_terms(csa_records: list[dict], engine) -> None:
    with engine.begin() as conn:
        for csa in csa_records:
            conn.execute(
                text("""
                    INSERT INTO csa_terms
                        (csa_id, client_id, netting_set_id, threshold_inr, mta_inr,
                         independent_amount_inr, rounding_inr, call_frequency, mpor_days,
                         eligible_collateral, effective_date, termination_date)
                    VALUES
                        (:csa_id, :client_id, :netting_set_id, :threshold_inr, :mta_inr,
                         :independent_amount_inr, :rounding_inr, :call_frequency, :mpor_days,
                         :eligible_collateral, :effective_date, :termination_date) ON CONFLICT DO NOTHING
                """),
                csa,
            )


def _write_netting_sets(netting_sets: list[dict], engine) -> None:
    with engine.begin() as conn:
        for ns in netting_sets:
            conn.execute(
                text("""
                    INSERT INTO netting_sets
                        (netting_set_id, client_id, csa_id, description, currency)
                    VALUES
                        (:netting_set_id, :client_id, :csa_id, :description, :currency) ON CONFLICT DO NOTHING
                """),
                ns,
            )


def _write_initial_inventory(clients: list[dict], engine, rng) -> None:
    from datetime import date
    records = []
    today = str(date(2026, 9, 30))
    for c in clients:
        base = c["credit_limit_inr"] * rng.uniform(0.1, 0.5)
        records.append({"date": today, "client_id": c["client_id"], "asset_type": "cash_inr", "notional_inr": base * rng.uniform(0.5, 1.0), "haircut": 0.0, "eligible_value_inr": base * rng.uniform(0.5, 1.0), "is_segregated": 0})
        records.append({"date": today, "client_id": c["client_id"], "asset_type": "gsec_short", "notional_inr": base * rng.uniform(0.5, 1.5), "haircut": 0.02, "eligible_value_inr": base * rng.uniform(0.5, 1.5) * 0.98, "is_segregated": 0})
        records.append({"date": today, "client_id": c["client_id"], "asset_type": "equity_nifty50", "notional_inr": base * rng.uniform(1.0, 3.0), "haircut": 0.15, "eligible_value_inr": base * rng.uniform(1.0, 3.0) * 0.85, "is_segregated": 0})

    with engine.begin() as conn:
        for r in records:
            conn.execute(
                text("""
                    INSERT INTO collateral_inventory
                        (date, client_id, asset_type, notional_inr, haircut, eligible_value_inr, is_segregated)
                    VALUES
                        (:date, :client_id, :asset_type, :notional_inr, :haircut, :eligible_value_inr, :is_segregated)
                """),
                r,
            )

def load_clients(db_path: str = "data/collateraliq.db") -> pd.DataFrame:
    """Load all active clients with their CSA terms."""
    engine = get_engine(db_path)
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT c.*, t.threshold_inr, t.mta_inr, t.independent_amount_inr,
                       t.rounding_inr, t.call_frequency, t.mpor_days,
                       t.eligible_collateral, t.netting_set_id, t.csa_id
                FROM clients c
                LEFT JOIN csa_terms t ON c.client_id = t.client_id
                WHERE c.is_active = 1
            """),
            conn,
        )
    return df
