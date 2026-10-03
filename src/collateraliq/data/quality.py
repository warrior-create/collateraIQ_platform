"""
Data quality checks for CollateralIQ.

Checks run after every ingest and before the daily pricing run.
Results are written to dq_log and can block pipeline if is_blocking=True.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from collateraliq.data.db import get_engine

logger = logging.getLogger(__name__)


class DQCheckFailed(Exception):
    """Raised when a blocking DQ check fails."""


def run_all_dq_checks(
    as_of: date,
    db_path: str = "data/collateraliq.db",
) -> list[dict[str, Any]]:
    """Run all DQ checks for a given date. Returns list of check results."""
    engine = get_engine(db_path)
    results = []

    checks = [
        _check_market_data_completeness,
        _check_stale_prices,
        _check_duplicate_trades,
        _check_missing_csa,
        _check_negative_notionals,
        _check_trade_count_reconciliation,
        _check_mtm_reasonableness,
    ]

    for check_fn in checks:
        result = check_fn(as_of, engine)
        results.append(result)
        _log_dq_result(result, engine)
        if result["is_blocking"] and result["records_failed"] > 0:
            raise DQCheckFailed(
                f"Blocking DQ check failed: {result['check_name']}\n"
                f"Details: {result['failure_details']}"
            )

    passed = sum(1 for r in results if r["records_failed"] == 0)
    logger.info(f"DQ checks complete: {passed}/{len(results)} passed")
    return results


def _check_market_data_completeness(as_of: date, engine) -> dict:
    """Ensure key tickers have data for the as_of date."""
    required_tickers = ["^NSEI", "^INDIAVIX", "USDINR=X", "RELIANCE.NS", "TCS.NS"]
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT ticker FROM market_data
                WHERE date = :d AND series = 'close'
                  AND ticker IN ('{}')
            """.format("','".join(required_tickers))),
            {"d": str(as_of)},
        )
        found = {row[0] for row in result}

    missing = set(required_tickers) - found
    return {
        "check_name": "market_data_completeness",
        "table_name": "market_data",
        "records_checked": len(required_tickers),
        "records_failed": len(missing),
        "failure_details": {"missing_tickers": list(missing)} if missing else {},
        "severity": "critical",
        "is_blocking": len(missing) > 2,
    }


def _check_stale_prices(as_of: date, engine) -> dict:
    """Flag prices that haven't been updated in more than 5 business days."""
    cutoff = pd.bdate_range(end=as_of, periods=6)[0].date()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT ticker, MAX(date) as last_date
                FROM market_data
                WHERE series = 'close'
                GROUP BY ticker
                HAVING MAX(date) < :cutoff
            """),
            {"cutoff": str(cutoff)},
        )
        stale = [{"ticker": row[0], "last_date": row[1]} for row in result]

    return {
        "check_name": "stale_prices",
        "table_name": "market_data",
        "records_checked": None,
        "records_failed": len(stale),
        "failure_details": {"stale_tickers": stale[:10]},
        "severity": "high",
        "is_blocking": False,
    }


def _check_duplicate_trades(as_of: date, engine) -> dict:
    """Check for duplicate trade IDs in the trades table."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT trade_id, COUNT(*) as cnt
                FROM trades
                GROUP BY trade_id
                HAVING cnt > 1
            """)
        )
        dupes = [{"trade_id": row[0], "count": row[1]} for row in result]

    return {
        "check_name": "duplicate_trades",
        "table_name": "trades",
        "records_checked": None,
        "records_failed": len(dupes),
        "failure_details": {"duplicates": dupes},
        "severity": "critical",
        "is_blocking": len(dupes) > 0,
    }


def _check_missing_csa(as_of: date, engine) -> dict:
    """Check every active client has a CSA record."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT c.client_id FROM clients c
                LEFT JOIN csa_terms t ON c.client_id = t.client_id
                WHERE c.is_active = 1 AND t.csa_id IS NULL
            """)
        )
        missing = [row[0] for row in result]

    return {
        "check_name": "missing_csa",
        "table_name": "clients/csa_terms",
        "records_checked": None,
        "records_failed": len(missing),
        "failure_details": {"clients_without_csa": missing},
        "severity": "critical",
        "is_blocking": len(missing) > 0,
    }


def _check_negative_notionals(as_of: date, engine) -> dict:
    """Check no trade has a negative (absolute) notional."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT trade_id, notional_inr FROM trades
                WHERE notional_inr <= 0
            """)
        )
        bad = [{"trade_id": row[0], "notional": row[1]} for row in result]

    return {
        "check_name": "negative_notionals",
        "table_name": "trades",
        "records_checked": None,
        "records_failed": len(bad),
        "failure_details": {"bad_trades": bad},
        "severity": "critical",
        "is_blocking": len(bad) > 0,
    }


def _check_trade_count_reconciliation(as_of: date, engine) -> dict:
    """Reconcile trade count between trades table and valuations."""
    with engine.connect() as conn:
        active_trades = conn.execute(
            text("SELECT COUNT(*) FROM trades WHERE status='active' AND trade_date <= :d AND maturity_date >= :d"),
            {"d": str(as_of)},
        ).scalar() or 0

        valued_trades = conn.execute(
            text("SELECT COUNT(*) FROM valuations WHERE date = :d"),
            {"d": str(as_of)},
        ).scalar() or 0

    diff = abs(active_trades - valued_trades)
    return {
        "check_name": "trade_count_reconciliation",
        "table_name": "trades/valuations",
        "records_checked": active_trades,
        "records_failed": diff,
        "failure_details": {"active_trades": active_trades, "valued_trades": valued_trades, "diff": diff},
        "severity": "high" if diff > 0 else "info",
        "is_blocking": False,  # Might legitimately differ on first run
    }


def _check_mtm_reasonableness(as_of: date, engine) -> dict:
    """Check for extreme MTM values (> 5× notional, likely pricing error)."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT v.trade_id, v.mtm_inr, t.notional_inr
                FROM valuations v
                JOIN trades t ON v.trade_id = t.trade_id
                WHERE v.date = :d
                  AND ABS(v.mtm_inr) > 5 * t.notional_inr
            """),
            {"d": str(as_of)},
        )
        extremes = [{"trade_id": row[0], "mtm": row[1], "notional": row[2]} for row in result]

    return {
        "check_name": "mtm_reasonableness",
        "table_name": "valuations",
        "records_checked": None,
        "records_failed": len(extremes),
        "failure_details": {"extreme_mtms": extremes[:5]},
        "severity": "high",
        "is_blocking": False,
    }


def _log_dq_result(result: dict, engine) -> None:
    """Write DQ check result to dq_log table."""
    import json as _json
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO dq_log
                    (date, check_name, table_name, records_checked, records_failed,
                     failure_details, severity, is_blocking)
                VALUES
                    (date('now'), :check_name, :table_name, :records_checked,
                     :records_failed, :failure_details, :severity, :is_blocking)
            """),
            {
                "check_name": result["check_name"],
                "table_name": result.get("table_name"),
                "records_checked": result.get("records_checked"),
                "records_failed": result["records_failed"],
                "failure_details": _json.dumps(result.get("failure_details", {})),
                "severity": result.get("severity", "info"),
                "is_blocking": 1 if result.get("is_blocking") else 0,
            },
        )
