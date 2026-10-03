"""
Data ingestion module: yfinance + FRED market data → SQLite/PostgreSQL.

Data sourced:
- NSE equities, Nifty 50, India VIX, USDINR: yfinance
- US Treasury rates, credit spreads (ICE BofA OAS): FRED
- INR repo rate (proxy for short-term INR rate): FRED
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import text

from collateraliq.data.db import get_engine, get_session
from collateraliq.data.audit import log_audit

logger = logging.getLogger(__name__)

# ─── Tickers ────────────────────────────────────────────────────────────────

EQUITY_TICKERS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "HUL",
    "ITC.NS": "ITC",
    "SBIN.NS": "SBI",
    "BHARTIARTL.NS": "Bharti Airtel",
    "AXISBANK.NS": "Axis Bank",
    "^NSEI": "Nifty 50",
    "^INDIAVIX": "India VIX",
    "USDINR=X": "USDINR Spot",
}

FRED_SERIES = {
    "DGS10": "US 10Y Treasury",
    "DGS2": "US 2Y Treasury",
    "SOFR": "SOFR rate",
    "INDIRLTLT01STM": "India 10Y Gov Bond Yield",
    "BAMLC0A0CM": "ICE BofA IG Corp OAS",
    "BAMLH0A0HYM2": "ICE BofA HY Corp OAS",
    "BAMLC0A4CBBB": "ICE BofA BBB Corp OAS",
}


def ingest_equity_data(
    start: str,
    end: str,
    db_path: str = "data/collateraliq.db",
) -> pd.DataFrame:
    """Download equity, index, VIX and FX data from yfinance."""
    logger.info(f"Ingesting equity/index data from {start} to {end}")

    tickers = list(EQUITY_TICKERS.keys())
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    engine = get_engine(db_path)
    records = []

    for ticker in tickers:
        try:
            if len(tickers) > 1:
                close = raw["Close"][ticker].dropna()
                volume = raw["Volume"][ticker].dropna() if ticker not in ["^NSEI", "^INDIAVIX", "USDINR=X"] else pd.Series(dtype=float)
            else:
                close = raw["Close"].dropna()
                volume = raw["Volume"].dropna()

            for dt, val in close.items():
                records.append({
                    "date": dt.date(),
                    "ticker": ticker,
                    "series": "close",
                    "value": float(val),
                    "source": "yfinance",
                    "is_interpolated": 0,
                })
            for dt, val in volume.items():
                if not np.isnan(val):
                    records.append({
                        "date": dt.date(),
                        "ticker": ticker,
                        "series": "volume",
                        "value": float(val),
                        "source": "yfinance",
                        "is_interpolated": 0,
                    })
        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        _upsert_market_data(df, engine)
        logger.info(f"Ingested {len(df)} equity/FX data points")
    return df


def ingest_fred_data(
    start: str,
    end: str,
    fred_api_key: str | None = None,
    db_path: str = "data/collateraliq.db",
) -> pd.DataFrame:
    """Download rate and credit spread data from FRED."""
    api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logger.warning("No FRED API key found. Using synthetic rate data.")
        return _synthetic_rate_data(start, end, db_path)

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except Exception as e:
        logger.warning(f"FRED API error: {e}. Using synthetic data.")
        return _synthetic_rate_data(start, end, db_path)

    engine = get_engine(db_path)
    records = []

    for series_id, description in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, start, end)
            data = data.dropna()
            for dt, val in data.items():
                records.append({
                    "date": dt.date() if hasattr(dt, "date") else dt,
                    "ticker": series_id,
                    "series": "rate" if "DGS" in series_id or "SOFR" in series_id or "INDIRLTLT" in series_id else "spread",
                    "value": float(val) / 100 if "DGS" in series_id or "SOFR" in series_id else float(val),
                    "source": "FRED",
                    "is_interpolated": 0,
                })
            logger.info(f"Downloaded {len(data)} points for {series_id}")
        except Exception as e:
            logger.warning(f"Failed to download {series_id}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        _upsert_market_data(df, engine)
    return df


def _synthetic_rate_data(start: str, end: str, db_path: str) -> pd.DataFrame:
    """Generate synthetic rate data when FRED is unavailable."""
    logger.info("Generating synthetic rate/spread data")
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start, end)
    records = []

    # INR 10Y yield: mean-reverting around 7%
    inr_10y = _gbm_path(0.07, 0.005, 0.30, len(dates), rng)
    # USD 10Y: mean-reverting around 4%
    usd_10y = _gbm_path(0.04, 0.005, 0.20, len(dates), rng)
    # SOFR: mean-reverting around 5%
    sofr = _gbm_path(0.05, 0.003, 0.15, len(dates), rng)
    # IG OAS: mean-reverting around 100bps
    ig_oas = _gbm_path(100, 5, 0.25, len(dates), rng) / 10000
    # HY OAS: mean-reverting around 400bps
    hy_oas = _gbm_path(400, 20, 0.30, len(dates), rng) / 10000

    for i, dt in enumerate(dates):
        for ticker, val, series in [
            ("INDIRLTLT01STM", inr_10y[i], "rate"),
            ("DGS10", usd_10y[i], "rate"),
            ("SOFR", sofr[i], "rate"),
            ("BAMLC0A0CM", ig_oas[i], "spread"),
            ("BAMLH0A0HYM2", hy_oas[i], "spread"),
        ]:
            records.append({
                "date": dt.date(),
                "ticker": ticker,
                "series": series,
                "value": float(max(val, 0.001)),
                "source": "synthetic",
                "is_interpolated": 0,
            })

    df = pd.DataFrame(records)
    engine = get_engine(db_path)
    _upsert_market_data(df, engine)
    return df


def _gbm_path(mu: float, sigma: float, vol: float, n: int, rng) -> np.ndarray:
    """Simple mean-reverting process for synthetic rates."""
    path = np.zeros(n)
    path[0] = mu
    kappa = 0.10  # mean reversion speed
    for i in range(1, n):
        drift = kappa * (mu - path[i - 1]) / 252
        shock = sigma * vol * rng.standard_normal() / np.sqrt(252)
        path[i] = max(path[i - 1] + drift + shock, 0.0001)
    return path


def _upsert_market_data(df: pd.DataFrame, engine) -> None:
    """Upsert market data rows (INSERT OR REPLACE for SQLite)."""
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO market_data
                        (date, ticker, series, value, source, is_interpolated)
                    VALUES
                        (:date, :ticker, :series, :value, :source, :is_interpolated) ON CONFLICT DO NOTHING
                """),
                {
                    "date": str(row["date"]),
                    "ticker": row["ticker"],
                    "series": row["series"],
                    "value": row["value"],
                    "source": row["source"],
                    "is_interpolated": row.get("is_interpolated", 0),
                },
            )


def load_market_data(
    date_from: str,
    date_to: str,
    tickers: list[str] | None = None,
    series: str | None = None,
    db_path: str = "data/collateraliq.db",
) -> pd.DataFrame:
    """Load market data from DB into a wide-format DataFrame."""
    engine = get_engine(db_path)
    q = "SELECT date, ticker, series, value FROM market_data WHERE date BETWEEN :d1 AND :d2"
    params: dict = {"d1": date_from, "d2": date_to}

    if tickers:
        placeholders = ",".join(f":t{i}" for i in range(len(tickers)))
        q += f" AND ticker IN ({placeholders})"
        params.update({f"t{i}": t for i, t in enumerate(tickers)})
    if series:
        q += " AND series = :series"
        params["series"] = series

    q += " ORDER BY date, ticker"

    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn, params=params, parse_dates=["date"])

    if df.empty:
        return df

    # Pivot to wide format: columns = (ticker, series)
    wide = df.pivot_table(index="date", columns=["ticker", "series"], values="value")
    wide.columns = ["_".join(c) for c in wide.columns]
    return wide


def data_hash(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of a DataFrame for lineage tracking."""
    buf = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


def sync_synthetic_market_data(start: date, end: date, db_path: str) -> None:
    """Generate both synthetic rates and equity data to avoid API calls."""
    start_str = str(start)
    end_str = str(end)
    _synthetic_rate_data(start_str, end_str, db_path)
    
    # Generate synthetic equity data
    logger.info("Generating synthetic equity data")
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start_str, end_str)
    records = []
    
    for ticker in list(EQUITY_TICKERS.keys()):
        # Random walk for each equity
        start_price = 22000 if ticker == "^NSEI" else 1000
        vol = 0.20
        path = _gbm_path(start_price, 0.05, vol, len(dates), rng)
        
        for i, dt in enumerate(dates):
            records.append({
                "date": dt.date(),
                "ticker": ticker,
                "series": "close",
                "value": float(path[i]),
                "source": "synthetic",
                "is_interpolated": 0,
            })
            
    df = pd.DataFrame(records)
    engine = get_engine(db_path)
    _upsert_market_data(df, engine)
