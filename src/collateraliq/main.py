"""
Command Line Interface (CLI) for CollateralIQ.

Provides entry points for:
- Database initialisation
- Running the daily monitoring pipeline
- Running IM adequacy backtesting
- Generating client synthetic data
- Running what-if scenarios
- Starting the pre-trade API
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import Optional

from collateraliq.data.db import init_db
from collateraliq.data.ingest import load_market_data, ingest_equity_data, ingest_fred_data
from collateraliq.clients.generator import generate_clients
from collateraliq.clients.trade_lifecycle import generate_trade_book, get_active_trades
from collateraliq.monitoring.daily_run import DailyRunner
from collateraliq.adequacy.im_backtest import run_im_backtest, compute_backtest_summary
from collateraliq.change.what_if import WhatIfEngine
from collateraliq.clients.generator import load_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def init_command(args: argparse.Namespace) -> None:
    """Initialise database and reference data."""
    logger.info(f"Initialising database at {args.db_path}...")
    init_db(args.db_path)
    logger.info("Generating synthetic clients...")
    clients = generate_clients(db_path=args.db_path)
    
    # Generate 1 year of synthetic market data history for FHS IM
    end_date = date.fromisoformat(args.date)
    start_date = end_date - timedelta(days=365)
    
    logger.info("Generating trade book...")
    generate_trade_book(clients, start_date, end_date, db_path=args.db_path)
    logger.info(f"Ingesting real market data from {start_date} to {end_date} via yfinance and FRED...")
    ingest_equity_data(str(start_date), str(end_date), args.db_path)
    ingest_fred_data(str(start_date), str(end_date), args.db_path)
    logger.info("Initialisation complete.")


def run_daily_command(args: argparse.Namespace) -> None:
    """Run the daily monitoring pipeline."""
    as_of = date.fromisoformat(args.date)
    logger.info(f"Running daily pipeline for {as_of}...")
    
    runner = DailyRunner(db_path=args.db_path)
    try:
        summary = runner.run(as_of, skip_dq=args.skip_dq)
        logger.info(f"Run {summary['run_id']} completed successfully.")
        logger.info(f"Total Exposure: ₹{summary['total_exposure_inr']/1e6:.1f}M")
        logger.info(f"Total IM Held: ₹{summary['total_im_inr']/1e6:.1f}M")
        logger.info(f"Critical Alerts: {summary['n_critical_alerts']}")
    except Exception as e:
        logger.error(f"Run failed: {e}")
        sys.exit(1)


def backtest_command(args: argparse.Namespace) -> None:
    """Run IM adequacy backtesting."""
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    logger.info(f"Running backtest from {start_date} to {end_date}...")
    
    df = run_im_backtest(start_date, end_date, args.db_path, args.confidence)
    summary = compute_backtest_summary(df, args.confidence)
    
    if not summary:
        logger.warning("No backtest results (maybe no breaches or no data).")
        return
        
    overall = summary["overall"]
    logger.info("=== Backtest Summary ===")
    logger.info(f"Observations: {overall['n_obs']}")
    logger.info(f"Breaches: {overall['n_breaches']}")
    logger.info(f"Breach Rate: {overall['breach_rate']*100:.2f}% (Expected: {overall['expected_breach_rate']*100:.2f}%)")
    logger.info(f"Kupiec p-value: {overall['kupiec_test'].get('p_value', 'N/A')}")
    if overall['kupiec_test'].get('reject_h0'):
        logger.warning("⚠️ Model REJECTED by Kupiec test (breach rate too high/low).")
    else:
        logger.info("✅ Model PASSED Kupiec test.")


def whatif_command(args: argparse.Namespace) -> None:
    """Run a what-if scenario."""
    import yaml
    from pathlib import Path
    
    as_of = date.fromisoformat(args.date)
    config_path = Path(args.scenario)
    if not config_path.exists():
        logger.error(f"Scenario config {args.scenario} not found.")
        sys.exit(1)
        
    scenario_config = yaml.safe_load(config_path.read_text())
    logger.info(f"Running what-if scenario: {scenario_config.get('name')} for {as_of}...")
    
    engine = WhatIfEngine(db_path=args.db_path)
    
    # Load required data
    clients_df = load_clients(args.db_path)
    trades_df = get_active_trades(as_of, args.db_path)
    
    # Dummy market load for demo
    runner = DailyRunner(db_path=args.db_path)
    market = runner._load_market(as_of)
    
    result = engine.run(scenario_config, as_of, clients_df, trades_df, market)
    
    logger.info("=== What-If Impact ===")
    logger.info(f"IM Before: ₹{result['total_im_before_inr']/1e6:.1f}M")
    logger.info(f"IM After:  ₹{result['total_im_after_inr']/1e6:.1f}M")
    logger.info(f"Change:    {result['total_im_change_pct']:+.2f}%")
    logger.info(f"Clients needing extra margin call: {result['n_clients_extra_call']}")


def api_command(args: argparse.Namespace) -> None:
    """Start the pre-trade API."""
    import uvicorn
    logger.info(f"Starting API on {args.host}:{args.port}...")
    uvicorn.run("collateraliq.api.pretrade_quote:app", host=args.host, port=args.port, reload=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CollateralIQ CLI")
    parser.add_argument("--db-path", default="data/collateraliq.db", help="Path to SQLite database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Init
    p_init = subparsers.add_parser("init", help="Initialise DB and synthetic data")
    p_init.add_argument("--date", default=str(date.today()), help="As-of date for init")
    p_init.add_argument("--clients", type=int, default=25, help="Number of clients to generate")

    # Run Daily
    p_run = subparsers.add_parser("run-daily", help="Run daily monitoring pipeline")
    p_run.add_argument("--date", required=True, help="As-of date (YYYY-MM-DD)")
    p_run.add_argument("--skip-dq", action="store_true", help="Skip data quality checks")

    # Backtest
    p_bt = subparsers.add_parser("backtest", help="Run IM adequacy backtest")
    p_bt.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    p_bt.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    p_bt.add_argument("--confidence", type=float, default=0.99, help="IM confidence level")

    # What-If
    p_wi = subparsers.add_parser("what-if", help="Run model change impact scenario")
    p_wi.add_argument("--date", required=True, help="As-of date (YYYY-MM-DD)")
    p_wi.add_argument("--scenario", required=True, help="Path to scenario YAML config")

    # API
    p_api = subparsers.add_parser("api", help="Start pre-trade API")
    p_api.add_argument("--host", default="0.0.0.0", help="API host")
    p_api.add_argument("--port", type=int, default=8001, help="API port")

    args = parser.parse_args()

    if args.command == "init":
        init_command(args)
    elif args.command == "run-daily":
        run_daily_command(args)
    elif args.command == "backtest":
        backtest_command(args)
    elif args.command == "what-if":
        whatif_command(args)
    elif args.command == "api":
        api_command(args)


if __name__ == "__main__":
    main()
