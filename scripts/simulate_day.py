"""
End-to-End Simulation Script for CollateralIQ

This script runs the entire lifecycle for a single day:
1. Initialises the database and synthetic clients/market data
2. Runs the daily monitoring pipeline
3. Runs an IM adequacy backtest for the past week
4. Runs a what-if scenario analysis
"""

import sys
import argparse
from datetime import date, timedelta
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collateraliq.main import init_command, run_daily_command, backtest_command, whatif_command

class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def main():
    today = date(2026, 9, 30)
    db_path = "data/collateraliq.db"
    
    # Ensure data dir exists
    Path(db_path).parent.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print(f"1. Initialising Database & Data (As of {today})")
    print("="*60)
    init_args = DummyArgs(db_path=db_path, date=str(today), clients=10)
    init_command(init_args)
    
    print("\n" + "="*60)
    print(f"2. Running Daily Pipeline (As of {today})")
    print("="*60)
    # Generate some market data for yesterday as well for attribution
    from collateraliq.data.ingest import sync_synthetic_market_data
    sync_synthetic_market_data(today - timedelta(days=2), today, db_path)
    
    run_args = DummyArgs(db_path=db_path, date=str(today), skip_dq=False)
    run_daily_command(run_args)
    
    print("\n" + "="*60)
    print("3. Running What-If Scenario Analysis")
    print("="*60)
    wi_args = DummyArgs(db_path=db_path, date=str(today), scenario="config/whatif_scenario1.yaml")
    whatif_command(wi_args)
    
    print("\n" + "="*60)
    print("4. Running IM Backtest (Simulated historical data)")
    print("="*60)
    # We will simulate the last 10 days by running the daily pipeline for each
    for d in range(10, 0, -1):
        sim_date = today - timedelta(days=d)
        print(f"   Simulating {sim_date}...")
        try:
            # Only run exposure computation to populate tables for backtest
            run_daily_command(DummyArgs(db_path=db_path, date=str(sim_date), skip_dq=True))
        except Exception:
            pass
            
    bt_args = DummyArgs(db_path=db_path, start_date=str(today - timedelta(days=10)), end_date=str(today), confidence=0.99)
    backtest_command(bt_args)
    
    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
