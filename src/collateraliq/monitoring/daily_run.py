"""
Daily run orchestrator.

Runs the full daily pipeline for a given date:
1. Load market data
2. Value all trades
3. Compute exposures (netting, collateral, EE/PFE)
4. Compute IM
5. Run margin calls
6. Run alert engine
7. Generate commentary with attribution
8. Write all results to DB

CLI: collateraliq run-daily --date 2026-09-30
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from collateraliq.data.db import get_engine
from collateraliq.data.ingest import load_market_data
from collateraliq.data.quality import run_all_dq_checks
from collateraliq.clients.generator import load_clients
from collateraliq.clients.trade_lifecycle import get_active_trades
from collateraliq.pricing.engine import value_all_trades
from collateraliq.exposure.netting import compute_netting_set_exposure, compute_client_exposures
from collateraliq.exposure.simulation import compute_ee_pfe
from collateraliq.exposure.cva import compute_cva, compute_sa_ccr
from collateraliq.exposure.sa_ccr_lite import compute_sa_ccr_ead
from collateraliq.margin.im_engine import IMEngine
from collateraliq.monitoring.attribution import compute_attribution, verify_attribution_reconciliation
from collateraliq.monitoring.alerts import AlertEngine
from collateraliq.commentary.generator import generate_daily_commentary
from collateraliq.monitoring.margin_calls import generate_margin_calls

logger = logging.getLogger(__name__)


class DailyRunner:
    """Orchestrates the complete daily pipeline."""

    def __init__(self, db_path: str = "data/collateraliq.db") -> None:
        self.db_path = db_path
        self.engine = get_engine(db_path)
        self.im_engine = IMEngine(db_path=db_path)
        self.alert_engine = AlertEngine(db_path=db_path)

    def run(self, as_of: date, skip_dq: bool = False) -> dict:
        """
        Run full daily pipeline.

        Returns summary dict with key metrics.
        """
        run_id = f"run_{as_of}_{uuid.uuid4().hex[:6]}"
        logger.info(f"Starting daily run for {as_of} [run_id={run_id}]")

        self._log_run_start(run_id, as_of)

        try:
            # Step 1: Load market data
            market = self._load_market(as_of)
            market_yesterday = self._load_market(as_of - timedelta(days=1))
            logger.info(f"Market data loaded: {len(market)} series")

            # Step 2: DQ checks
            if not skip_dq:
                try:
                    run_all_dq_checks(as_of, self.db_path)
                except Exception as e:
                    logger.warning(f"DQ check warning: {e}")

            # Step 3: Load clients and trades
            clients_df = load_clients(self.db_path)
            trades_df = get_active_trades(as_of, self.db_path)
            logger.info(f"Loaded {len(clients_df)} clients, {len(trades_df)} active trades")

            # Step 4: Valuation
            valuations_df = value_all_trades(trades_df, market, as_of, self.db_path)
            logger.info(f"Valued {len(valuations_df)} trades")

            # Step 5: Netting and exposure
            netting_exp = compute_netting_set_exposure(valuations_df, trades_df)

            # Step 6: IM calculation
            im_results = self.im_engine.run(as_of, clients_df, trades_df, market)
            logger.info(f"IM computed for {clients_df['client_id'].nunique()} clients")

            # Step 7: VM ledger (simplified — use previous day + call)
            vm_df = self._get_vm_ledger(as_of)

            # Full exposure calculation
            csa_df = clients_df[["client_id", "threshold_inr", "mta_inr",
                                  "independent_amount_inr", "rounding_inr", "mpor_days"]]
            exposures_df = compute_client_exposures(netting_exp, trades_df, csa_df, vm_df, im_results, pd.DataFrame())

            # Pathwise simulation for EPE / PFE
            ee_col, epe_col, pfe95_col, pfe99_col = [], [], [], []
            for _, c_row in exposures_df.iterrows():
                cid = c_row["client_id"]
                c_trades = trades_df[trades_df["client_id"] == cid]
                if not c_trades.empty:
                    sim_res = compute_ee_pfe(cid, c_trades, market, n_paths=100) # 100 paths for speed
                    ee_col.append(sim_res["ee_profile"][0]) # Spot EE
                    epe_col.append(sim_res["epe"])
                    pfe95_col.append(sim_res["pfe_95"])
                    pfe99_col.append(sim_res["pfe_99"])
                else:
                    ee_col.append(0.0)
                    epe_col.append(0.0)
                    pfe95_col.append(0.0)
                    pfe99_col.append(0.0)
            
            exposures_df["ee_inr"] = ee_col
            exposures_df["epe_inr"] = epe_col
            exposures_df["pfe_95_inr"] = pfe95_col
            exposures_df["pfe_99_inr"] = pfe99_col

            # Step 8: Write exposures to DB
            self._write_exposures(as_of, exposures_df, im_results, vm_df)

            # Step 9: Attribution (per client)
            attributions = {}
            yesterday_trades = get_active_trades(as_of - timedelta(days=1), self.db_path)
            yesterday_valuations = self._load_yesterday_valuations(as_of)

            for _, client in clients_df.iterrows():
                client_id = client["client_id"]
                c_trades_today = trades_df[trades_df["client_id"] == client_id].copy()
                if not c_trades_today.empty and not valuations_df.empty:
                    c_trades_today = c_trades_today.merge(valuations_df[["trade_id", "mtm_inr"]], on="trade_id", how="left")

                c_trades_yesterday = yesterday_trades[yesterday_trades["client_id"] == client_id].copy() \
                    if not yesterday_trades.empty else pd.DataFrame()
                if not c_trades_yesterday.empty and not yesterday_valuations.empty:
                    c_trades_yesterday = c_trades_yesterday.merge(yesterday_valuations[["trade_id", "mtm_inr"]], on="trade_id", how="left")
                
                if "mtm_inr" not in c_trades_today.columns and not c_trades_today.empty:
                    c_trades_today["mtm_inr"] = 0.0
                if "mtm_inr" not in c_trades_yesterday.columns and not c_trades_yesterday.empty:
                    c_trades_yesterday["mtm_inr"] = 0.0

                exp_row = exposures_df[exposures_df["client_id"] == client_id]
                exp_today = float(exp_row["exposure_inr"].iloc[0]) if not exp_row.empty else 0
                exp_yesterday = float(self._get_yesterday_exposure(client_id, as_of))

                vm_held_today = float(vm_df[vm_df["client_id"] == client_id]["vm_held_inr"].sum()) \
                    if not vm_df.empty else 0
                
                # Fetch yesterday's VM
                vm_held_yesterday = float(self._get_yesterday_vm(client_id, as_of))

                attr = compute_attribution(
                    exp_today, exp_yesterday, c_trades_today, c_trades_yesterday,
                    market, market_yesterday, vm_held_today, vm_held_yesterday, client_id
                )
                try:
                    verify_attribution_reconciliation(attr)
                except AssertionError as e:
                    logger.warning(str(e))
                attributions[client_id] = attr

            # Step 10: Alert engine
            alerts = self.alert_engine.evaluate(
                as_of, exposures_df, im_results, vm_df, market, market_yesterday, clients_df
            )

            # Step 11: Commentary
            commentaries = []
            for _, client in clients_df.iterrows():
                client_id = client["client_id"]
                attr = attributions.get(client_id, {})
                exp_row = exposures_df[exposures_df["client_id"] == client_id]
                exposure = float(exp_row["exposure_inr"].iloc[0]) if not exp_row.empty else 0

                im_row = im_results[
                    (im_results["client_id"] == client_id) & (im_results["product_type"] == "Portfolio")
                ]
                im_held = float(im_row["im_total_inr"].sum()) if not im_row.empty else 0

                vm_held = float(vm_df[vm_df["client_id"] == client_id]["vm_held_inr"].sum()) \
                    if not vm_df.empty else 0
                vm_call = float(exp_row["vm_call_inr"].iloc[0]) if not exp_row.empty and "vm_call_inr" in exp_row.columns else 0

                coverage = (im_held + vm_held) / exposure if exposure > 0 else 1.0
                client_alerts = [a for a in alerts if a.get("client_id") == client_id]

                comm = generate_daily_commentary(
                    as_of, client_id, attr, exposure, im_held, vm_held, vm_call,
                    coverage, client_alerts, self.db_path
                )
                commentaries.append(comm)

            # Step 12: Margin Calls & Settlement (Stretch Goal)
            generate_margin_calls(as_of, self.db_path)

            # Summary
            total_exposure = float(exposures_df["exposure_inr"].sum())
            total_im = float(im_results[im_results["product_type"] == "Portfolio"]["im_total_inr"].sum())
            n_critical_alerts = sum(1 for a in alerts if a.get("severity") == "critical")

            self._log_run_end(run_id, as_of, "success")

            return {
                "run_id": run_id,
                "date": str(as_of),
                "status": "success",
                "total_exposure_inr": total_exposure,
                "total_im_inr": total_im,
                "n_active_trades": len(trades_df),
                "n_alerts": len(alerts),
                "n_critical_alerts": n_critical_alerts,
                "n_commentaries": len(commentaries),
            }

        except Exception as e:
            logger.error(f"Daily run failed: {e}", exc_info=True)
            self._log_run_end(run_id, as_of, "failed", str(e))
            raise

    def _load_market(self, as_of: date) -> dict:
        """Load market data for a date as a flat dict."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT ticker, series, value FROM market_data WHERE date = :d"),
                {"d": str(as_of)},
            ).fetchall()

        market = {}
        for ticker, series, value in rows:
            market[f"{ticker}_{series}"] = float(value)

        # Defaults if data missing
        market.setdefault("^NSEI_close", 22000.0)
        market.setdefault("^INDIAVIX_close", 15.0)
        market.setdefault("USDINR=X_close", 84.0)
        market.setdefault("INDIRLTLT01STM_rate", 0.070)
        market.setdefault("SOFR_rate", 0.053)
        market.setdefault("BAMLC0A0CM_spread", 0.010)
        return market

    def _get_vm_ledger(self, as_of: date) -> pd.DataFrame:
        """Load VM ledger for a date."""
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM vm_ledger WHERE date = :d"),
                conn,
                params={"d": str(as_of)},
            )
        return df

    def _get_yesterday_exposure(self, client_id: str, as_of: date) -> float:
        """Get previous day's exposure for attribution."""
        yesterday = as_of - timedelta(days=1)
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT exposure_inr FROM exposures
                    WHERE client_id = :cid AND date = :d
                    LIMIT 1
                """),
                {"cid": client_id, "d": str(yesterday)},
            ).fetchone()
        return float(result[0]) if result else 0.0

    def _get_yesterday_vm(self, client_id: str, as_of: date) -> float:
        """Get previous day's vm_held for attribution."""
        yesterday = as_of - timedelta(days=1)
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT SUM(vm_held_inr) FROM vm_ledger
                    WHERE client_id = :cid AND date = :d
                """),
                {"cid": client_id, "d": str(yesterday)},
            ).scalar()
        return float(result) if result else 0.0

    def _load_yesterday_valuations(self, as_of: date) -> pd.DataFrame:
        yesterday = as_of - timedelta(days=1)
        with self.engine.connect() as conn:
            return pd.read_sql(
                text("SELECT * FROM valuations WHERE date = :d"),
                conn,
                params={"d": str(yesterday)},
            )

    def _write_exposures(
        self,
        as_of: date,
        exposures_df: pd.DataFrame,
        im_df: pd.DataFrame,
        vm_df: pd.DataFrame,
    ) -> None:
        """Write exposure results to DB."""
        with self.engine.begin() as conn:
            for _, row in exposures_df.iterrows():
                cid = row["client_id"]
                im_row = im_df[(im_df["client_id"] == cid) & (im_df["product_type"] == "Portfolio")]
                im_held = float(im_row["im_total_inr"].sum()) if not im_row.empty else 0.0
                
                vm_row = vm_df[vm_df["client_id"] == cid]
                vm_held = float(vm_row["vm_held_inr"].sum()) if not vm_row.empty else 0.0
                
                net_mtm = float(row.get("net_mtm_inr", 0))
                # For simplified SA-CCR, we assume a standard aggregated notional of 10x MTM (placeholder since trades_df isn't here)
                notional_proxy = abs(net_mtm) * 10
                
                ead = compute_sa_ccr_ead(
                    mtm=net_mtm,
                    vm_held=vm_held,
                    im_held=im_held,
                    notional=notional_proxy,
                    asset_class="equity"
                )

                credit_limit = 1e9  # Placeholder
                conn.execute(
                    text("""
                        INSERT INTO exposures
                            (date, client_id, netting_set_id, gross_mtm_inr, net_mtm_inr,
                             exposure_inr, collateralised_exposure_inr, ead_sa_ccr_inr, limit_util_pct,
                             ee_inr, epe_inr, pfe_95_inr, pfe_99_inr)
                        VALUES
                            (:date, :cid, :ns, :gross, :net, :exp, :coll_exp, :ead, :util,
                             :ee, :epe, :pfe95, :pfe99) ON CONFLICT DO NOTHING
                    """),
                    {
                        "date": str(as_of),
                        "cid": row["client_id"],
                        "ns": row.get("netting_set_id", f"NS_{row['client_id']}_01"),
                        "gross": float(row.get("gross_mtm_inr", 0)),
                        "net": float(row.get("net_mtm_inr", 0)),
                        "exp": float(row.get("exposure_inr", 0)),
                        "coll_exp": float(row.get("collateralised_exposure_inr", 0)),
                        "ead": float(ead),
                        "util": float(row.get("exposure_inr", 0)) / credit_limit * 100,
                        "ee": float(row.get("ee_inr", 0)),
                        "epe": float(row.get("epe_inr", 0)),
                        "pfe95": float(row.get("pfe_95_inr", 0)),
                        "pfe99": float(row.get("pfe_99_inr", 0)),
                    },
                )

    def _log_run_start(self, run_id: str, as_of: date) -> None:
        from datetime import datetime
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO pipeline_runs
                        (run_id, run_date, pipeline_step, status, start_time)
                    VALUES (:run_id, :run_date, 'run-daily', 'running', :start_time) ON CONFLICT DO NOTHING
                """),
                {"run_id": run_id, "run_date": str(as_of), "start_time": datetime.utcnow().isoformat()},
            )

    def _log_run_end(self, run_id: str, as_of: date, status: str, error: str = "") -> None:
        from datetime import datetime
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE pipeline_runs
                    SET status = :status, end_time = :end_time, error_message = :error
                    WHERE run_id = :run_id
                """),
                {
                    "run_id": run_id,
                    "status": status,
                    "end_time": datetime.utcnow().isoformat(),
                    "error": error,
                },
            )
