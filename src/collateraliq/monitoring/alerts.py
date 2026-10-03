"""
Alert engine for the daily monitoring run.

Evaluates alert rules from alerts.yaml against the daily computed values.
Writes triggered alerts to the alerts table with severity and details.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text

from collateraliq.data.db import get_engine

logger = logging.getLogger(__name__)


class AlertEngine:
    """Evaluates alert rules and writes to DB."""

    def __init__(
        self,
        config_path: str = "config/alerts.yaml",
        db_path: str = "data/collateraliq.db",
    ) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text())["alerts"]
        self.db_path = db_path
        self.engine = get_engine(db_path)

    def evaluate(
        self,
        as_of: date,
        exposures_df: pd.DataFrame,
        im_df: pd.DataFrame,
        vm_df: pd.DataFrame,
        market: dict,
        market_yesterday: dict | None = None,
        clients_df: pd.DataFrame | None = None,
    ) -> list[dict]:
        """
        Evaluate all alert rules and return list of triggered alerts.
        """
        triggered = []

        for _, client in (clients_df.iterrows() if clients_df is not None else []):
            client_id = client["client_id"]
            credit_limit = client.get("credit_limit_inr", 1e10)

            exp_row = exposures_df[exposures_df["client_id"] == client_id]
            if exp_row.empty:
                continue

            exposure = float(exp_row["exposure_inr"].iloc[0])
            coll_exp = float(exp_row["collateralised_exposure_inr"].iloc[0])

            im_row = im_df[(im_df["client_id"] == client_id) & im_df["product_type"].isna()]
            im_held = float(im_row["im_total_inr"].sum()) if not im_row.empty else 0

            vm_row = vm_df[vm_df["client_id"] == client_id] if not vm_df.empty else pd.DataFrame()
            vm_held = float(vm_row["vm_held_inr"].sum()) if not vm_row.empty else 0

            # 1. Credit limit breach
            if self.config["limit_breach"]["enabled"]:
                limit_util = exposure / credit_limit if credit_limit > 0 else 0
                if limit_util >= self.config["limit_breach"]["threshold_pct"]:
                    triggered.append(self._make_alert(
                        as_of, client_id, "limit_breach", "critical",
                        f"Client {client_id} exposure {exposure/1e6:.1f}M INR at {limit_util*100:.0f}% of limit",
                        {"exposure_inr": exposure, "limit_inr": credit_limit, "utilisation_pct": limit_util * 100},
                    ))

            # 2. Margin deficit
            if self.config["margin_deficit"]["enabled"]:
                min_cov = self.config["margin_deficit"]["min_coverage_ratio"]
                if exposure > 0 and (vm_held + im_held) < exposure * min_cov:
                    triggered.append(self._make_alert(
                        as_of, client_id, "margin_deficit", "critical",
                        f"Client {client_id} margin deficit: held {(vm_held+im_held)/1e6:.1f}M vs exposure {exposure/1e6:.1f}M",
                        {"vm_held": vm_held, "im_held": im_held, "exposure": exposure},
                    ))

            # 3. Coverage ratio drop
            if self.config["coverage_ratio_drop"]["enabled"]:
                coverage_ratio = (vm_held + im_held) / exposure if exposure > 0 else 1.0
                # Would need yesterday's ratio; proxy: flag if < 90%
                if coverage_ratio < (1 - self.config["coverage_ratio_drop"]["drop_threshold_pct"]):
                    triggered.append(self._make_alert(
                        as_of, client_id, "coverage_ratio_drop", "high",
                        f"Client {client_id} coverage ratio {coverage_ratio:.0%}",
                        {"coverage_ratio": coverage_ratio},
                    ))

        # 4. Large price moves (firm-wide)
        if self.config["large_price_move"]["enabled"] and market_yesterday:
            nifty_move = abs(
                market.get("^NSEI_close", 22000) / market_yesterday.get("^NSEI_close", 22000) - 1
            )
            vix = market.get("^INDIAVIX_close", 20.0)
            daily_vol = vix / 100 / np.sqrt(252)
            zscore = nifty_move / daily_vol if daily_vol > 0 else 0

            if zscore > self.config["large_price_move"]["zscore_threshold"]:
                triggered.append(self._make_alert(
                    as_of, None, "large_price_move", "medium",
                    f"Nifty moved {nifty_move*100:.1f}% ({zscore:.1f}σ) — review all exposures",
                    {"nifty_move_pct": nifty_move * 100, "zscore": zscore},
                ))

        # Write to DB
        self._write_alerts(triggered)
        logger.info(f"Alert engine: {len(triggered)} alerts triggered on {as_of}")
        return triggered

    def _make_alert(
        self,
        as_of: date,
        client_id: str | None,
        alert_type: str,
        severity: str,
        message: str,
        details: dict,
    ) -> dict:
        return {
            "alert_id": str(uuid.uuid4())[:8],
            "date": str(as_of),
            "client_id": client_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "details": json.dumps(details),
            "is_resolved": 0,
        }

    def _write_alerts(self, alerts: list[dict]) -> None:
        with self.engine.begin() as conn:
            for alert in alerts:
                conn.execute(
                    text("""
                        INSERT INTO alerts
                            (alert_id, date, client_id, alert_type, severity, message, details, is_resolved)
                        VALUES
                            (:alert_id, :date, :client_id, :alert_type, :severity, :message, :details, :is_resolved) ON CONFLICT DO NOTHING
                    """),
                    alert,
                )
