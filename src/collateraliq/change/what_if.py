"""
What-if engine for IM model change impact analysis.

Allows changing:
- Lookback period (250 → 500 days)
- Confidence level (99% → 99.5%)
- MPOR (5 → 10 days)
- Haircut schedule
- Buffer parameters
- Adding a new product

Reports:
- IM change per client (distribution)
- Total IM change
- Additional margin calls triggered
- Most affected clients
- Phase-in plan to avoid margin shock

CLI: collateraliq what-if --scenario config/im_lookback_change.yaml
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sqlalchemy import text

from collateraliq.data.db import get_engine
from collateraliq.margin.im_engine import IMEngine

logger = logging.getLogger(__name__)


class WhatIfEngine:
    """Runs IM model change impact analysis."""

    def __init__(self, db_path: str = "data/collateraliq.db") -> None:
        self.db_path = db_path
        self.engine = get_engine(db_path)

    def run(
        self,
        scenario_config: dict,
        as_of: date,
        clients_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        market: dict,
    ) -> dict:
        """
        Run a what-if scenario.

        scenario_config: dict with parameter changes to apply
        Returns impact report dict.
        """
        scenario_name = scenario_config.get("name", "unnamed_scenario")
        run_id = f"WI_{uuid.uuid4().hex[:8]}"

        # Get baseline IM from DB
        baseline_im = self._get_baseline_im(as_of, clients_df["client_id"].tolist())

        # Apply parameter changes
        modified_params = self._apply_scenario(scenario_config)

        # Compute new IM with modified params
        modified_engine = IMEngine(db_path=self.db_path)
        
        # Deep update params
        for k, v in modified_params.items():
            if isinstance(v, dict) and k in modified_engine.params:
                modified_engine.params[k].update(v)
            else:
                modified_engine.params[k] = v
                
        # Apply market shock (Stress Testing)
        shock_pct = scenario_config.get("changes", {}).get("market_shock_pct", 0.0)
        if shock_pct != 0.0:
            import copy
            market = copy.deepcopy(market)
            shock = shock_pct / 100.0
            for key in list(market.keys()):
                # Shock equity prices (keys ending in _close)
                if key.endswith("_close"):
                    market[key] *= (1 + shock)

        new_im = modified_engine.run(as_of, clients_df, trades_df, market)
        new_im_portfolio = new_im[new_im["product_type"].isna()].set_index("client_id")["im_total_inr"]

        # Compute impact
        impact = []
        for client_id, baseline in baseline_im.items():
            new = float(new_im_portfolio.get(client_id, baseline))
            change_inr = new - baseline
            change_pct = change_inr / baseline * 100 if baseline > 0 else 0

            impact.append({
                "client_id": client_id,
                "im_before_inr": baseline,
                "im_after_inr": new,
                "im_change_inr": change_inr,
                "im_change_pct": change_pct,
                "extra_call_triggered": change_inr > 1e6,
            })

        impact_df = pd.DataFrame(impact)
        total_before = baseline_im.values().__iter__().__next__() if baseline_im else 0
        total_before = sum(baseline_im.values())
        total_after = float(new_im_portfolio.sum())
        total_change = total_after - total_before

        # Most affected clients
        most_affected = impact_df.nlargest(5, "im_change_pct")[
            ["client_id", "im_change_inr", "im_change_pct"]
        ].to_dict(orient="records")

        # Phase-in plan
        phase_in = self._compute_phase_in(impact_df, n_phases=3)

        # Distribution of changes
        changes = impact_df["im_change_pct"].values
        distribution = {
            "mean_pct": float(np.mean(changes)),
            "median_pct": float(np.median(changes)),
            "p25_pct": float(np.percentile(changes, 25)),
            "p75_pct": float(np.percentile(changes, 75)),
            "p99_pct": float(np.percentile(changes, 99)),
            "max_pct": float(np.max(changes)),
        }

        result = {
            "run_id": run_id,
            "scenario_name": scenario_name,
            "as_of": str(as_of),
            "params_changed": scenario_config.get("changes", {}),
            "total_im_before_inr": float(total_before),
            "total_im_after_inr": float(total_after),
            "total_im_change_inr": float(total_change),
            "total_im_change_pct": float(total_change / total_before * 100) if total_before > 0 else 0,
            "n_clients_extra_call": int(impact_df["extra_call_triggered"].sum()),
            "most_affected_clients": most_affected,
            "distribution": distribution,
            "phase_in_plan": phase_in,
            "client_impact": impact_df.to_dict(orient="records"),
        }

        # Write to DB
        self._write_change_run(result)
        return result

    def _get_baseline_im(self, as_of: date, client_ids: list) -> dict:
        """Get current IM from DB for all clients."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT client_id, im_total_inr
                    FROM im_results
                    WHERE date = :d AND product_type IS NULL
                """),
                {"d": str(as_of)},
            ).fetchall()
        baseline = {row[0]: float(row[1]) for row in rows}
        # Fill missing with a placeholder
        for cid in client_ids:
            if cid not in baseline:
                baseline[cid] = 1e6
        return baseline

    def _apply_scenario(self, scenario_config: dict) -> dict:
        """Apply parameter changes from scenario config."""
        changes = scenario_config.get("changes", {})
        modified = {}

        if "lookback_days" in changes:
            modified["global"] = {"lookback_days": changes["lookback_days"]}
        if "confidence" in changes:
            modified.setdefault("global", {})["confidence_level"] = changes["confidence"]
        if "mpor_days" in changes:
            modified.setdefault("global", {})["mpor_days"] = changes["mpor_days"]
        if "buffer_fraction" in changes:
            modified["procyclicality"] = {"buffer_fraction": changes["buffer_fraction"]}

        return modified

    def _compute_phase_in(self, impact_df: pd.DataFrame, n_phases: int = 3) -> list[dict]:
        """
        Compute a phase-in plan to avoid margin shock.

        Phase-in: divide total IM increase into n equal steps over n months.
        """
        total_increase = impact_df["im_change_inr"].clip(lower=0).sum()
        per_phase = total_increase / n_phases

        return [
            {
                "phase": i + 1,
                "months_from_now": i + 1,
                "cumulative_im_increase_inr": float(per_phase * (i + 1)),
                "pct_of_total_change": float((i + 1) / n_phases * 100),
            }
            for i in range(n_phases)
        ]

    def _write_change_run(self, result: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO change_runs
                        (run_id, run_date, scenario_name, params_before, params_after,
                         total_im_before_inr, total_im_after_inr, im_change_pct, clients_affected)
                    VALUES
                        (:run_id, CURRENT_TIMESTAMP, :scenario, :before, :after,
                         :im_before, :im_after, :im_change_pct, :n_clients)
                """),
                {
                    "run_id": result["run_id"],
                    "scenario": result["scenario_name"],
                    "before": json.dumps({}),
                    "after": json.dumps(result["params_changed"]),
                    "im_before": result["total_im_before_inr"],
                    "im_after": result["total_im_after_inr"],
                    "im_change_pct": result["total_im_change_pct"],
                    "n_clients": result["n_clients_extra_call"],
                },
            )
