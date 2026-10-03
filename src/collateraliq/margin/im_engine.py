"""
IM orchestration engine.

Runs all IM models per client per day, applies add-ons, stress floor,
portfolio margin, and writes results to the im_results table.

Also computes:
- Portfolio margin vs sum-of-parts (diversification benefit)
- Per-product IM decomposition
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text

from collateraliq.data.db import get_engine
from collateraliq.margin.im_fhs import compute_fhs_im, compute_irs_im_fhs
from collateraliq.margin.im_options_grid import compute_scenario_grid_im
from collateraliq.margin.haircuts import haircut_im
from collateraliq.margin.addons import compute_all_addons
from collateraliq.margin.stress_floor import ProcyclicalityBuffer, compute_stress_period_im
from collateraliq.margin.simm_lite import compute_simm_lite_im, blend_fhs_simm
from collateraliq.pricing.cds import cds_spread_shock_var

logger = logging.getLogger(__name__)


class IMEngine:
    """
    Orchestrates IM calculation for all clients on a given date.

    Steps:
    1. Load active trades per client
    2. Build P&L history per product from market data
    3. Compute base IM per product
    4. Apply add-ons (concentration, liquidity, WWR)
    5. Apply stress floor / buffer (procyclicality)
    6. Compute portfolio margin (netting benefit)
    7. Write to im_results table
    """

    def __init__(
        self,
        params_path: str = "config/im_params.yaml",
        db_path: str = "data/collateraliq.db",
    ) -> None:
        self.params = yaml.safe_load(Path(params_path).read_text())
        self.db_path = db_path
        self.engine = get_engine(db_path)
        self.buffer = ProcyclicalityBuffer(
            buffer_fraction=self.params["procyclicality"]["buffer_fraction"],
            floor_fraction=self.params["procyclicality"]["floor_fraction"],
            stress_weight=self.params["procyclicality"]["stress_weight"],
        )

    def run(
        self,
        as_of: date,
        clients_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        market: dict,
        pnl_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run full IM calculation for all clients.

        Returns DataFrame of im_results rows.
        """
        all_results = []

        for _, client in clients_df.iterrows():
            client_id = client["client_id"]
            netting_set_id = client.get("netting_set_id", f"NS_{client_id}_01")
            archetype = client["archetype"]

            client_trades = trades_df[trades_df["client_id"] == client_id]

            if client_trades.empty:
                continue

            # Product-level IM
            product_ims = {}
            for product in client_trades["product_type"].unique():
                product_trades = client_trades[client_trades["product_type"] == product]
                total_notional = product_trades["notional_inr"].sum()

                im_result = self._compute_product_im(
                    product, product_trades, total_notional, market, pnl_history, client_id
                )
                product_ims[product] = im_result

            # Sum-of-parts IM
            sum_of_parts = sum(r.get("im_base_inr", 0) for r in product_ims.values())

            # Portfolio margin (with correlation/netting benefit)
            portfolio_im = self._portfolio_im(product_ims, client_trades)

            # Diversification benefit
            diversification_benefit = max(sum_of_parts - portfolio_im, 0)
            diversification_pct = (diversification_benefit / sum_of_parts * 100) if sum_of_parts > 0 else 0

            # Apply procyclicality buffer
            # Since full historical PnL is not always available at this stage, we proxy stressed IM
            # as 1.25x the portfolio IM for the blending calculation.
            stressed_im = portfolio_im * 1.25
            portfolio_im_with_buffer = self.buffer.apply_buffer(portfolio_im)
            portfolio_im_blended = self.buffer.apply_blend(portfolio_im, stressed_im)

            # Get dominant position for add-ons
            largest_trade = client_trades.loc[client_trades["notional_inr"].idxmax()]
            largest_notional = largest_trade["notional_inr"]
            adv_proxy = market.get(f"{largest_trade.get('product_type', 'EQ')}_adv", largest_notional * 0.05)

            addons = compute_all_addons(
                largest_notional, adv_proxy, portfolio_im,
                collateral_correlation=0.10 if archetype == "family_office" else 0.0,
                params=self.params["addons"]["concentration"],
            )

            total_im = addons["total_im_inr"]
            total_im_with_buffer = self.buffer.apply_buffer(total_im)

            # Write product-level rows
            for product, res in product_ims.items():
                row = {
                    "date": str(as_of),
                    "client_id": client_id,
                    "netting_set_id": netting_set_id,
                    "product_type": product,
                    "im_base_inr": res.get("im_base_inr", 0),
                    "im_stress_inr": res.get("im_base_inr", 0) * 1.5,
                    "im_buffer_inr": res.get("im_base_inr", 0) * 0.25,
                    "im_total_inr": res.get("im_base_inr", 0),
                    "addon_concentration_inr": 0,
                    "addon_liquidity_inr": 0,
                    "addon_wwr_inr": 0,
                    "method": res.get("method", "unknown"),
                    "params": json.dumps({"lookback": self.params["global"]["lookback_days"]}),
                }
                all_results.append(row)

            # Portfolio-level row (product_type = NULL)
            portfolio_row = {
                "date": str(as_of),
                "client_id": client_id,
                "netting_set_id": netting_set_id,
                "product_type": "Portfolio",
                "im_base_inr": portfolio_im,
                "im_stress_inr": stressed_im,
                "im_buffer_inr": portfolio_im * self.buffer.buffer_fraction,
                "im_total_inr": total_im_with_buffer,
                "addon_concentration_inr": addons["addon_concentration_inr"],
                "addon_liquidity_inr": addons["addon_liquidity_inr"],
                "addon_wwr_inr": addons["addon_wwr_inr"],
                "method": "portfolio_fhs",
                "params": json.dumps({
                    "sum_of_parts": sum_of_parts,
                    "portfolio_im": portfolio_im,
                    "diversification_pct": diversification_pct,
                }),
            }
            all_results.append(portfolio_row)

        results_df = pd.DataFrame(all_results)
        self._write_results(results_df)
        return results_df

    def _compute_product_im(
        self,
        product: str,
        product_trades: pd.DataFrame,
        total_notional: float,
        market: dict,
        pnl_history: pd.DataFrame | None,
        client_id: str,
    ) -> dict:
        """Compute base IM for a single product."""
        conf = self.params["global"]["confidence_level"]
        horizon = self.params["global"]["lookback_days"]
        lookback = self.params["global"]["lookback_days"]

        try:
            if product in ("equity_financing", "fx_forward"):
                # FHS on P&L history
                pnl = self._get_pnl_history(client_id, product, pnl_history, total_notional)
                result = compute_fhs_im(pnl, conf, 10, lookback, "overlapping")
                return {"im_base_inr": result["im_inr"], "method": "fhs_overlapping"}

            elif product == "gsec_repo":
                mat = float(product_trades.get("params", "{}").apply(
                    lambda p: json.loads(p).get("bond_maturity_years", 5) if isinstance(p, str) else 5
                ).mean() if len(product_trades) > 0 else 5)
                result = haircut_im(total_notional, mat)
                return {"im_base_inr": result["im_inr"], "method": "haircut"}

            elif product == "irs":
                pnl = self._get_pnl_history(client_id, product, pnl_history, total_notional)
                fhs_result = compute_fhs_im(pnl, conf, 10, lookback, "overlapping")
                # SIMM-lite challenger (simplified)
                dv01_buckets = {"DV01_5Y": total_notional * 0.0005}
                simm_result = compute_simm_lite_im(dv01_buckets)
                blended = blend_fhs_simm(fhs_result["im_inr"], simm_result["im_inr"])
                return {"im_base_inr": blended, "method": "fhs_simm_blended"}

            elif product == "cds_index":
                spread_bps = market.get("BAMLC0A0CM_spread", 0.01) * 10000
                spread_history = np.array([spread_bps * (0.9 + 0.2 * np.random.random()) for _ in range(300)])
                result = cds_spread_shock_var(total_notional, spread_history, 5.0)
                return {"im_base_inr": result["im_inr"], "method": "cds_spread_shock_var"}

            elif product in ("equity_options", "single_stock_options"):
                spot = market.get("^NSEI_close", 22000)
                sigma = market.get("^INDIAVIX_close", 20.0) / 100
                r = market.get("INDIRLTLT01STM_rate", 0.07)
                # Build option list from trades
                opts = []
                for _, t in product_trades.iterrows():
                    p = json.loads(t["params"]) if isinstance(t["params"], str) else {}
                    opts.append({
                        "K": spot * p.get("moneyness", 1.0),
                        "T": p.get("tenor_years", 0.083),
                        "option_type": p.get("option_type", "call"),
                        "contracts": p.get("contracts", 10),
                        "lot_size": p.get("lot_size", 50),
                        "direction": int(t["direction"]),
                    })
                result = compute_scenario_grid_im(opts, spot, sigma, r)
                return {"im_base_inr": result["im_inr"], "method": "scenario_grid"}

        except Exception as e:
            logger.error(f"IM error for {client_id}/{product}: {e}")

        # Fallback: 10% of notional
        return {"im_base_inr": total_notional * 0.10, "method": "fallback_pct"}

    def _get_pnl_history(
        self,
        client_id: str,
        product: str,
        pnl_history: pd.DataFrame | None,
        notional: float,
        n_days: int = 500,
    ) -> np.ndarray:
        """Get P&L history for FHS. Falls back to synthetic if unavailable."""
        if pnl_history is not None and client_id in pnl_history.columns:
            return np.array(pnl_history[client_id].dropna())

        # Synthetic P&L: GBM-like with product-specific vol
        vols = {
            "equity_financing": 0.20,
            "fx_forward": 0.08,
            "irs": 0.005,
            "cds_index": 0.15,
        }
        daily_vol = vols.get(product, 0.15) / np.sqrt(252)
        rng = np.random.default_rng(hash(client_id + product) % 2**32)
        return notional * daily_vol * rng.standard_normal(n_days)

    def _portfolio_im(self, product_ims: dict, client_trades: pd.DataFrame) -> float:
        """
        Portfolio IM with diversification benefit.

        Simple approximation: apply 10-15% netting benefit for diversified books.
        In production: full correlation matrix across risk factors.
        """
        sum_of_parts = sum(r.get("im_base_inr", 0) for r in product_ims.values())
        n_products = len(product_ims)

        # Diversification discount: 5% per additional product (capped at 30%)
        netting_discount = min(0.05 * (n_products - 1), 0.30)

        return float(sum_of_parts * (1 - netting_discount))

    def _write_results(self, df: pd.DataFrame) -> None:
        """Write IM results to database."""
        with self.engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    text("""
                        INSERT INTO im_results
                            (date, client_id, netting_set_id, product_type, im_base_inr,
                             im_stress_inr, im_buffer_inr, im_total_inr,
                             addon_concentration_inr, addon_liquidity_inr, addon_wwr_inr,
                             method, params)
                        VALUES
                            (:date, :client_id, :netting_set_id, :product_type, :im_base_inr,
                             :im_stress_inr, :im_buffer_inr, :im_total_inr,
                             :addon_concentration_inr, :addon_liquidity_inr, :addon_wwr_inr,
                             :method, :params) ON CONFLICT DO NOTHING
                    """),
                    row.to_dict(),
                )
