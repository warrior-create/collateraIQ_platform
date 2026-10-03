"""
CollateralIQ: Client Exposure, Collateral and Initial Margin Platform

A full Credit Exposure Management (CEM) desk simulator built for the CEM practitioner role.

Modules
-------
data        : market data ingestion, quality checks, audit trail, lineage
clients     : synthetic client generator, CSA terms, trade lifecycle
pricing     : equity, options, FX forward, bonds, IRS, CDS
exposure    : netting, collateralised exposure, EE/PFE, CVA, SA-CCR
margin      : IM calibration (FHS, scenario grid, haircuts, add-ons, SIMM-lite)
adequacy    : IM backtest, default-loss simulation, procyclicality study
monitoring  : daily run, attribution, alerts, margin calls, early warning
commentary  : deterministic templates, number verifier, optional LLM polish
change      : what-if engine, impact report, phase-in analysis
collateral  : cheapest-to-deliver optimiser, liquidation horizon
api         : FastAPI pre-trade quote service
reporting   : Excel/VBA, PDF, Streamlit export

All client/trade data is synthetic. Market data sourced from yfinance and FRED.
"""

__version__ = "0.1.0"
__author__ = "Credit Risk Methodology Group"
