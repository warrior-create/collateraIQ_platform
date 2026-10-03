# CollateralIQ: Enterprise CEM Platform

CollateralIQ is a comprehensive, institutional-grade Credit Exposure Management (CEM) platform. It provides end-to-end tooling to manage client portfolios, calculate collateralised exposure, calibrate Initial Margin (IM) using advanced risk models, run regulatory backtests, provide daily commentary, and allow complex what-if scenario testing. 

Built to simulate a mini CEM desk, it manages 25 client portfolios with real market data, modeling everything from trade inception to margin calls and dispute management.

## 🚀 Key Points (What It Solves)
- **Margin Procyclicality**: Solves the "margin spike" problem during crises by implementing EMIR-style anti-procyclicality buffers and stress floors.
- **Cheapest-to-Deliver (CTD) Optimization**: Uses linear programming to minimize funding costs by allocating the optimal mix of eligible collateral (Cash, G-Secs, Equities) to meet margin requirements.
- **Risk & Limit Monitoring**: Provides early-warning ML models to predict client limit breaches before they happen, incorporating Wrong-Way Risk (WWR) and liquidity add-ons.
- **Pre-Rollout What-If Analysis**: Allows risk managers to backtest new margin models (e.g., shifting from a 250-day to a 500-day lookback) and generate client-by-client phase-in plans to avoid margin shock.

---

## 🏛 Architecture

CollateralIQ operates on a daily pipeline architecture with an interactive front-end:

1. **Ingestion & Generation**: Pulls market data and generates synthetic trades for 5 client archetypes (Hedge Funds, Pension Funds, Corporates, etc.).
2. **Pricing Engine**: Vectorized pricer supporting Equity Options, FX Forwards, IRS, Bonds, and CDS.
3. **Exposure & Margin Engine**: Computes MTM, CVA, EE, PFE, and SA-CCR. Calculates IM using Historical FHS (Filtered Historical Simulation), SIMM-lite, and scenario grids.
4. **Monitoring & Analytics**: Day-on-day economic attribution, margin call generation, and automated commentary generation.
5. **Presentation Layer**: Streamlit-based interactive BI dashboard and FastAPI endpoints for live pre-trade quotes.

---

## 🛠 Tech Stack
- **Backend & Compute**: Python 3.12+, Pandas (Vectorized operations), NumPy, SciPy (Linear Programming for Optimization).
- **Web App & Dashboard**: Streamlit, Plotly (Interactive visualizations).
- **API**: FastAPI, Uvicorn, Pydantic.
- **Persistence**: SQLite (Local development/demo), SQLAlchemy ORM.
- **Configuration**: YAML-based modular configuration for archetypes, CSA parameters, and stress scenarios.

---

## 📦 Core Modules
- `src/collateraliq/clients/`: Client generation, trade lifecycle simulation, and CSA terms.
- `src/collateraliq/pricing/`: From-scratch pricing models with vectorized Greeks computation.
- `src/collateraliq/exposure/`: Netting, Exposure at Default (SA-CCR), Monte Carlo PFE.
- `src/collateraliq/margin/`: Core IM calculations (FHS, Procyclicality, Add-ons, Haircuts).
- `src/collateraliq/monitoring/`: Daily Run orchestrator, Economic Attribution, ML Early Warning scoring.
- `src/collateraliq/change/`: What-If scenario engine and phase-in planning.
- `src/collateraliq/collateral/`: LP-based CTD optimization and substitution workflows.
- `app/streamlit_app.py`: Full multi-page Business Intelligence application.

---

## 💻 How to Run

### 1. Environment Setup
Ensure you have Python 3.12+ installed.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize Database & Seed Data
Initialize the database schemas and generate synthetic trades and market data.
```bash
python src/collateraliq/main.py init
```

### 3. Run the Daily Pipeline
Execute the end-to-end daily run to compute exposures, margins, and generate margin calls.
```bash
python src/collateraliq/main.py run-daily --date 2026-10-02
```

### 4. Run What-If Scenario (CLI)
Test a model change before rolling it out.
```bash
python src/collateraliq/main.py what-if --date 2026-10-02 --scenario config/whatif_scenario1.yaml
```

### 5. Launch the Dashboard
Fire up the Streamlit UI to interact with the platform.
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Stretch Goals Implemented
The platform successfully incorporates all 10 target stretch additions:
1. **Procyclicality Study**: Evaluates margin stability under COVID-19/2022 market shocks.
2. **Liquidation Horizon**: Concentration add-ons based on Average Daily Volume (ADV).
3. **Wrong-Way Risk (WWR)**: Correlation penalties for correlated portfolios.
4. **What-If Engine**: Comprehensive UI for scenario testing.
5. **Economic Attribution**: Day-on-day exact reconciliation of margin movements.
6. **Pre-Trade API**: Async quoting for new trades.
7. **Data Lineage**: Auto-generated Mermaid diagrams mapping data flow provenance.
8. **Collateral Substitution Workflow**: Interactive eligibility checks for asset swaps.
9. **Regulatory Capital**: Visual SA-CCR vs CEM comparisons.
10. **Performance Benchmark**: Benchmarking for iterative vs fully vectorized operations.
