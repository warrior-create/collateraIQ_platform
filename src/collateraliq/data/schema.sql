-- CollateralIQ Database Schema
-- SQLite (dev) / PostgreSQL (prod) compatible DDL
-- All tables include audit columns: created_at, updated_at



-- ============================================================
-- REFERENCE / MASTER DATA
-- ============================================================

CREATE TABLE IF NOT EXISTS clients (
    client_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    archetype       TEXT NOT NULL,   -- leveraged_hf | long_only | corporate | family_office | macro
    credit_limit_inr REAL NOT NULL,
    rating          TEXT,            -- internal rating AAA-D
    onboard_date    DATE NOT NULL,
    is_active       INTEGER DEFAULT 1,
    seed_offset     INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS csa_terms (
    csa_id          TEXT PRIMARY KEY,
    client_id       TEXT NOT NULL REFERENCES clients(client_id),
    netting_set_id  TEXT NOT NULL,
    threshold_inr   REAL NOT NULL DEFAULT 0,
    mta_inr         REAL NOT NULL DEFAULT 0,
    independent_amount_inr REAL NOT NULL DEFAULT 0,
    rounding_inr    REAL NOT NULL DEFAULT 100000,
    call_frequency  TEXT NOT NULL DEFAULT 'daily',   -- daily | weekly
    mpor_days       INTEGER NOT NULL DEFAULT 5,
    eligible_collateral JSON,        -- list of {asset, haircut}
    governing_law   TEXT DEFAULT 'ISDA_2002',
    effective_date  DATE NOT NULL,
    termination_date DATE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS netting_sets (
    netting_set_id  TEXT PRIMARY KEY,
    client_id       TEXT NOT NULL REFERENCES clients(client_id),
    csa_id          TEXT REFERENCES csa_terms(csa_id),
    description     TEXT,
    currency        TEXT DEFAULT 'INR',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- TRADE BOOK
-- ============================================================

CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    client_id       TEXT NOT NULL REFERENCES clients(client_id),
    netting_set_id  TEXT NOT NULL REFERENCES netting_sets(netting_set_id),
    product_type    TEXT NOT NULL,   -- equity_financing | gsec_repo | irs | cds_index | equity_options | fx_forward | single_stock_options
    direction       INTEGER NOT NULL DEFAULT 1,  -- 1=long/receive, -1=short/pay
    notional_inr    REAL NOT NULL,
    currency        TEXT DEFAULT 'INR',
    trade_date      DATE NOT NULL,
    maturity_date   DATE NOT NULL,
    status          TEXT DEFAULT 'active',   -- active | matured | terminated | amended
    params          JSON,            -- product-specific parameters (strike, rate, etc.)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trade_events (
    event_id        TEXT PRIMARY KEY,
    trade_id        TEXT NOT NULL REFERENCES trades(trade_id),
    event_type      TEXT NOT NULL,   -- new | amendment | maturity | roll | termination
    event_date      DATE NOT NULL,
    old_params      JSON,
    new_params      JSON,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- MARKET DATA
-- ============================================================

CREATE TABLE IF NOT EXISTS market_data (
    data_id         SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    ticker          TEXT NOT NULL,
    series          TEXT NOT NULL,   -- close | open | volume | yield | spread | vix | fx
    value           REAL NOT NULL,
    source          TEXT NOT NULL,   -- yfinance | FRED | FBIL | synthetic
    is_interpolated INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, ticker, series)
);

CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date);
CREATE INDEX IF NOT EXISTS idx_market_data_ticker ON market_data(ticker);

-- ============================================================
-- VALUATIONS & EXPOSURE
-- ============================================================

CREATE TABLE IF NOT EXISTS valuations (
    val_id          SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    trade_id        TEXT NOT NULL REFERENCES trades(trade_id),
    mtm_inr         REAL NOT NULL,
    delta_inr       REAL,
    gamma_inr       REAL,
    vega_inr        REAL,
    dv01_inr        REAL,
    model_version   TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, trade_id)
);

CREATE TABLE IF NOT EXISTS exposures (
    exp_id          SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    client_id       TEXT NOT NULL REFERENCES clients(client_id),
    netting_set_id  TEXT NOT NULL,
    gross_mtm_inr   REAL,
    net_mtm_inr     REAL,            -- after netting
    exposure_inr    REAL,            -- max(net_mtm, 0)
    collateralised_exposure_inr REAL,
    ee_inr          REAL,            -- Expected Exposure
    epe_inr         REAL,            -- Expected Positive Exposure
    pfe_95_inr      REAL,
    pfe_99_inr      REAL,
    cva_inr         REAL,
    ead_sa_ccr_inr  REAL,
    limit_util_pct  REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, client_id, netting_set_id)
);

-- ============================================================
-- INITIAL MARGIN
-- ============================================================

CREATE TABLE IF NOT EXISTS im_results (
    im_id           SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    client_id       TEXT NOT NULL,
    netting_set_id  TEXT NOT NULL,
    product_type    TEXT NOT NULL DEFAULT 'Portfolio', -- 'Portfolio' for portfolio-level
    im_base_inr     REAL NOT NULL,
    im_stress_inr   REAL,
    im_buffer_inr   REAL,
    im_total_inr    REAL NOT NULL,
    addon_concentration_inr REAL DEFAULT 0,
    addon_liquidity_inr     REAL DEFAULT 0,
    addon_wwr_inr           REAL DEFAULT 0,
    method          TEXT,            -- fhs | scenario_grid | haircut | simm_lite
    params          JSON,            -- calibration parameters snapshot
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, client_id, netting_set_id, product_type)
);

-- ============================================================
-- COLLATERAL & MARGIN CALLS
-- ============================================================

CREATE TABLE IF NOT EXISTS vm_ledger (
    ledger_id       SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    client_id       TEXT NOT NULL,
    netting_set_id  TEXT NOT NULL,
    vm_required_inr REAL,
    vm_held_inr     REAL,
    vm_call_inr     REAL,           -- call amount (positive = call from client)
    vm_return_inr   REAL,           -- return to client (if over-collateralised)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS collateral_inventory (
    inv_id          SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    client_id       TEXT NOT NULL,
    asset_type      TEXT NOT NULL,
    notional_inr    REAL NOT NULL,
    haircut         REAL NOT NULL DEFAULT 0,
    eligible_value_inr REAL,        -- notional × (1 - haircut)
    is_segregated   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS margin_calls (
    call_id         TEXT PRIMARY KEY,
    date            DATE NOT NULL,
    client_id       TEXT NOT NULL,
    call_type       TEXT NOT NULL,  -- vm | im | im_top_up
    amount_inr      REAL NOT NULL,
    status          TEXT DEFAULT 'issued',  -- issued | disputed | settled | escalated | cured
    issued_at       TIMESTAMP,
    dispute_at      TIMESTAMP,
    settlement_at   TIMESTAMP,
    escalation_at   TIMESTAMP,
    cure_deadline   TIMESTAMP,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS disputes (
    dispute_id      TEXT PRIMARY KEY,
    call_id         TEXT REFERENCES margin_calls(call_id),
    client_id       TEXT NOT NULL,
    dispute_date    DATE NOT NULL,
    disputed_amount_inr REAL,
    bank_amount_inr REAL,
    resolution_date DATE,
    resolution_type TEXT,           -- agreed | escalated | withdrawn
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- MONITORING & COMMENTARY
-- ============================================================

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        TEXT PRIMARY KEY,
    date            DATE NOT NULL,
    client_id       TEXT,
    alert_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,  -- critical | high | medium | low
    message         TEXT NOT NULL,
    details         JSON,
    is_resolved     INTEGER DEFAULT 0,
    resolved_at     TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS commentary (
    comm_id         TEXT PRIMARY KEY,
    date            DATE NOT NULL,
    client_id       TEXT,           -- NULL for firm-wide commentary
    commentary_type TEXT,           -- daily | alert | exception
    template_text   TEXT,
    polished_text   TEXT,           -- LLM-polished version
    verified        INTEGER DEFAULT 0,  -- 1 = passed number verifier
    numbers_extracted JSON,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- BACKTESTING & ANALYSIS
-- ============================================================

CREATE TABLE IF NOT EXISTS im_backtest (
    bt_id           SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    client_id       TEXT NOT NULL,
    product_type    TEXT,
    mpor_loss_inr   REAL,
    im_held_inr     REAL,
    is_breach       INTEGER,        -- 1 = IM did not cover loss
    shortfall_inr   REAL DEFAULT 0,
    method          TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS default_loss (
    dl_id           SERIAL PRIMARY KEY,
    sim_date        DATE NOT NULL,
    client_id       TEXT NOT NULL,
    close_out_loss_inr REAL,
    vm_held_inr     REAL,
    im_held_inr     REAL,
    liquidation_cost_inr REAL,
    net_shortfall_inr REAL,
    is_covered      INTEGER,        -- 1 = IM + VM covered the loss
    scenario_label  TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- CHANGE IMPACT / MODEL GOVERNANCE
-- ============================================================

CREATE TABLE IF NOT EXISTS change_runs (
    run_id          TEXT PRIMARY KEY,
    run_date        TIMESTAMP NOT NULL,
    scenario_name   TEXT NOT NULL,
    params_before   JSON,
    params_after    JSON,
    total_im_before_inr REAL,
    total_im_after_inr  REAL,
    im_change_pct   REAL,
    clients_affected INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_inventory (
    model_id        TEXT PRIMARY KEY,
    model_name      TEXT NOT NULL,
    product_type    TEXT,
    version         TEXT,
    parameters      JSON,
    effective_date  DATE,
    retired_date    DATE,
    approved_by     TEXT,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- GOVERNANCE & AUDIT
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id        SERIAL PRIMARY KEY,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_or_process TEXT NOT NULL,
    action          TEXT NOT NULL,
    table_name      TEXT,
    record_id       TEXT,
    old_value       JSON,
    new_value       JSON,
    data_hash       TEXT,
    run_id          TEXT,
    ip_address      TEXT
);

CREATE TABLE IF NOT EXISTS dq_log (
    dq_id           SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    check_name      TEXT NOT NULL,
    table_name      TEXT,
    records_checked INTEGER,
    records_failed  INTEGER,
    failure_details JSON,
    severity        TEXT,
    is_blocking     INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily pipeline run log
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    run_date        DATE NOT NULL,
    pipeline_step   TEXT NOT NULL,
    status          TEXT,           -- success | failed | partial
    start_time      TIMESTAMP,
    end_time        TIMESTAMP,
    records_processed INTEGER,
    error_message   TEXT,
    config_hash     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
