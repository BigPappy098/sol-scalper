-- Initial schema for SOL Scalper

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV candle data at multiple resolutions
CREATE TABLE IF NOT EXISTS candles (
    ts          TIMESTAMPTZ NOT NULL,
    timeframe   TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    trade_count INTEGER DEFAULT 0,
    vwap        DOUBLE PRECISION DEFAULT 0,
    PRIMARY KEY (ts, timeframe)
);

SELECT create_hypertable('candles', 'ts', if_not_exists => TRUE);

-- Create index for fast timeframe queries
CREATE INDEX IF NOT EXISTS idx_candles_timeframe_ts ON candles (timeframe, ts DESC);

-- Compress old candle data (older than 7 days)
ALTER TABLE candles SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'timeframe'
);

SELECT add_compression_policy('candles', INTERVAL '7 days', if_not_exists => TRUE);

-- Our executed trades
CREATE TABLE IF NOT EXISTS trades (
    id              SERIAL PRIMARY KEY,
    ts_entry        TIMESTAMPTZ NOT NULL,
    ts_exit         TIMESTAMPTZ,
    side            TEXT,
    entry_price     DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,
    quantity        DOUBLE PRECISION,
    pnl_usd        DOUBLE PRECISION,
    pnl_pct         DOUBLE PRECISION,
    strategy_name   TEXT,
    signal_confidence DOUBLE PRECISION,
    exit_reason     TEXT,
    fees_usd        DOUBLE PRECISION DEFAULT 0,
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades (ts_entry DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy_name, ts_entry DESC);

-- Strategy performance snapshots
CREATE TABLE IF NOT EXISTS strategy_metrics (
    ts              TIMESTAMPTZ NOT NULL,
    strategy_name   TEXT NOT NULL,
    window_hours    INTEGER NOT NULL,
    win_rate        DOUBLE PRECISION,
    avg_r           DOUBLE PRECISION,
    sharpe          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    trade_count     INTEGER,
    weight          DOUBLE PRECISION,
    PRIMARY KEY (ts, strategy_name, window_hours)
);

SELECT create_hypertable('strategy_metrics', 'ts', if_not_exists => TRUE);

-- Model registry
CREATE TABLE IF NOT EXISTS models (
    id              SERIAL PRIMARY KEY,
    ts_trained      TIMESTAMPTZ NOT NULL,
    model_name      TEXT,
    model_version   INTEGER,
    artifact_path   TEXT,
    val_metrics     JSONB DEFAULT '{}'::jsonb,
    is_active       BOOLEAN DEFAULT FALSE,
    config          JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_models_active ON models (model_name, is_active) WHERE is_active = TRUE;

-- Equity snapshots for tracking account value over time
CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts              TIMESTAMPTZ NOT NULL PRIMARY KEY,
    equity_usd      DOUBLE PRECISION,
    open_positions  INTEGER DEFAULT 0,
    daily_pnl       DOUBLE PRECISION DEFAULT 0
);

SELECT create_hypertable('equity_snapshots', 'ts', if_not_exists => TRUE);
