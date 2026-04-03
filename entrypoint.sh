#!/bin/bash
set -e

DATA_DIR="/data"
# Use local container storage for postgres - RunPod network volumes don't support
# POSIX permissions (chmod/chown are no-ops), which breaks initdb.
PG_DATA="/var/lib/postgresql/16/main"
PG_BACKUP="$DATA_DIR/postgres_backup"

# ============================================================
# Initialize persistent data directory on RunPod network volume
# ============================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "WARNING: /data is not mounted. Data will not persist across pod restarts."
    echo "Create a RunPod Network Volume and mount it at /data."
    mkdir -p "$DATA_DIR"
fi

# Create persistent subdirectories on network volume
mkdir -p "$PG_BACKUP"
mkdir -p "$DATA_DIR/redis"
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/logs"

# ============================================================
# Initialize PostgreSQL on local storage (supports permissions)
# ============================================================

if [ ! -f "$PG_DATA/PG_VERSION" ]; then
    echo "Initializing PostgreSQL database..."
    mkdir -p "$PG_DATA"
    chown postgres:postgres "$PG_DATA"
    chmod 700 "$PG_DATA"
    su - postgres -c "/usr/lib/postgresql/16/bin/initdb -D $PG_DATA"

    # Configure PostgreSQL
    echo "listen_addresses = 'localhost'" >> "$PG_DATA/postgresql.conf"
    echo "shared_preload_libraries = 'timescaledb'" >> "$PG_DATA/postgresql.conf"
    echo "max_connections = 50" >> "$PG_DATA/postgresql.conf"
    echo "shared_buffers = 256MB" >> "$PG_DATA/postgresql.conf"

    # Allow local connections without password
    echo "local all all trust" > "$PG_DATA/pg_hba.conf"
    echo "host all all 127.0.0.1/32 trust" >> "$PG_DATA/pg_hba.conf"

    # Start PostgreSQL temporarily to create DB and restore/migrate
    su - postgres -c "/usr/lib/postgresql/16/bin/pg_ctl -D $PG_DATA -l $DATA_DIR/logs/postgres_init.log start"
    sleep 2

    su - postgres -c "createdb scalper" || true

    if [ -f "$PG_BACKUP/scalper.sql" ]; then
        echo "Restoring database from network volume backup..."
        su - postgres -c "psql -d scalper < $PG_BACKUP/scalper.sql"
    else
        echo "Running initial migrations..."
        su - postgres -c "psql -d scalper -f /root/sol-scalper/src/db/migrations/001_init.sql"
    fi

    su - postgres -c "/usr/lib/postgresql/16/bin/pg_ctl -D $PG_DATA stop"
    echo "PostgreSQL initialized."
fi

# Symlink model directory
mkdir -p /root/sol-scalper/data
ln -sfn "$DATA_DIR/models" /root/sol-scalper/data/models

# ============================================================
# Set default environment variables if not provided
# ============================================================

export TRADING_MODE="${TRADING_MODE:-paper}"
export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://postgres@localhost:5432/scalper}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
export SYMBOL="${SYMBOL:-SOL}"
export RISK_PER_TRADE="${RISK_PER_TRADE:-0.01}"
export MAX_LEVERAGE="${MAX_LEVERAGE:-5.0}"

# Write env vars to a file so supervisord processes can access them
env | grep -E '^(TRADING_MODE|HL_|TELEGRAM_|ANTHROPIC_|DATABASE_URL|REDIS_URL|SYMBOL|RISK_|MAX_)' > /root/sol-scalper/.env.runtime 2>/dev/null || true

# ============================================================
# Start all services via supervisord
# ============================================================

echo ""
echo "=========================================="
echo "  SOL Scalper - Starting Services"
echo "=========================================="
echo "  Mode:   $TRADING_MODE"
echo "  Symbol: $SYMBOL"
echo "  Risk:   $RISK_PER_TRADE per trade"
echo "  Data:   $DATA_DIR"
echo "=========================================="
echo ""

exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/sol-scalper.conf
