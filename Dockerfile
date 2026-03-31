FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python-is-python3 \
    build-essential \
    libpq-dev \
    gcc \
    git \
    curl \
    wget \
    nano \
    less \
    tmux \
    supervisor \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# --- Install PostgreSQL 16 + TimescaleDB ---
RUN curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey | gpg --dearmor -o /usr/share/keyrings/timescaledb-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/timescaledb-keyring.gpg] https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/timescaledb.list \
    && apt-get update \
    && apt-get install -y postgresql-16 timescaledb-2-postgresql-16 \
    && rm -rf /var/lib/apt/lists/*

# --- Install Redis ---
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*

# --- Clone the repo into /root/sol-scalper (allows git pull for updates) ---
WORKDIR /root/sol-scalper

# Copy everything into the image
COPY . /root/sol-scalper/

# --- Install Python dependencies ---
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Make scripts executable ---
RUN chmod +x /root/sol-scalper/scripts/*.sh \
    && chmod +x /root/sol-scalper/entrypoint.sh

# --- Supervisord config ---
COPY supervisord.conf /etc/supervisor/conf.d/sol-scalper.conf

# --- Shell setup ---
COPY .bashrc_docker /root/.bashrc

WORKDIR /root/sol-scalper

ENTRYPOINT ["/root/sol-scalper/entrypoint.sh"]
CMD ["/bin/bash", "-l"]
