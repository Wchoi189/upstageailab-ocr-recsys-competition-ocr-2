# Single image with NVIDIA CUDA + Airflow 3.1.5 (Python 3.11)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    AIRFLOW_HOME=/opt/airflow \
    PYTHONUNBUFFERED=1

USER root

# Base OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl ca-certificates jq git \
    build-essential gcc \
    libpq-dev libssl-dev libffi-dev \
    locales \
  && rm -rf /var/lib/apt/lists/*

# Python 3.11 + pip bootstrap (Ubuntu 22.04 needs deadsnakes for 3.11)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-distutils python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip /usr/local/bin/pip3 || true && \
    rm -rf /var/lib/apt/lists/*

# Create airflow user and directories
RUN useradd -ms /bin/bash -d ${AIRFLOW_HOME} airflow && \
    mkdir -p ${AIRFLOW_HOME}/{dags,logs,plugins,src,config} && \
    chown -R airflow:airflow ${AIRFLOW_HOME}

# Airflow install with constraints for Python 3.11
ARG AIRFLOW_VERSION=3.1.5
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

# Remove distutils blinker package to avoid conflict
RUN rm -rf /usr/lib/python3/dist-packages/blinker* || true

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --break-system-packages \
                          "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
                          "apache-airflow-providers-fab" \
      --constraint "${CONSTRAINT_URL}"

ENV AIRFLOW__CORE__AUTH_MANAGER=airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager

# Project Python deps
COPY requirements.txt /requirements.txt
RUN python -m pip install --no-cache-dir -r /requirements.txt

# Environment
ENV PYTHONPATH="${AIRFLOW_HOME}:${AIRFLOW_HOME}/src" \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor

USER airflow
WORKDIR ${AIRFLOW_HOME}
