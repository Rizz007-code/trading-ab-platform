# airflow/Dockerfile
FROM apache/airflow:2.9.1-python3.11

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir \
    yfinance==0.2.40 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    sqlalchemy==2.0.30 \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.1 \
    loguru==0.7.2 \
    tenacity==8.3.0 \
    ta==0.11.0 \
    scipy==1.13.0

# Copy project source into the Airflow container
# DAGs are mounted via volume in docker-compose, but we copy the
# data pipeline modules so tasks can import them
COPY --chown=airflow:root data/ /opt/airflow/data/
COPY --chown=airflow:root .env.example /opt/airflow/.env

WORKDIR /opt/airflow
