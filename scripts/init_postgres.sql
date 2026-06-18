-- scripts/init_postgres.sql
-- Runs automatically on first postgres container start.
-- Creates the airflow DB alongside our trading DB.

-- Trading DB is already created by POSTGRES_DB env var.
-- We just need the Airflow DB.

SELECT 'CREATE DATABASE airflow_db'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'airflow_db'
)\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO trading_user;
