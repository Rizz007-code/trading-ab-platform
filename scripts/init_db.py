# scripts/init_db.py
"""
One-time database initializer.
Run this ONCE before starting the platform for the first time.

Usage:
    python scripts/init_db.py

What it does:
    1. Verifies DB connection
    2. Creates all tables defined in models.py
    3. Prints confirmation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from data.database.connection import check_db_connection, init_db, DATABASE_URL


def main():
    logger.info("=" * 50)
    logger.info("Trading Platform — Database Initializer")
    logger.info("=" * 50)
    logger.info(f"Target: {DATABASE_URL.split('@')[-1]}")  # Hide credentials in log

    # Step 1: Check connection
    logger.info("\n[1/2] Checking database connection...")
    if not check_db_connection():
        logger.error("Cannot connect to database. Check your .env settings and ensure PostgreSQL is running.")
        sys.exit(1)

    # Step 2: Create tables
    logger.info("\n[2/2] Creating tables...")
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        sys.exit(1)

    logger.info("\n" + "=" * 50)
    logger.info("✅ Database initialized successfully!")
    logger.info("\nTables created:")
    logger.info("  • raw_prices")
    logger.info("  • features")
    logger.info("  • strategy_signals")
    logger.info("  • backtest_results")
    logger.info("  • experiments")
    logger.info("  • experiment_results")
    logger.info("  • ml_predictions")
    logger.info("\nNext step: python data/ingestion/fetcher.py full")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
