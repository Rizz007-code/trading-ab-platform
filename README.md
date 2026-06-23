# 📈 Trading A/B Platform

> **Which trading strategy is actually better? Don't guess — prove it with data and statistics.**

The Trading A/B Platform is an end-to-end **MLOps system** that fetches real stock-market data, runs competing trading strategies against history (**backtesting**), and uses **statistical A/B testing** to decide — rigorously — which strategy wins. It then trains a **machine-learning model** that recommends the best strategy for *current* market conditions. Everything is wrapped in a REST API, a visual dashboard, and a daily automated data pipeline.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow">
  <img alt="Airflow" src="https://img.shields.io/badge/Airflow-orchestration-017CEE?logo=apacheairflow">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

> ⚠️ **Educational project only — not financial advice.** Backtested results do not guarantee future returns. Nothing here is a recommendation to trade real money. See the [disclaimer](#%EF%B8%8F-disclaimer).

---

## 📑 Table of Contents

1. [What problem does this solve?](#-what-problem-does-this-solve)
2. [New to these terms? Start here](#-new-to-these-terms-start-here)
3. [Features](#-features)
4. [How it works](#-how-it-works)
5. [The services & ports](#-the-services--ports)
6. [Tech stack](#-tech-stack)
7. [Prerequisites](#-prerequisites)
8. [Quick start (the easy way, with Docker)](#-quick-start-the-easy-way-with-docker)
9. [Running locally (without Docker)](#-running-locally-without-docker-advanced)
10. [Using the platform](#-using-the-platform-a-typical-workflow)
11. [Project structure](#-project-structure)
12. [Testing](#-testing)
13. [Development phases](#-development-phases)
14. [Disclaimer](#%EF%B8%8F-disclaimer)
15. [License](#-license)

---

## 🎯 What problem does this solve?

Everyone has an opinion on which trading strategy is "best." But opinions are cheap, and a strategy that *looks* good might just have gotten lucky. This platform answers the question **scientifically**:

- **Backtest fairly.** Run two strategies on the *same* stock over the *same* period, with realistic trading costs (commission + slippage).
- **Test for significance.** Don't trust a small difference — run real **statistical tests** (Welch's t-test, Mann-Whitney U, bootstrap confidence intervals) to check whether one strategy genuinely beats the other or it's just noise.
- **Pick a winner only when the data says so.** A strategy is declared the winner by Sharpe ratio **only if** the difference is statistically significant.
- **Predict the future winner.** A machine-learning model learns from past experiments to recommend which strategy fits *today's* market conditions.
- **Automate everything.** A daily pipeline keeps the data fresh, with full experiment tracking and a dashboard to explore results.

This is essentially the same **A/B-testing discipline** used by tech companies to compare website variants — applied to trading strategies.

---

## 🧑‍🎓 New to these terms? Start here

| Term | Plain-English meaning |
|---|---|
| **Trading strategy** | A set of rules that decides when to **BUY**, **SELL**, or **HOLD** a stock. |
| **Signal** | The output of a strategy on a given day: BUY / SELL / HOLD. |
| **Backtest** | Replaying a strategy over historical data to see how it *would* have performed. |
| **A/B test** | Comparing two options (here, two strategies) head-to-head to find the better one — and proving the difference is real, not luck. |
| **Technical indicator** | A number computed from price/volume that hints at trends — e.g. **Moving Average (MA)**, **RSI**, **MACD**. |
| **Sharpe ratio** | Return earned per unit of risk. Higher = better risk-adjusted performance. The main "who wins" metric here. |
| **Max drawdown** | The worst peak-to-trough drop — how much you'd have lost at the scariest moment. |
| **p-value** | The chance the observed difference happened by luck. Small (< 0.05) = "probably real." |
| **MLflow** | A tool that tracks every model experiment and stores the trained models. |
| **Airflow** | A scheduler that runs the data pipeline automatically every day. |

---

## ✨ Features

- 📥 **Automated data ingestion** — pulls historical & daily stock prices from Yahoo Finance (`yfinance`) into PostgreSQL, with data-quality validation (gaps, outliers, bad prices).
- 🧮 **Feature engineering** — computes MA-50/200, RSI, MACD, volatility, ATR, volume z-score, market regime (bull/bear/sideways), and relative strength vs. the S&P 500.
- 🎯 **Pluggable strategies** — three built in (and easy to add more):
  - **Strategy A** — 50-day Moving-Average crossover (the baseline)
  - **Strategy B** — MA crossover **+ RSI filter** (avoids buying overbought)
  - **Strategy C** — MACD crossover (momentum-based)
- 🔬 **Backtesting simulator** — realistic transaction costs, position sizing, and portfolio tracking.
- 📊 **Statistical A/B engine** — t-test, Mann-Whitney U, and bootstrap confidence intervals decide the winner.
- 🤖 **ML strategy selector** — XGBoost & LightGBM (5-fold cross-validation) trained to recommend the best strategy for current conditions, tracked & versioned in **MLflow**.
- 🌐 **REST API** — FastAPI endpoints to list strategies, run experiments, and get predictions (interactive Swagger docs included).
- 🖥️ **Streamlit dashboard** — 5 pages: overview KPIs, run an experiment, browse history, ML predictions, and strategy details.
- 🔄 **Daily orchestration** — an Airflow DAG keeps data fresh: `fetch → validate → engineer features → health check`.
- 🐳 **One-command setup** — `docker compose up` launches the whole stack; GitHub Actions CI runs lint + tests.

---

## ⚙️ How it works

The platform is a pipeline of stages. Data flows left-to-right:

```
 (1) INGESTION           (2) FEATURES           (3) STRATEGIES
 ┌──────────────┐        ┌──────────────┐       ┌──────────────────────┐
 │ yfinance     │        │ Compute      │       │ A: MA crossover      │
 │ → PostgreSQL │  ───►  │ indicators   │  ───► │ B: MA + RSI filter   │
 │ + validate   │        │ (RSI, MACD…) │       │ C: MACD crossover    │
 └──────────────┘        └──────────────┘       └──────────┬───────────┘
                                                            │ BUY/SELL/HOLD signals
                                                            ▼
 (6) DASHBOARD + API     (5) ML SELECTOR        (4) BACKTEST + A/B TEST
 ┌──────────────┐        ┌──────────────┐       ┌──────────────────────┐
 │ Streamlit    │  ◄───  │ XGBoost /    │  ◄─── │ Simulate both        │
 │ + FastAPI    │        │ LightGBM     │       │ → metrics (Sharpe…)  │
 │ (explore &   │        │ recommends   │       │ → t-test / bootstrap │
 │  trigger)    │        │ best strategy│       │ → declare winner     │
 └──────────────┘        └──────┬───────┘       └──────────────────────┘
                                │ tracked in MLflow

      🔄  Airflow runs steps (1) → (2) automatically every day
```

**In words:**
1. **Ingestion** — `yfinance` downloads prices into PostgreSQL; a validator catches bad data.
2. **Features** — technical indicators are computed and stored.
3. **Strategies** — each strategy turns prices+features into BUY/SELL/HOLD signals.
4. **Backtest + A/B test** — the simulator replays two strategies; the A/B engine computes performance metrics and runs statistical tests to pick a significant winner.
5. **ML selector** — a model learns from many past experiments to predict the best strategy for current market features; everything is logged in MLflow.
6. **API + dashboard** — you trigger experiments and view results through the FastAPI backend and the Streamlit UI.

---

## 🔌 The services & ports

When the stack is running, these are available in your browser:

| Service | URL | What it is |
|---|---|---|
| **Dashboard** | http://localhost:8501 | 👈 **Start here** — the Streamlit UI |
| **API docs** | http://localhost:8000/docs | Interactive Swagger API documentation |
| **MLflow** | http://localhost:5001 | Experiment tracking & model registry |
| **Airflow** | http://localhost:8080 | Pipeline scheduler (login: `admin` / `admin`) |
| **PostgreSQL** | localhost:5432 | The trading database |

---

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Market data** | yfinance (Yahoo Finance) |
| **Database** | PostgreSQL 16 + SQLAlchemy 2.0 |
| **Strategies / indicators** | pandas · NumPy · `ta` |
| **Statistics** | SciPy (t-test, Mann-Whitney, bootstrap) |
| **Machine learning** | XGBoost · LightGBM · scikit-learn |
| **Experiment tracking** | MLflow |
| **API** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit + Plotly |
| **Orchestration** | Apache Airflow (LocalExecutor) |
| **Infra / CI** | Docker Compose · GitHub Actions · Ruff · Pytest |

---

## ✅ Prerequisites

To run the easy (Docker) way you only need:

1. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** — installed and running. It bundles Python, PostgreSQL, Airflow, and MLflow for you.

That's it — **no API keys required** (Yahoo Finance data is free), and you don't need to install Python or a database separately.

---

## 🚀 Quick start (the easy way, with Docker)

```bash
# 1. Get the code
git clone https://github.com/Rizz007-code/trading-ab-platform.git
cd trading-ab-platform

# 2. Create your config file (defaults work out of the box)
cp .env.example .env

# 3. Start the entire platform (first run downloads images — be patient)
docker compose up --build
```

That single command launches PostgreSQL, Airflow, MLflow, the API, and the dashboard. Once everything is healthy, open:

- 👉 **http://localhost:8501** — the dashboard
- **http://localhost:8000/docs** — the API
- **http://localhost:5001** — MLflow

To load some initial data, trigger the **data pipeline** DAG in Airflow (http://localhost:8080, login `admin`/`admin`), or run the fetcher manually (see below).

> Stop everything with `docker compose down`. Add `-v` to also wipe the database volumes.

---

## 🛠️ Running locally (without Docker, advanced)

Requires **Python 3.12+** and a running **PostgreSQL 16**.

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env               # edit DB credentials if needed

# 3. Initialise the database schema
python scripts/init_db.py

# 4. Load market data (first time pulls a few years of history)
python -m data.ingestion.fetcher

# 5. Start the API (terminal 1)
uvicorn api.main:app --reload --port 8000

# 6. Start the dashboard (terminal 2)
streamlit run dashboard/app.py
```

> **MLflow (optional, terminal 3):** `mlflow server --host 0.0.0.0 --port 5001` — needed for ML training/prediction features.

---

## 🧭 Using the platform (a typical workflow)

1. **Load data** — run the Airflow DAG (or the fetcher) to populate prices & features.
2. **Run an A/B experiment** — in the dashboard's *Run Experiment* page (or `POST /api/v1/experiments/run`), pick two strategies and a ticker (e.g. compare **Strategy A vs. Strategy B** on `AAPL`, 2022–2024).
3. **Read the result** — see each strategy's Sharpe ratio, returns, drawdown, and whether the winner is **statistically significant**.
4. **Browse history** — every experiment is stored; review past results on the *Experiment History* page.
5. **Train the ML selector** — once you have several experiments, train the model (logged to MLflow).
6. **Get a recommendation** — `POST /api/v1/predictions/` (or the *ML Predictions* page) returns the model's suggested strategy for a ticker's current conditions, with a confidence score.

---

## 📁 Project structure

```
trading-ab-platform/
├── data/
│   ├── ingestion/         # Fetch prices (yfinance) + data-quality validation
│   ├── features/          # Technical-indicator feature engineering
│   └── database/          # SQLAlchemy models, connection, schema
├── strategies/
│   ├── base_strategy.py   # Abstract base — all strategies extend this
│   ├── strategy_a.py      # MA crossover (baseline)
│   ├── strategy_b.py      # MA crossover + RSI filter
│   ├── strategy_c.py      # MACD crossover
│   └── simulator.py       # Backtester (costs, position sizing, returns)
├── ab_testing/
│   ├── engine.py          # Orchestrates a full A/B experiment
│   ├── metrics.py         # Sharpe, CAGR, volatility, drawdown, win rate
│   └── statistical_tests.py  # t-test, Mann-Whitney U, bootstrap CI
├── ml/
│   ├── trainer.py         # Train XGBoost/LightGBM strategy selector
│   ├── predictor.py       # Load model from MLflow → predict best strategy
│   └── mlflow_utils.py    # MLflow logging & model-registry helpers
├── api/
│   ├── main.py            # FastAPI app
│   ├── routers/           # /strategies, /experiments, /predictions
│   └── schemas.py         # Pydantic request/response models
├── dashboard/
│   └── app.py             # Streamlit UI (5 pages)
├── airflow/
│   ├── Dockerfile         # Isolated Airflow image (avoids dep conflicts)
│   └── dags/              # Daily data-pipeline DAG
├── scripts/               # DB init helpers
├── tests/                 # unit/ + integration/ (pytest)
├── docker-compose.yml     # Full stack: postgres, airflow, mlflow, api, dashboard
├── Dockerfile.api · Dockerfile.dashboard
└── requirements.txt · pyproject.toml
```

---

## 🧪 Testing

```bash
# Run the full test suite
pytest

# Just the fast unit tests (no database needed)
pytest tests/unit

# With coverage (fails under 60%)
pytest --cov=. --cov-report=term-missing
```

Tests cover the metrics math, statistical tests, strategy signal generation, API schemas, and an API integration test. Linting uses **Ruff**; both run automatically in GitHub Actions CI on every push.

---

## 🗺️ Development phases

The platform was built in clear, reviewable stages:

| Phase | Description | Status |
|:---:|---|:---:|
| 0 | Project scaffold, PostgreSQL schema, Docker & CI setup | ✅ |
| 1 | Data ingestion — yfinance → PostgreSQL with validation | ✅ |
| 2 | Feature engineering — technical indicators & market regime | ✅ |
| 3 | Trading strategies (A: MA, B: MA+RSI, C: MACD) + base class | ✅ |
| 4 | Backtesting simulator + statistical A/B test engine | ✅ |
| 5 | ML strategy selector (XGBoost/LightGBM) + MLflow tracking | ✅ |
| 6 | FastAPI REST API (strategies, experiments, predictions) | ✅ |
| 7 | Streamlit dashboard (overview, run, history, ML, strategies) | ✅ |
| 8 | Airflow orchestration — daily automated data pipeline | ✅ |

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. It is **not financial advice** and is **not** a recommendation to buy or sell any security. Backtested and simulated performance is hypothetical, has many limitations, and **does not guarantee future results**. Trading involves substantial risk of loss. Do your own research and consult a licensed professional before making any investment decisions.

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and learn from.
