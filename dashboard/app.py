# dashboard/app.py
"""
Trading A/B Platform — Streamlit Dashboard (Phase 7)

Five pages navigated via the sidebar:
  🏠 Overview          — platform KPIs and recent experiments
  🧪 Run Experiment    — form to trigger a new A/B backtest
  📋 Experiment History — browse and inspect past experiments
  🤖 ML Predictions    — model recommendations per ticker
  📊 Strategies        — inspect registered trading strategies

All data comes from the FastAPI backend at API_BASE_URL.
Start the API first:  uvicorn api.main:app --reload --port 8000
Then run:             streamlit run dashboard/app.py
"""

import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",
    "AMZN", "META", "NFLX", "BABA", "SPY",
]

STRATEGIES = ["MACrossoverStrategy", "MARSIStrategy", "MACDStrategy"]

st.set_page_config(
    page_title = "Trading A/B Platform",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ── API client ────────────────────────────────────────────────────────────────

def _api_get(path: str, params: dict | None = None) -> tuple:
    """GET request to the API. Returns (data, error_string)."""
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to API at {API_BASE_URL}. Is the server running?"
    except requests.exceptions.HTTPError as exc:
        return None, f"API {exc.response.status_code}: {exc.response.text[:200]}"
    except Exception as exc:
        return None, str(exc)


def _api_post(path: str, payload: dict) -> tuple:
    """POST request to the API. Returns (data, error_string)."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}{path}", json=payload, timeout=120
        )
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to API at {API_BASE_URL}. Is the server running?"
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text[:200]
        return None, f"API {exc.response.status_code}: {detail}"
    except Exception as exc:
        return None, str(exc)


# ── Chart helpers ─────────────────────────────────────────────────────────────

_COLOR_A  = "#4C72B0"
_COLOR_B  = "#DD8452"
_COLOR_WIN = "#2ECC71"

def _bar_comparison(
    values_a: list,
    values_b: list,
    name_a:   str,
    name_b:   str,
    labels:   list,
    title:    str,
    fmt:      str = ".3f",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name         = name_a,
        x            = labels,
        y            = values_a,
        marker_color = _COLOR_A,
        text         = [f"{v:{fmt}}" for v in values_a],
        textposition = "outside",
    ))
    fig.add_trace(go.Bar(
        name         = name_b,
        x            = labels,
        y            = values_b,
        marker_color = _COLOR_B,
        text         = [f"{v:{fmt}}" for v in values_b],
        textposition = "outside",
    ))
    fig.update_layout(
        title        = title,
        barmode      = "group",
        height       = 360,
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor= "rgba(0,0,0,0)",
        legend       = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis        = dict(gridcolor="#E8E8E8"),
    )
    return fig


def _gauge(value_pct: float, label: str) -> go.Figure:
    color = _COLOR_WIN if value_pct >= 60 else ("#E67E22" if value_pct >= 40 else "#E74C3C")
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value_pct,
        title = {"text": label, "font": {"size": 14}},
        gauge = {
            "axis":  {"range": [0, 100], "ticksuffix": "%"},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#FADBD8"},
                {"range": [40, 70], "color": "#FDEBD0"},
                {"range": [70, 100],"color": "#D5F5E3"},
            ],
        },
        number = {"suffix": "%", "font": {"size": 28}},
    ))
    fig.update_layout(height=240, margin=dict(t=50, b=10, l=20, r=20))
    return fig


def _status_badge(status: str) -> str:
    icons = {"completed": "🟢", "running": "🟡", "failed": "🔴", "pending": "⚪"}
    return f"{icons.get(status, '⚪')} {status.capitalize()}"


# ── Shared component: experiment result display ───────────────────────────────

def _show_experiment_result(result: dict) -> None:
    """Render a full A/B result dict (from POST /run or GET /{id})."""
    winner      = result.get("winner")
    sig         = result.get("is_significant", False)
    lift        = result.get("lift_pct")
    sa          = result.get("strategy_a", {})
    sb          = result.get("strategy_b", {})
    tests       = result.get("statistical_tests", {})
    name_a      = sa.get("name", "Strategy A")
    name_b      = sb.get("name", "Strategy B")

    # ── Winner banner
    if winner and sig:
        st.success(
            f"🏆 **Winner: {winner}**"
            + (f"   |   Sharpe lift: **{lift:.1f}%**" if lift else "")
        )
    elif not sig:
        st.info("📊 No statistically significant winner (results are too close to call).")
    else:
        st.warning("⚠️ Result inconclusive.")

    # ── KPI cards
    st.markdown("#### Performance Metrics")
    col_a, col_b = st.columns(2)

    def _metric_col(col, label, m, is_winner):
        border = "border: 2px solid #2ECC71;" if is_winner else ""
        col.markdown(
            f"""<div style="padding:12px; border-radius:8px; background:#F8F9FA; {border}">
            <h4 style="margin:0">{label}</h4>
            <small>{m.get('description','')}</small></div>""",
            unsafe_allow_html=True,
        )
        col.metric("Sharpe Ratio",   f"{m.get('sharpe', 0):.4f}")
        col.metric("Annual Return",  f"{m.get('annual_return', 0)*100:.2f}%")
        col.metric("Max Drawdown",   f"{m.get('max_drawdown', 0)*100:.2f}%")
        col.metric("Win Rate",       f"{m.get('win_rate', 0)*100:.2f}%")
        col.metric("Total Return",   f"{m.get('total_return', 0)*100:.2f}%")
        col.metric("Trades",         m.get('num_trades', 0))

    _metric_col(col_a, name_a, sa, winner == name_a)
    _metric_col(col_b, name_b, sb, winner == name_b)

    # ── Bar charts
    st.markdown("#### Visual Comparison")
    tab_ret, tab_risk, tab_trade = st.tabs(["Returns", "Risk", "Trading Activity"])

    with tab_ret:
        st.plotly_chart(
            _bar_comparison(
                [sa.get("sharpe", 0),        sb.get("sharpe", 0)],
                [sa.get("annual_return", 0), sb.get("annual_return", 0)],
                name_a, name_b,
                ["Sharpe Ratio", "Annual Return"],
                "Returns Comparison",
            ),
            use_container_width=True,
        )

    with tab_risk:
        st.plotly_chart(
            _bar_comparison(
                [abs(sa.get("max_drawdown", 0)), sa.get("volatility", 0)],
                [abs(sb.get("max_drawdown", 0)), sb.get("volatility", 0)],
                name_a, name_b,
                ["Max Drawdown", "Volatility"],
                "Risk Comparison (lower is better)",
            ),
            use_container_width=True,
        )

    with tab_trade:
        st.plotly_chart(
            _bar_comparison(
                [sa.get("win_rate", 0), sa.get("num_trades", 0)],
                [sb.get("win_rate", 0), sb.get("num_trades", 0)],
                name_a, name_b,
                ["Win Rate", "Num Trades"],
                "Trading Activity",
            ),
            use_container_width=True,
        )

    # ── Statistical tests
    st.markdown("#### Statistical Tests")
    t   = tests.get("t_test", {})
    mw  = tests.get("mann_whitney", {})
    ci  = tests.get("bootstrap_ci_sharpe_diff", {})

    test_df = pd.DataFrame({
        "Test":        ["Welch's t-test", "Mann-Whitney U", "Bootstrap Sharpe CI"],
        "Statistic":   [
            f"{t.get('statistic', 0):.4f}",
            f"{mw.get('statistic', 0):.0f}",
            f"[{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}]",
        ],
        "p-value":     [
            f"{t.get('p_value', 1):.4f}",
            f"{mw.get('p_value', 1):.4f}",
            "—",
        ],
        "Significant": [
            "✅ Yes" if t.get("is_significant")  else "❌ No",
            "✅ Yes" if mw.get("is_significant") else "❌ No",
            "✅ Yes" if ci.get("excludes_zero")  else "❌ No",
        ],
    })
    st.dataframe(test_df, use_container_width=True, hide_index=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_overview() -> None:
    st.title("📈 Trading A/B Platform")
    st.caption("Real-time overview of experiments, results, and ML predictions.")

    # Health check
    health, err = _api_get("/health")
    if err:
        st.error(f"⚠️ {err}")
        st.stop()

    db_icon = "🟢" if health.get("db_connected") else "🔴"
    exps, _  = _api_get("/api/v1/experiments", params={"limit": 100})
    strats, _ = _api_get("/api/v1/strategies")
    exps      = exps or []
    strats    = strats or []

    completed = [e for e in exps if e.get("status") == "completed"]
    latest_winner = completed[0].get("winner") if completed else "—"

    # ── KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("API",              f"{db_icon} {health.get('status','').title()}")
    k2.metric("Experiments",      len(exps))
    k3.metric("Completed",        len(completed))
    k4.metric("Strategies",       len(strats))
    k5.metric("Latest Winner",    latest_winner or "—")

    st.divider()

    # ── Recent experiments table
    st.subheader("Recent Experiments")
    if exps:
        df = pd.DataFrame(exps)
        df["Status"] = df["status"].apply(_status_badge)
        show_cols = {
            "name": "Name", "ticker": "Ticker",
            "strategy_a": "Strategy A", "strategy_b": "Strategy B",
            "Status": "Status", "winner": "Winner",
        }
        display_df = df[[c for c in show_cols if c in df.columns]].rename(columns=show_cols)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No experiments yet. Use **🧪 Run Experiment** to create the first one.")

    st.divider()

    # ── Available strategies
    st.subheader("Available Strategies")
    if strats:
        cols = st.columns(len(strats))
        for col, s in zip(cols, strats):
            with col:
                st.markdown(
                    f"""<div style="padding:14px;border-radius:8px;
                    background:#F0F2F6;height:100%">
                    <b>{s['name']}</b><br>
                    <small style="color:#666">{s['description']}</small>
                    </div>""",
                    unsafe_allow_html=True,
                )


def page_run_experiment() -> None:
    st.title("🧪 Run A/B Experiment")
    st.caption(
        "Compare two strategies on historical data. "
        "Results include Sharpe ratio, drawdown, and three statistical tests."
    )

    with st.form("run_exp_form"):
        c1, c2 = st.columns(2)
        strategy_a = c1.selectbox("Strategy A (Control)",   STRATEGIES, index=0)
        strategy_b = c2.selectbox("Strategy B (Challenger)", STRATEGIES, index=1)

        c3, c4 = st.columns(2)
        ticker     = c3.selectbox("Ticker", TICKERS, index=0)
        exp_name   = c4.text_input("Experiment Name (optional)", placeholder="auto-generated if blank")

        c5, c6 = st.columns(2)
        start_date = c5.date_input("Start Date", value=date.today() - timedelta(days=730))
        end_date   = c6.date_input("End Date",   value=date.today() - timedelta(days=1))

        with st.expander("Advanced settings"):
            ca, cb, cc = st.columns(3)
            initial_capital  = ca.number_input("Initial Capital ($)", value=100_000, step=10_000, min_value=1_000)
            confidence_level = cb.slider("Confidence Level", 0.80, 0.99, 0.95, step=0.01)
            n_bootstrap      = cc.number_input("Bootstrap Resamples", value=500, step=100, min_value=100, max_value=5000)
            save_to_db       = st.checkbox("Save to database", value=True)

        submitted = st.form_submit_button("▶ Run Experiment", type="primary", use_container_width=True)

    if submitted:
        if strategy_a == strategy_b:
            st.warning("Strategy A and B must be different.")
            return
        if start_date >= end_date:
            st.warning("Start date must be before end date.")
            return

        payload = {
            "strategy_a":      strategy_a,
            "strategy_b":      strategy_b,
            "ticker":          ticker,
            "start_date":      str(start_date),
            "end_date":        str(end_date),
            "experiment_name": exp_name or None,
            "initial_capital":  float(initial_capital),
            "confidence_level": confidence_level,
            "n_bootstrap":      int(n_bootstrap),
            "save_to_db":       save_to_db,
        }

        with st.spinner(f"Running backtest for {ticker}… (this may take 10–30 seconds)"):
            result, err = _api_post("/api/v1/experiments/run", payload)

        if err:
            st.error(f"Experiment failed: {err}")
        else:
            st.success("Experiment complete!")
            st.session_state["last_result"] = result
            _show_experiment_result(result)


def page_experiment_history() -> None:
    st.title("📋 Experiment History")

    col_filter, col_limit = st.columns([3, 1])
    ticker_filter = col_filter.text_input("Filter by ticker (blank = all)", "")
    limit         = col_limit.selectbox("Show", [10, 25, 50, 100], index=1)

    exps, err = _api_get("/api/v1/experiments", params={"limit": limit})
    if err:
        st.error(err)
        return
    if not exps:
        st.info("No experiments in the database yet.")
        return

    if ticker_filter:
        exps = [e for e in exps if ticker_filter.upper() in e.get("ticker", "").upper()]

    # Summary table
    df = pd.DataFrame(exps)
    df["Status"]  = df["status"].apply(_status_badge)
    df["Winner"]  = df["winner"].fillna("—")
    show_cols = {
        "id": "ID", "name": "Name", "ticker": "Ticker",
        "strategy_a": "Strategy A", "strategy_b": "Strategy B",
        "Status": "Status", "Winner": "Winner", "created_at": "Created",
    }
    display_df = df[[c for c in show_cols if c in df.columns]].rename(columns=show_cols)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # Drill-down
    st.subheader("Experiment Detail")
    exp_ids = [e["id"] for e in exps]
    if exp_ids:
        selected_id = st.selectbox("Select Experiment ID", exp_ids)
        detail, err = _api_get(f"/api/v1/experiments/{selected_id}")
        if err:
            st.error(err)
            return

        st.caption(
            f"**{detail['name']}** | {detail['ticker']} | "
            f"{detail['start_date']} → {detail['end_date']} | "
            f"{_status_badge(detail['status'])}"
        )

        result = detail.get("result")
        if result is None:
            st.info("This experiment has no result yet (status may be pending/failed).")
        else:
            # Reconstruct the same format as the run response
            strats, _ = _api_get("/api/v1/strategies")
            name_map  = {s["name"]: s["description"] for s in (strats or [])}
            sa_name   = detail["strategy_a"]
            sb_name   = detail["strategy_b"]

            # Build a compatible result dict from the detail schema
            run_result = {
                "experiment_name":  detail["name"],
                "ticker":           detail["ticker"],
                "start_date":       str(detail["start_date"]),
                "end_date":         str(detail["end_date"]),
                "winner":           result.get("winner"),
                "is_significant":   result.get("is_significant", False),
                "lift_pct":         result.get("lift_pct"),
                "confidence_level": result.get("confidence_level", 0.95),
                "strategy_a": {
                    "name":          sa_name,
                    "description":   name_map.get(sa_name, ""),
                    "sharpe":        result.get("sharpe_a", 0),
                    "annual_return": result.get("annual_return_a", 0),
                    "volatility":    result.get("volatility_a", 0),
                    "max_drawdown":  result.get("max_drawdown_a", 0),
                    "win_rate":      result.get("win_rate_a", 0),
                    "total_return":  0,
                    "num_trades":    0,
                },
                "strategy_b": {
                    "name":          sb_name,
                    "description":   name_map.get(sb_name, ""),
                    "sharpe":        result.get("sharpe_b", 0),
                    "annual_return": result.get("annual_return_b", 0),
                    "volatility":    result.get("volatility_b", 0),
                    "max_drawdown":  result.get("max_drawdown_b", 0),
                    "win_rate":      result.get("win_rate_b", 0),
                    "total_return":  0,
                    "num_trades":    0,
                },
                "statistical_tests": {
                    "t_test":       {"statistic": 0, "p_value": result.get("p_value", 1), "is_significant": result.get("is_significant", False)},
                    "mann_whitney": {"statistic": 0, "p_value": 1, "is_significant": False},
                    "bootstrap_ci_sharpe_diff": {
                        "lower":         result.get("ci_lower", 0),
                        "upper":         result.get("ci_upper", 0),
                        "n_bootstrap":   0,
                        "excludes_zero": not (result.get("ci_lower", 0) <= 0 <= result.get("ci_upper", 0)),
                    },
                },
            }
            _show_experiment_result(run_result)


def page_predictions() -> None:
    st.title("🤖 ML Predictions")
    st.caption(
        "The ML model predicts which strategy is best for current market conditions "
        "using RSI, MACD, volatility, regime, and MA ratio features."
    )

    tab_single, tab_batch = st.tabs(["Single Ticker", "Batch"])

    # ── Single ticker
    with tab_single:
        c1, c2, c3 = st.columns([2, 2, 1])
        ticker   = c1.selectbox("Ticker", TICKERS, key="pred_ticker")
        as_of    = c2.date_input("As Of Date", value=date.today(), key="pred_date")
        save_db  = c3.checkbox("Save to DB", value=True, key="pred_save")

        if st.button("🔮 Predict", type="primary"):
            payload = {
                "ticker":     ticker,
                "as_of_date": str(as_of),
                "save_to_db": save_db,
            }
            with st.spinner(f"Running prediction for {ticker}…"):
                result, err = _api_post("/api/v1/predictions/", payload)

            if err:
                st.error(f"Prediction failed: {err}")
            else:
                st.session_state[f"pred_{ticker}"] = result

        # Show result if available
        pred = st.session_state.get(f"pred_{ticker}")
        if pred and not isinstance(pred, str):
            st.divider()
            r1, r2 = st.columns([1, 2])

            with r1:
                st.plotly_chart(
                    _gauge(pred["confidence"] * 100, pred["predicted_strategy"]),
                    use_container_width=True,
                )
                st.markdown(
                    f"""
                    | Field | Value |
                    |---|---|
                    | **Ticker** | {pred['ticker']} |
                    | **Recommended** | **{pred['predicted_strategy']}** |
                    | **Confidence** | {pred['confidence']*100:.1f}% |
                    | **Feature Date** | {pred['feature_date']} |
                    """
                )

            with r2:
                if pred.get("probabilities"):
                    prob_df = pd.DataFrame(
                        list(pred["probabilities"].items()),
                        columns=["Strategy", "Probability"],
                    ).sort_values("Probability", ascending=False)

                    fig = go.Figure(go.Bar(
                        x            = prob_df["Probability"] * 100,
                        y            = prob_df["Strategy"],
                        orientation  = "h",
                        marker_color = [
                            _COLOR_WIN if s == pred["predicted_strategy"] else _COLOR_A
                            for s in prob_df["Strategy"]
                        ],
                        text         = [f"{p*100:.1f}%" for p in prob_df["Probability"]],
                        textposition = "outside",
                    ))
                    fig.update_layout(
                        title        = "Strategy Probabilities",
                        xaxis_title  = "Probability (%)",
                        height       = 240,
                        plot_bgcolor = "rgba(0,0,0,0)",
                        paper_bgcolor= "rgba(0,0,0,0)",
                        xaxis        = dict(range=[0, 110]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # History
        st.divider()
        st.subheader(f"Prediction History — {ticker}")
        hist, err = _api_get(f"/api/v1/predictions/{ticker}/history", params={"limit": 10})
        if err:
            st.caption(f"Could not load history: {err}")
        elif hist:
            hist_df = pd.DataFrame(hist)[
                ["date", "predicted_strategy", "confidence", "market_regime", "model_name"]
            ].rename(columns={
                "date": "Date", "predicted_strategy": "Strategy",
                "confidence": "Confidence", "market_regime": "Regime",
                "model_name": "Model",
            })
            hist_df["Confidence"] = hist_df["Confidence"].apply(
                lambda x: f"{x*100:.1f}%" if x else "—"
            )
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No history yet for this ticker.")

    # ── Batch
    with tab_batch:
        selected_tickers = st.multiselect(
            "Select tickers for batch prediction",
            TICKERS,
            default=["AAPL", "MSFT", "GOOGL"],
        )
        batch_date  = st.date_input("As Of Date", value=date.today(), key="batch_date")
        batch_save  = st.checkbox("Save to DB", value=True, key="batch_save")

        if st.button("🔮 Predict All", type="primary"):
            if not selected_tickers:
                st.warning("Select at least one ticker.")
            else:
                payload = {
                    "tickers":    selected_tickers,
                    "as_of_date": str(batch_date),
                    "save_to_db": batch_save,
                }
                with st.spinner(f"Running batch predictions for {len(selected_tickers)} tickers…"):
                    results, err = _api_post("/api/v1/predictions/batch", payload)

                if err:
                    st.error(f"Batch prediction failed: {err}")
                else:
                    st.session_state["batch_results"] = results

        if "batch_results" in st.session_state:
            rows = st.session_state["batch_results"]
            if rows:
                st.divider()
                ok   = [r for r in rows if "error" not in r]
                fail = [r for r in rows if "error" in r]

                if ok:
                    batch_df = pd.DataFrame([{
                        "Ticker":    r["ticker"],
                        "Strategy":  r.get("predicted_strategy", "—"),
                        "Confidence": f"{r.get('confidence', 0)*100:.1f}%" if r.get("confidence") else "—",
                    } for r in ok])
                    st.dataframe(batch_df, use_container_width=True, hide_index=True)

                    # Horizontal bar chart
                    conf_vals = [r.get("confidence", 0) * 100 for r in ok]
                    tickers   = [r["ticker"] for r in ok]
                    strats    = [r.get("predicted_strategy", "") for r in ok]
                    fig = go.Figure(go.Bar(
                        x            = conf_vals,
                        y            = tickers,
                        orientation  = "h",
                        text         = [f"{s} ({c:.0f}%)" for s, c in zip(strats, conf_vals)],
                        textposition = "outside",
                        marker_color = _COLOR_A,
                    ))
                    fig.update_layout(
                        title         = "Batch Prediction Confidence",
                        xaxis_title   = "Confidence (%)",
                        height        = max(200, len(ok) * 45),
                        plot_bgcolor  = "rgba(0,0,0,0)",
                        paper_bgcolor = "rgba(0,0,0,0)",
                        xaxis         = dict(range=[0, 115]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if fail:
                    st.warning("Some tickers failed:")
                    for r in fail:
                        st.caption(f"  ❌ {r['ticker']}: {r['error']}")


def page_strategies() -> None:
    st.title("📊 Strategies")
    st.caption("Inspect the registered trading strategy implementations.")

    strats, err = _api_get("/api/v1/strategies")
    if err:
        st.error(err)
        return
    if not strats:
        st.info("No strategies registered.")
        return

    cols = st.columns(len(strats))
    for col, s in zip(cols, strats):
        with col:
            st.markdown(f"### {s['name']}")
            st.caption(f"v{s['version']}")
            st.info(s["description"])
            st.markdown("**Parameters**")
            st.json(s["parameters"])

    st.divider()
    st.subheader("How to use strategies in an A/B experiment")
    st.markdown("""
1. Go to **🧪 Run Experiment**
2. Pick any two strategies from the dropdowns
3. Select a ticker and date range
4. Click **▶ Run Experiment**

The engine will backtest both strategies and run three statistical tests:
- **Welch's t-test** — tests for difference in mean daily returns
- **Mann-Whitney U** — non-parametric test (handles non-normal return distributions)
- **Bootstrap CI** — 95% confidence interval for the Sharpe ratio difference

A winner is declared only when at least one test is statistically significant (p < 0.05).
""")


# ── Sidebar navigation ────────────────────────────────────────────────────────

def main() -> None:
    with st.sidebar:
        st.markdown("## 📈 Trading A/B Platform")
        st.caption(f"API: `{API_BASE_URL}`")
        st.divider()

        page = st.radio(
            "Navigate",
            options = [
                "🏠 Overview",
                "🧪 Run Experiment",
                "📋 Experiment History",
                "🤖 ML Predictions",
                "📊 Strategies",
            ],
            label_visibility = "collapsed",
        )

        st.divider()
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption("Phase 7 — Streamlit Dashboard")

    dispatch = {
        "🏠 Overview":          page_overview,
        "🧪 Run Experiment":    page_run_experiment,
        "📋 Experiment History":page_experiment_history,
        "🤖 ML Predictions":    page_predictions,
        "📊 Strategies":        page_strategies,
    }
    dispatch[page]()


if __name__ == "__main__":
    main()
