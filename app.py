"""Interactive backtester. Run locally with `streamlit run app.py`, or deploy on Streamlit Community Cloud.

Reads the cached price data committed in data/, so the app does not depend on a live market-data API.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from algo_trading import metrics as M  # noqa: E402
from algo_trading import stats_tests as T  # noqa: E402
from algo_trading import strategies as S  # noqa: E402
from algo_trading.backtest import Costs, run_backtest  # noqa: E402
from algo_trading.data import BENCHMARK, SECTORS, load_data  # noqa: E402

st.set_page_config(page_title="Algorithmic Trading Strategy Backtester", layout="wide")


@st.cache_data
def data():
    px, rf, info = load_data(refresh=False)
    return px, rf, info


px, rf, info = data()
sectors = [s for s in SECTORS if s in px.columns]

st.title("Algorithmic Trading Strategy Backtester")
st.caption(
    f"Daily adjusted ETF prices {info['start']} to {info['end']}. Signals trade at a later close than the one "
    "that formed them, every trade pays commission and slippage, and cash earns the T-bill rate. "
    "Educational project, not investment advice."
)

with st.sidebar:
    st.header("Strategy")
    kind = st.radio("Type", ["SMA crossover (SPY)", "Time-series momentum (SPY)", "Sector momentum (11 sector ETFs)"])
    if kind.startswith("SMA"):
        fast = st.slider("Fast moving average (days)", 5, 150, 50, 5)
        slow = st.slider("Slow moving average (days)", 20, 300, 200, 10)
        if fast >= slow:
            st.error("The fast window must be shorter than the slow window.")
            st.stop()
        targets = S.sma_crossover(px[BENCHMARK], fast, slow)
        label = f"SMA {fast}/{slow}"
    elif kind.startswith("Time"):
        lb = st.slider("Lookback (months)", 1, 18, 12)
        skip = st.slider("Skip most recent (months)", 0, 3, 1)
        targets = S.ts_momentum(px[BENCHMARK], lb * S.MONTH, skip * S.MONTH)
        label = f"TS momentum {lb}-{skip}"
    else:
        lb = st.slider("Lookback (months)", 1, 18, 12)
        top = st.slider("Sectors held", 1, 6, 3)
        absf = st.checkbox("Absolute momentum filter (cash if negative)", True)
        targets = S.xs_momentum(px[sectors], lb * S.MONTH, S.MONTH, top, absf)
        label = f"Sector momentum top {top}"

    st.header("Execution")
    cost = st.slider("Cost per trade (bps of amount traded)", 0, 100, 10)
    lag = st.slider("Execution lag (days after signal)", 1, 5, 1)
    years = st.slider("Period", int(px.index[0].year) + 1, int(px.index[-1].year),
                      (int(px.index[0].year) + 1, int(px.index[-1].year)))

res = run_backtest(px, targets, rf, Costs(cost / 2, cost / 2), lag)
bench = run_backtest(px, S.buy_and_hold(px[[BENCHMARK]]), rf, Costs(cost / 2, cost / 2), lag)
start = max(pd.Timestamp(f"{years[0]}-01-01"), targets.dropna(how="all").index[0])
end = pd.Timestamp(f"{years[1]}-12-31")
r, b = res.returns.loc[start:end], bench.returns.loc[start:end]

p, pb = M.performance(r, rf, b), M.performance(b, rf, b)
c = st.columns(5)
c[0].metric("CAGR", f"{p['CAGR']:.1%}", f"{p['CAGR'] - pb['CAGR']:+.1%} vs SPY")
c[1].metric("Sharpe", f"{p['Sharpe']:.2f}", f"{p['Sharpe'] - pb['Sharpe']:+.2f} vs SPY")
c[2].metric("Max drawdown", f"{p['Max drawdown']:.0%}", f"{p['Max drawdown'] - pb['Max drawdown']:+.0%} vs SPY")
c[3].metric("Volatility", f"{p['Volatility']:.1%}")
yrs = len(r) / 252
c[4].metric("Turnover / yr", f"{res.turnover.loc[start:end].sum() / yrs:.1f}×")

st.subheader("Growth of $1, net of costs")
eq = pd.DataFrame({label: (1 + r).cumprod(), "SPY buy & hold": (1 + b).cumprod()})
st.line_chart(eq, color=["#257de0", "#666666"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Drawdown from peak")
    st.line_chart(pd.DataFrame({label: M.drawdown(r), "SPY buy & hold": M.drawdown(b)}),
                  color=["#257de0", "#666666"])
with col2:
    st.subheader("Share of capital invested")
    st.area_chart(res.weights.loc[start:end].sum(axis=1).rename("Invested"), color="#257de0")

st.subheader("Is the result distinguishable from luck?")
bs = T.bootstrap_sharpe_diff(r, b, rf, n_boot=500)
c = st.columns(3)
c[0].metric("Probabilistic Sharpe (SR > 0)", f"{T.probabilistic_sharpe(r, rf):.0%}")
c[1].metric("Sharpe − SPY, 95% bootstrap CI", f"{bs['ci_low']:+.2f} to {bs['ci_high']:+.2f}")
c[2].metric("Bootstrap draws with no improvement", f"{bs['p_le_zero']:.0%}")
st.caption("Every slider you move is another trial. The more settings you try, the higher the bar a good-looking "
           "Sharpe has to clear (see the deflated Sharpe ratio in the README).")

st.subheader("Annual returns")
ann = M.annual_returns({label: r, "SPY buy & hold": b})
st.dataframe(ann.style.format("{:.1%}"), width="stretch")
