"""Performance and risk metrics on daily returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult

TD = 252


def drawdown(returns: pd.Series) -> pd.Series:
    eq = (1 + returns).cumprod()
    return eq / eq.cummax() - 1


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch, in trading days, spent below a previous equity peak."""
    under = (drawdown(returns) < 0).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def cagr(returns: pd.Series) -> float:
    years = len(returns) / TD
    return float((1 + returns).prod() ** (1 / years) - 1)


def sharpe(returns: pd.Series, rf: pd.Series | None = None) -> float:
    ex = returns - (rf.reindex(returns.index).fillna(0) if rf is not None else 0)
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(TD)) if sd > 0 else 0.0


def sortino(returns: pd.Series, rf: pd.Series | None = None) -> float:
    ex = returns - (rf.reindex(returns.index).fillna(0) if rf is not None else 0)
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(TD)) if downside > 0 else 0.0


def alpha_beta(returns: pd.Series, bench: pd.Series, rf: pd.Series | None = None) -> tuple[float, float]:
    """CAPM beta and annualised alpha against the benchmark, on excess returns."""
    r = rf.reindex(returns.index).fillna(0) if rf is not None else 0
    y, x = returns - r, bench.reindex(returns.index) - r
    beta = float(np.cov(y, x, ddof=1)[0, 1] / x.var())
    alpha = float((y.mean() - beta * x.mean()) * TD)
    return alpha, beta


def performance(
    res: BacktestResult | pd.Series,
    rf: pd.Series | None = None,
    bench: pd.Series | None = None,
) -> dict:
    r = res.returns if isinstance(res, BacktestResult) else res
    dd = drawdown(r)
    out = {
        "CAGR": cagr(r),
        "Volatility": float(r.std() * np.sqrt(TD)),
        "Sharpe": sharpe(r, rf),
        "Sortino": sortino(r, rf),
        "Max drawdown": float(dd.min()),
        "Calmar": cagr(r) / abs(dd.min()) if dd.min() < 0 else np.nan,
        "Longest drawdown (days)": max_drawdown_duration(r),
        "Best year": np.nan,
        "Worst year": np.nan,
        "Daily hit rate": float((r > 0).mean()),
    }
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    out["Best year"], out["Worst year"] = float(yearly.max()), float(yearly.min())
    if isinstance(res, BacktestResult):
        years = len(r) / TD
        out["Turnover / yr"] = float(res.turnover.sum() / years)
        out["Cost drag / yr"] = float(res.costs.sum() / years)
        out["Time invested"] = float((res.weights.sum(axis=1) > 1e-9).mean())
        out["Trades"] = int((res.turnover > 1e-9).sum())
    if bench is not None:
        a, b = alpha_beta(r, bench, rf)
        out["Beta"], out["Alpha / yr"] = b, a
    return out


def annual_returns(series: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in series.items()})


def rolling_sharpe(returns: pd.Series, rf: pd.Series | None = None, window: int = 3 * TD) -> pd.Series:
    ex = returns - (rf.reindex(returns.index).fillna(0) if rf is not None else 0)
    return ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(TD)
