"""Daily portfolio backtester with execution lag, drifting weights and trading costs.

Timeline for one day d (all at the close):
    1. Holdings chosen at the close of d-1 earn the asset returns of day d;
       uninvested capital earns the T-bill rate.
    2. Weights drift with those returns.
    3. If a target decided `lag` days earlier is due, trade to it. Cost is
       (commission + slippage) in basis points times the traded notional,
       charged on day d.

So with lag = 1 a signal formed with Monday's close is traded at Tuesday's close and
first earns Wednesday's return. Nothing about day d is known before day d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    commission_bps: float = 5.0  # fees and commissions per unit traded
    slippage_bps: float = 5.0    # half-spread plus market impact per unit traded

    @property
    def total(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    returns: pd.Series        # net daily returns
    gross_returns: pd.Series  # before trading costs
    weights: pd.DataFrame     # end-of-day weights after trading
    turnover: pd.Series       # one-way traded notional per day (fraction of equity)
    costs: pd.Series          # cost paid per day (fraction of equity)

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    rf: pd.Series | None = None,
    costs: Costs | None = None,
    lag: int = 1,
) -> BacktestResult:
    """Simulate a strategy. `targets` holds weights decided at each close (NaN rows = no trade)."""
    if lag < 1:
        raise ValueError("lag must be >= 1: a signal cannot trade on the close that formed it")
    costs = costs or Costs()
    cols = list(targets.columns)
    px = prices[cols].reindex(targets.index)
    rets = px.pct_change(fill_method=None).to_numpy()
    listed = px.notna().to_numpy()
    rets = np.where(np.isfinite(rets), rets, 0.0)
    rf_arr = (rf.reindex(targets.index).fillna(0.0).to_numpy() if rf is not None
              else np.zeros(len(targets)))

    # Shift targets forward by the execution lag.
    tgt = targets.shift(lag).to_numpy()
    n, k = tgt.shape
    w = np.zeros(k)
    net = np.zeros(n)
    gross = np.zeros(n)
    turn = np.zeros(n)
    cost = np.zeros(n)
    W = np.zeros((n, k))
    c = costs.total
    for d in range(n):
        r_assets = rets[d]
        g = float(w @ r_assets + (1.0 - w.sum()) * rf_arr[d])
        gross[d] = g
        w = w * (1 + r_assets) / (1 + g) if (1 + g) != 0 else w
        row = tgt[d]
        if not np.all(np.isnan(row)):
            new = np.nan_to_num(row, nan=0.0)
            new = np.where(listed[d], new, 0.0)  # cannot hold an asset that does not trade yet
            if new.sum() > 1 + 1e-9:
                raise ValueError("targets imply leverage (weights sum above 1)")
            t = float(np.abs(new - w).sum())
            turn[d] = t
            cost[d] = t * c
            w = new
        net[d] = (1 + g) * (1 - cost[d]) - 1
        W[d] = w

    idx = targets.index
    return BacktestResult(
        returns=pd.Series(net, idx, name="net"),
        gross_returns=pd.Series(gross, idx, name="gross"),
        weights=pd.DataFrame(W, idx, cols),
        turnover=pd.Series(turn, idx, name="turnover"),
        costs=pd.Series(cost, idx, name="costs"),
    )
