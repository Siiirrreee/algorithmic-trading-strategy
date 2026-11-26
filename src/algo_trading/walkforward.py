"""Parameter grids, in-sample/out-of-sample splits and walk-forward selection.

Choosing the parameter set with the best full-history backtest is itself a form of
look-ahead. Two defences:

* Holdout split: rank parameters on the in-sample period only and report how the
  chosen set does on the later, untouched period.
* Walk-forward: each January, pick the parameters with the best Sharpe over the
  trailing window and trade them for the next 12 months only. The stitched
  out-of-sample record is what a researcher could actually have achieved.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import strategies as S
from .backtest import Costs, run_backtest
from .metrics import sharpe


def sma_grid(fast=(10, 20, 50, 100), slow=(100, 150, 200, 250)) -> list[tuple[int, int]]:
    return [(f, s) for f, s in itertools.product(fast, slow) if f < s]


def xs_grid(lookbacks_m=(3, 6, 9, 12), top_ns=(2, 3, 4)) -> list[tuple[int, int]]:
    return list(itertools.product(lookbacks_m, top_ns))


def grid_returns(prices, rf, costs, lag, kind: str, grid, benchmark="SPY", sectors=None):
    """Net daily returns for every parameter set (each one is a causal backtest)."""
    out, targets = {}, {}
    for p in grid:
        if kind == "sma":
            tg = S.sma_crossover(prices[benchmark], *p)
        elif kind == "xs":
            tg = S.xs_momentum(prices[sectors], lookback=p[0] * S.MONTH, top_n=p[1])
        else:
            raise ValueError(kind)
        targets[p] = tg
        out[p] = run_backtest(prices, tg, rf, costs, lag).returns
    return pd.DataFrame(out), targets


def split_table(rets: pd.DataFrame, rf, split_date: str, names=("fast", "slow")) -> pd.DataFrame:
    """In-sample vs out-of-sample Sharpe for every parameter set."""
    rows = []
    for p in rets.columns:
        r = rets[p]
        first = r.ne(0).idxmax()  # skip the warm-up period with no position
        r = r.loc[first:]
        rows.append({names[0]: p[0], names[1]: p[1],
                     "IS Sharpe": sharpe(r.loc[:split_date], rf),
                     "OOS Sharpe": sharpe(r.loc[split_date:].iloc[1:], rf)})
    df = pd.DataFrame(rows)
    df["IS rank"] = df["IS Sharpe"].rank(ascending=False).astype(int)
    df["OOS rank"] = df["OOS Sharpe"].rank(ascending=False).astype(int)
    return df.sort_values("IS rank").reset_index(drop=True)


def walk_forward(prices, rf, costs, lag, rets: pd.DataFrame, targets: dict, train_years: int = 5):
    """Re-select parameters every January on the trailing window; stitch the chosen targets."""
    years = sorted(set(rets.index.year))
    cols = next(iter(targets.values())).columns
    stitched = pd.DataFrame(np.nan, index=rets.index, columns=cols)
    choices = []
    for y in years:
        train = rets.loc[str(y - train_years): str(y - 1)]
        if train.empty or train.index[0].year > y - train_years:
            continue
        # A parameter set must have been fully warmed up across the training window.
        live = [p for p in rets.columns if targets[p].loc[train.index].notna().all(axis=1).mean() > 0.9]
        if not live:
            continue
        scores = {p: sharpe(train[p], rf) for p in live}
        best = max(scores, key=scores.get)
        choices.append({"year": y, "params": best, "train Sharpe": scores[best]})
        mask = stitched.index.year == y
        stitched.loc[mask] = targets[best].loc[mask].to_numpy()
    if not choices:
        raise ValueError("not enough history for walk-forward")
    start = f"{choices[0]['year']}-01-01"
    res = run_backtest(prices, stitched, rf, costs, lag)
    res_oos = res.returns.loc[start:]
    return res, res_oos, pd.DataFrame(choices)


def cost_sensitivity(prices, rf, targets: dict, lag: int, levels_bps=(0, 5, 10, 25, 50)) -> pd.DataFrame:
    """Sharpe and CAGR for each strategy as round-trip-independent cost per trade rises."""
    from .metrics import cagr

    rows = []
    for bps in levels_bps:
        c = Costs(commission_bps=bps / 2, slippage_bps=bps / 2)
        row = {"Cost per trade (bps)": bps}
        for name, tg in targets.items():
            r = run_backtest(prices, tg, rf, c, lag).returns
            row[f"{name} Sharpe"] = sharpe(r, rf)
            row[f"{name} CAGR"] = cagr(r)
        rows.append(row)
    return pd.DataFrame(rows)
