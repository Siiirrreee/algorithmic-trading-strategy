"""Trading signals.

Every function returns *target weights decided at the close of each date*, using
only prices up to and including that date. Rows that are all NaN mean "no
rebalance today". The backtester applies an execution lag, so a signal formed
at Monday's close is traded at Tuesday's close at the earliest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH = 21


def sma_crossover(price: pd.Series, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Long the asset when the fast moving average is above the slow one, else cash."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    f = price.rolling(fast, min_periods=fast).mean()
    s = price.rolling(slow, min_periods=slow).mean()
    w = (f > s).astype(float).where(s.notna())
    return w.to_frame(price.name)


def ts_momentum(price: pd.Series, lookback: int = 12 * MONTH, skip: int = MONTH) -> pd.DataFrame:
    """Time-series momentum: long when the 12-month return, skipping the latest month, is positive.

    Skipping the most recent month avoids the well-documented one-month reversal effect.
    """
    mom = price.shift(skip) / price.shift(lookback) - 1
    w = (mom > 0).astype(float).where(mom.notna())
    return w.to_frame(price.name)


def momentum_score(prices: pd.DataFrame, lookback: int = 12 * MONTH, skip: int = MONTH) -> pd.DataFrame:
    """Return from t-lookback to t-skip. NaN for assets without enough history (point-in-time)."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def xs_momentum(
    prices: pd.DataFrame,
    lookback: int = 12 * MONTH,
    skip: int = MONTH,
    top_n: int = 3,
    abs_filter: bool = True,
) -> pd.DataFrame:
    """Cross-sectional (relative) momentum across sectors, rebalanced at each month end.

    Hold the top_n assets by momentum score with equal weight 1/top_n. With the
    absolute filter on (dual momentum), a selected asset whose own score is
    negative is replaced by cash.
    """
    score = momentum_score(prices, lookback, skip)
    dates = month_ends(prices.index)
    w = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for d in dates:
        row = score.loc[d].dropna()
        if len(row) < top_n:
            continue
        picks = row.nlargest(top_n)
        if abs_filter:
            picks = picks[picks > 0]
        target = pd.Series(0.0, index=prices.columns)
        target[picks.index] = 1.0 / top_n
        w.loc[d] = target
    return w


def buy_and_hold(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    df = prices.to_frame() if isinstance(prices, pd.Series) else prices
    w = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    first = df.dropna().index[0]
    w.loc[first] = 1.0 / df.shape[1]
    return w


def equal_weight_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal weight across all assets that exist at each month end (the sector benchmark)."""
    w = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for d in month_ends(prices.index):
        live = prices.loc[d].notna()
        if live.sum() == 0:
            continue
        w.loc[d] = live.astype(float) / live.sum()
    return w
