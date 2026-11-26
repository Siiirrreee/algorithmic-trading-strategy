"""Is the backtest result real? Statistical tests for Sharpe ratios.

* Probabilistic Sharpe ratio (Bailey & Lopez de Prado, 2012): probability the true
  Sharpe exceeds a threshold, adjusting the standard error for skew and fat tails.
* Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): the same, but the threshold
  is the Sharpe you would expect from the *best of N* useless strategies, which
  corrects for having tried many parameter sets.
* Stationary bootstrap (Politis & Romano, 1994): resamples blocks of days, keeping
  volatility clustering and the strategy/benchmark pairing, to get a confidence
  interval for the Sharpe difference versus the benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TD = 252
EULER = 0.5772156649


def _excess(r: pd.Series, rf: pd.Series | None) -> np.ndarray:
    return (r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)).to_numpy()


def sharpe_se(x: np.ndarray) -> float:
    """Standard error of the per-period Sharpe ratio under non-normal returns."""
    sr = x.mean() / x.std(ddof=1)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x, fisher=False)
    return float(np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (len(x) - 1)))


def probabilistic_sharpe(r: pd.Series, rf: pd.Series | None = None, threshold_annual: float = 0.0) -> float:
    x = _excess(r, rf)
    sr = x.mean() / x.std(ddof=1)
    return float(stats.norm.cdf((sr - threshold_annual / np.sqrt(TD)) / sharpe_se(x)))


def expected_max_sharpe(trial_sharpes_annual: np.ndarray) -> float:
    """Expected maximum annual Sharpe among N trials if every true Sharpe were zero."""
    n = len(trial_sharpes_annual)
    if n < 2:
        return 0.0
    sd = np.std(trial_sharpes_annual, ddof=1)
    return float(sd * ((1 - EULER) * stats.norm.ppf(1 - 1 / n) + EULER * stats.norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(r: pd.Series, trial_sharpes_annual: np.ndarray, rf: pd.Series | None = None) -> tuple[float, float]:
    """Return (deflated Sharpe probability, the multiple-testing Sharpe hurdle)."""
    hurdle = expected_max_sharpe(np.asarray(trial_sharpes_annual))
    return probabilistic_sharpe(r, rf, hurdle), hurdle


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    jumps = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_sharpe_diff(
    strat: pd.Series,
    bench: pd.Series,
    rf: pd.Series | None = None,
    n_boot: int = 2000,
    mean_block: float = 21,
    seed: int = 42,
) -> dict:
    """Bootstrap distribution of Sharpe(strategy) - Sharpe(benchmark), paired by day."""
    a, b = _excess(strat, rf), _excess(bench.reindex(strat.index), rf)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ix = stationary_bootstrap_indices(len(a), mean_block, rng)
        sa, sb = a[ix], b[ix]
        diffs[i] = (sa.mean() / sa.std() - sb.mean() / sb.std()) * np.sqrt(TD)
    obs = (a.mean() / a.std() - b.mean() / b.std()) * np.sqrt(TD)
    return {
        "observed": float(obs),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_le_zero": float((diffs <= 0).mean()),
        "draws": diffs,
    }


def t_stat_mean(r: pd.Series, rf: pd.Series | None = None) -> float:
    x = _excess(r, rf)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
