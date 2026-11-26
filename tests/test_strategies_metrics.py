import numpy as np
import pandas as pd
import pytest

from algo_trading import metrics as M
from algo_trading import stats_tests as T
from algo_trading import strategies as S
from algo_trading import synthetic_data


def test_sma_signal_logic():
    idx = pd.bdate_range("2020-01-01", periods=30)
    up = pd.Series(np.arange(30, dtype=float) + 1, index=idx, name="X")
    w = S.sma_crossover(up, 3, 10)
    assert w["X"].iloc[:9].isna().all()
    assert (w["X"].iloc[9:] == 1).all()
    w_down = S.sma_crossover(up[::-1].set_axis(idx), 3, 10)
    assert (w_down["X"].iloc[9:] == 0).all()
    with pytest.raises(ValueError):
        S.sma_crossover(up, 10, 3)


def test_momentum_skips_latest_month():
    idx = pd.bdate_range("2020-01-01", periods=300)
    p = pd.Series(100.0, index=idx)
    p.iloc[-10:] = 50.0  # a crash inside the skipped month must not flip the signal
    p.iloc[:100] = 90.0  # 12 months ago the price was lower, so momentum is positive
    w = S.ts_momentum(p.rename("X"))
    assert w["X"].iloc[-1] == 1.0


def test_xs_momentum_weights():
    px, _ = synthetic_data(n_days=252 * 4, seed=5)
    sectors = px[["XLB", "XLE", "XLF", "XLK", "XLC"]]
    w = S.xs_momentum(sectors, top_n=2, abs_filter=False).dropna(how="all")
    assert np.allclose(w.sum(axis=1), 1.0)
    assert ((w > 0).sum(axis=1) == 2).all()
    launch = px["XLC"].first_valid_index()
    # XLC is ineligible until it has 12 months of history.
    assert w.loc[w.index < launch + pd.Timedelta(days=365), "XLC"].sum() == 0
    w_abs = S.xs_momentum(sectors, top_n=2, abs_filter=True).dropna(how="all")
    assert (w_abs.sum(axis=1) <= 1 + 1e-12).all()


def test_drawdown_and_cagr():
    r = pd.Series([0.10, -0.50, 0.20, 0.0])
    dd = M.drawdown(r)
    assert dd.min() == pytest.approx(-0.5)
    assert M.max_drawdown_duration(r) == 3
    r2 = pd.Series([0.0004] * 252)
    assert M.cagr(r2) == pytest.approx(1.0004**252 - 1)


def test_sharpe_and_beta():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0.0004, 0.01, 5000))
    y = 0.5 * x + pd.Series(rng.normal(0, 0.002, 5000))
    a, b = M.alpha_beta(y, x)
    assert b == pytest.approx(0.5, abs=0.02)
    assert M.sharpe(x) == pytest.approx(x.mean() / x.std() * np.sqrt(252))


def test_psr_and_deflation():
    rng = np.random.default_rng(2)
    noise = pd.Series(rng.normal(0, 0.01, 5000))
    assert 0.02 < T.probabilistic_sharpe(noise) < 0.98
    good = pd.Series(rng.normal(0.001, 0.01, 5000))
    assert T.probabilistic_sharpe(good) > 0.99
    trials = rng.normal(0, 0.3, 50)
    dsr, hurdle = T.deflated_sharpe(good, trials)
    assert hurdle > 0 and dsr <= T.probabilistic_sharpe(good)


def test_bootstrap_identical_series():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0003, 0.01, 1000))
    out = T.bootstrap_sharpe_diff(r, r, n_boot=100)
    assert out["observed"] == pytest.approx(0) and out["ci_low"] == pytest.approx(0, abs=1e-12)
