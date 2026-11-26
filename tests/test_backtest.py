import numpy as np
import pandas as pd
import pytest

from algo_trading import Costs, run_backtest, synthetic_data
from algo_trading import strategies as S

ZERO = Costs(0, 0)


@pytest.fixture(scope="module")
def data():
    return synthetic_data(n_days=252 * 6, seed=3)


def _strategies(px):
    return {
        "sma": S.sma_crossover(px["SPY"], 50, 200),
        "tsmom": S.ts_momentum(px["SPY"]),
        "xs": S.xs_momentum(px[["XLB", "XLE", "XLF", "XLK", "XLC"]], top_n=2),
    }


def test_no_look_ahead(data):
    """Changing prices after a cut-off date must not change anything before it."""
    px, rf = data
    cut = px.index[900]
    shocked = px.copy()
    rng = np.random.default_rng(0)
    shocked.loc[shocked.index > cut] *= rng.uniform(0.5, 1.5, size=shocked.loc[shocked.index > cut].shape)
    base, alt = _strategies(px), _strategies(shocked)
    for name in base:
        pd.testing.assert_frame_equal(base[name].loc[:cut], alt[name].loc[:cut], check_names=False)
        r1 = run_backtest(px, base[name], rf).returns.loc[:cut]
        r2 = run_backtest(shocked, alt[name], rf).returns.loc[:cut]
        pd.testing.assert_series_equal(r1, r2)


def test_execution_lag_delays_returns():
    idx = pd.bdate_range("2020-01-01", periods=6)
    px = pd.DataFrame({"A": [100, 100, 100, 110, 121, 121.0]}, index=idx)
    tg = pd.DataFrame({"A": [np.nan, 1, np.nan, np.nan, np.nan, np.nan]}, index=idx)
    r = run_backtest(px, tg, costs=ZERO, lag=1).returns
    # Signal at day 1 close -> traded day 2 close -> first return earned on day 3 (+10%).
    assert r.iloc[:3].abs().sum() == 0
    assert r.iloc[3] == pytest.approx(0.10)
    r2 = run_backtest(px, tg, costs=ZERO, lag=2).returns
    assert r2.iloc[3] == 0 and r2.iloc[4] == pytest.approx(0.10)


def test_lag_zero_is_rejected(data):
    px, rf = data
    with pytest.raises(ValueError):
        run_backtest(px, S.buy_and_hold(px["SPY"]), rf, lag=0)


def test_buy_and_hold_matches_asset(data):
    px, rf = data
    r = run_backtest(px, S.buy_and_hold(px["SPY"]), rf, ZERO, lag=1).returns
    asset = px["SPY"].pct_change()
    np.testing.assert_allclose(r.iloc[2:], asset.iloc[2:], atol=1e-12)


def test_matches_loop_reference(data):
    """A constant 60/40 asset/cash mix rebalanced daily, checked against a plain formula."""
    px, rf = data
    tg = pd.DataFrame({"SPY": 0.6}, index=px.index)
    r = run_backtest(px, tg, rf, ZERO, lag=1).returns
    expected = 0.6 * px["SPY"].pct_change() + 0.4 * rf
    np.testing.assert_allclose(r.iloc[2:], expected.iloc[2:], atol=1e-12)


def test_round_trip_cost():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({"A": 100.0}, index=idx)
    tg = pd.DataFrame({"A": [1, np.nan, 0, np.nan, np.nan]}, index=idx)
    res = run_backtest(px, tg, costs=Costs(5, 5), lag=1)
    assert res.turnover.sum() == pytest.approx(2.0)
    assert res.equity.iloc[-1] == pytest.approx((1 - 0.001) ** 2)


def test_cannot_hold_unlisted_asset(data):
    px, rf = data
    tg = pd.DataFrame({"XLC": 1.0}, index=px.index)
    res = run_backtest(px, tg, rf, ZERO)
    launch = px["XLC"].first_valid_index()
    assert res.weights.loc[res.weights.index < launch, "XLC"].abs().sum() == 0
    assert res.weights["XLC"].iloc[-1] == 1.0


def test_leverage_rejected(data):
    px, rf = data
    tg = pd.DataFrame({"SPY": 0.8, "XLB": 0.8}, index=px.index)
    with pytest.raises(ValueError):
        run_backtest(px, tg, rf)


def test_weights_drift_between_rebalances():
    idx = pd.bdate_range("2020-01-01", periods=4)
    px = pd.DataFrame({"A": [100, 100, 200, 200.0], "B": [100, 100, 100, 100.0]}, index=idx)
    tg = pd.DataFrame({"A": [0.5, np.nan, np.nan, np.nan], "B": [0.5, np.nan, np.nan, np.nan]}, index=idx)
    w = run_backtest(px, tg, costs=ZERO).weights
    assert w.loc[idx[2], "A"] == pytest.approx(2 / 3)
