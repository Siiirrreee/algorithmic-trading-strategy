"""Algorithmic trading strategy research: signals, a cost-aware backtester and statistical tests."""
from .backtest import BacktestResult, Costs, run_backtest
from .data import BENCHMARK, SECTORS, UNIVERSE, load_data, synthetic_data
from .metrics import drawdown, performance, rolling_sharpe, sharpe
from .stats_tests import bootstrap_sharpe_diff, deflated_sharpe, probabilistic_sharpe
from .strategies import buy_and_hold, equal_weight_monthly, sma_crossover, ts_momentum, xs_momentum

__all__ = [
    "BacktestResult", "Costs", "run_backtest", "BENCHMARK", "SECTORS", "UNIVERSE", "load_data",
    "synthetic_data", "drawdown", "performance", "rolling_sharpe", "sharpe", "bootstrap_sharpe_diff",
    "deflated_sharpe", "probabilistic_sharpe", "buy_and_hold", "equal_weight_monthly", "sma_crossover",
    "ts_momentum", "xs_momentum",
]
