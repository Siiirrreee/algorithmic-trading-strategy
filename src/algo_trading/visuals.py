"""Matplotlib charts. Each entity keeps one fixed color across every chart."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

from .metrics import drawdown, rolling_sharpe

BENCH_COLOR = "#666666"
COLORS = {
    "SPY buy & hold": BENCH_COLOR,
    "Sector equal weight": "#a6a6a6",
    "SMA 50/200 trend": "#257de0",
    "Time-series momentum": "#f47230",
    "Sector momentum": "#00b894",
    "SMA walk-forward": "#6a4ce2",
}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _c(name):
    return COLORS.get(name, "#257de0")


def equity_curves(series: dict[str, pd.Series], path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in series.items():
        eq = (1 + r).cumprod()
        lw = 1.5 if "SPY" in name or "equal" in name else 2
        ax.plot(eq.index, eq, color=_c(name), lw=lw, label=name)
        ax.annotate(f"{eq.iloc[-1]:.1f}×", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Growth of $1 (log scale), net of costs")
    ax.set_title("Growth of $1: strategies vs. buy-and-hold")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def drawdowns(series: dict[str, pd.Series], path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, r in series.items():
        dd = drawdown(r) * 100
        ax.plot(dd.index, dd, color=_c(name), lw=1.5, label=f"{name} (max {dd.min():.0f}%)")
    ax.set_ylabel("Drawdown from peak (%)")
    ax.set_title("Drawdowns: how far below the previous high")
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def sma_heatmap(table: pd.DataFrame, split: str, path):
    fast = sorted(table["fast"].unique())
    slow = sorted(table["slow"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    vals = table[["IS Sharpe", "OOS Sharpe"]].to_numpy()
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    for ax, col, title in zip(axes, ["IS Sharpe", "OOS Sharpe"],
                              [f"In-sample (to {split[:4]})", f"Out-of-sample ({split[:4]}–)"]):
        grid = np.full((len(fast), len(slow)), np.nan)
        for _, row in table.iterrows():
            grid[fast.index(row["fast"]), slow.index(row["slow"])] = row[col]
        im = ax.imshow(grid, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(fast)):
            for j in range(len(slow)):
                if not np.isnan(grid[i, j]):
                    v = grid[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if v > (vmin + vmax) / 2 else TEXT)
        ax.set_xticks(range(len(slow)), slow)
        ax.set_yticks(range(len(fast)), fast)
        ax.set_xlabel("Slow window (days)")
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel("Fast window (days)")
    fig.colorbar(im, ax=axes, label="Sharpe ratio", shrink=0.9)
    fig.suptitle("SMA crossover on SPY: does the best in-sample setting stay best?", x=0.06, ha="left",
                 fontweight="bold", color=TEXT, y=1.03)
    fig.savefig(path)
    plt.close(fig)


def cost_sensitivity(table: pd.DataFrame, names: list[str], path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = table["Cost per trade (bps)"]
    for n in names:
        ax.plot(x, table[f"{n} Sharpe"], marker="o", ms=6, color=_c(n), label=n)
    ax.set_xlabel("Cost per unit traded (basis points)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sensitivity to trading costs")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def rolling_sharpes(series: dict[str, pd.Series], rf, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, r in series.items():
        rs = rolling_sharpe(r, rf)
        ax.plot(rs.index, rs, color=_c(name), lw=1.5, label=name)
    ax.axhline(0, color=TEXT2, lw=1)
    ax.set_ylabel("Sharpe ratio, trailing 3 years")
    ax.set_title("Rolling 3-year Sharpe ratio: edges come and go")
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(path)
    plt.close(fig)


def bootstrap_hist(draws: dict[str, np.ndarray], path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(min(d.min() for d in draws.values()), max(d.max() for d in draws.values()), 60)
    for name, d in draws.items():
        ax.hist(d, bins=bins, alpha=0.55, color=_c(name), label=name, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=TEXT, lw=1.2)
    ax.set_xlabel("Sharpe ratio minus SPY buy-and-hold Sharpe (bootstrap draws)")
    ax.set_ylabel("Count")
    ax.set_title("Is the Sharpe improvement distinguishable from luck?")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
