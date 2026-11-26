"""Run the full research pipeline: backtests, robustness checks, statistical tests, charts, README.

    python main.py                 # use cached data in data/ (downloads if missing)
    python main.py --refresh-data  # re-download prices and T-bill rates
    python main.py --synthetic     # offline run on simulated prices (smoke test)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from algo_trading import metrics as M  # noqa: E402
from algo_trading import stats_tests as T  # noqa: E402
from algo_trading import strategies as S  # noqa: E402
from algo_trading import visuals as V  # noqa: E402
from algo_trading import walkforward as W  # noqa: E402
from algo_trading.backtest import Costs, run_backtest  # noqa: E402
from algo_trading.data import BENCHMARK, SECTORS, load_data, synthetic_data  # noqa: E402

RESULTS, VISUALS = ROOT / "results", ROOT / "visuals"
SPLIT = "2013-01-01"

ACTIVE = ["SMA 50/200 trend", "Time-series momentum", "Sector momentum"]
BENCHES = ["SPY buy & hold", "Sector equal weight"]


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def first_signal(tg: pd.DataFrame, lag: int) -> pd.Timestamp:
    i = tg.index.get_loc(tg.dropna(how="all").index[0])
    return tg.index[min(i + lag + 1, len(tg) - 1)]


def perf_table(rets: dict[str, pd.Series], res: dict, rf, bench) -> pd.DataFrame:
    rows = {}
    for name, r in rets.items():
        p = M.performance(r, rf, bench.reindex(r.index))
        if name in res:  # trading stats over the same window
            b = res[name]
            yrs = len(r) / 252
            p["Turnover / yr"] = float(b.turnover.reindex(r.index).sum() / yrs)
            p["Cost drag / yr"] = float(b.costs.reindex(r.index).sum() / yrs)
            p["Time invested"] = float((b.weights.reindex(r.index).sum(axis=1) > 1e-9).mean())
        rows[name] = p
    return pd.DataFrame(rows).T


def fmt_perf(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in ["CAGR", "Volatility"]:
        out[c] = df[c].map(pct)
    out["Sharpe"] = df["Sharpe"].map(num)
    out["Sortino"] = df["Sortino"].map(num)
    out["Max drawdown"] = df["Max drawdown"].map(lambda v: pct(v, 0))
    out["Calmar"] = df["Calmar"].map(num)
    out["Beta"] = df["Beta"].map(num)
    out["Alpha / yr"] = df["Alpha / yr"].map(pct)
    if "Turnover / yr" in df:
        out["Turnover / yr"] = df["Turnover / yr"].map(lambda v: "–" if pd.isna(v) else f"{v:.1f}×")
        out["Cost drag / yr"] = df["Cost drag / yr"].map(lambda v: pct(v, 2))
        out["Time invested"] = df["Time invested"].map(lambda v: pct(v, 0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="commission + slippage per unit traded")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-readme", action="store_true")
    a = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    VISUALS.mkdir(exist_ok=True)
    if a.synthetic:
        px, rf = synthetic_data(n_days=252 * 20, seed=1)
        info = {"source": "synthetic GBM", "start": str(px.index[0].date()), "end": str(px.index[-1].date()),
                "days": len(px)}
    else:
        px, rf, info = load_data(refresh=a.refresh_data)
    split = SPLIT if not a.synthetic else str(px.index[len(px) * 2 // 3].date())
    costs = Costs(a.cost_bps / 2, a.cost_bps / 2)
    lag = a.lag
    sectors = [s for s in SECTORS if s in px.columns]
    print(f"Data: {info['source']}, {info['start']} to {info['end']} ({info['days']} days)")

    # ---------------------------------------------------------------- strategies
    targets = {
        "SPY buy & hold": S.buy_and_hold(px[[BENCHMARK]]),
        "Sector equal weight": S.equal_weight_monthly(px[sectors]),
        "SMA 50/200 trend": S.sma_crossover(px[BENCHMARK], 50, 200),
        "Time-series momentum": S.ts_momentum(px[BENCHMARK]),
        "Sector momentum": S.xs_momentum(px[sectors], top_n=3, abs_filter=True),
    }
    res = {k: run_backtest(px, tg, rf, costs, lag) for k, tg in targets.items()}
    start = max(first_signal(targets[k], lag) for k in ACTIVE)
    rets = {k: v.returns.loc[start:] for k, v in res.items()}
    spy = rets["SPY buy & hold"]

    perf = perf_table(rets, res, rf, spy)
    perf_is = perf_table({k: r.loc[:split] for k, r in rets.items()}, res, rf, spy)
    perf_oos = perf_table({k: r.loc[split:].iloc[1:] for k, r in rets.items()}, res, rf, spy)

    # ---------------------------------------------------------------- robustness
    sma_rets, sma_tg = W.grid_returns(px, rf, costs, lag, "sma", W.sma_grid())
    sma_split = W.split_table(sma_rets.loc[start:], rf, split)
    xs_rets, xs_tg = W.grid_returns(px, rf, costs, lag, "xs", W.xs_grid(), sectors=sectors)
    xs_split = W.split_table(xs_rets.loc[start:], rf, split, names=("lookback (months)", "top N"))
    tsm_trials = {m: run_backtest(px, S.ts_momentum(px[BENCHMARK], m * S.MONTH), rf, costs, lag).returns.loc[start:]
                  for m in (3, 6, 9, 12)}

    wf_res, wf_oos, wf_choices = W.walk_forward(px, rf, costs, lag, sma_rets, sma_tg)
    wf_choices["params"] = wf_choices["params"].map(lambda p: f"{p[0]}/{p[1]}")
    wf_start = wf_oos.index[0]
    wf_cmp = {
        "SMA walk-forward": wf_oos,
        "SMA 50/200 trend": res["SMA 50/200 trend"].returns.loc[wf_start:],
        "SPY buy & hold": res["SPY buy & hold"].returns.loc[wf_start:],
    }
    wf_perf = perf_table(wf_cmp, {"SMA walk-forward": wf_res, "SMA 50/200 trend": res["SMA 50/200 trend"]},
                         rf, wf_cmp["SPY buy & hold"])

    cost_tbl = W.cost_sensitivity(px.loc[:], rf, {k: targets[k] for k in ACTIVE}, lag)

    # ---------------------------------------------------------------- statistics
    trials = {
        "SMA 50/200 trend": [M.sharpe(sma_rets[c].loc[start:], rf) for c in sma_rets.columns],
        "Time-series momentum": [M.sharpe(r, rf) for r in tsm_trials.values()],
        "Sector momentum": [M.sharpe(xs_rets[c].loc[start:], rf) for c in xs_rets.columns],
    }
    stat_rows, draws = {}, {}
    for k in ACTIVE:
        bs = T.bootstrap_sharpe_diff(rets[k], spy, rf, n_boot=a.boot)
        dsr, hurdle = T.deflated_sharpe(rets[k], np.array(trials[k]), rf)
        draws[k] = bs.pop("draws")
        stat_rows[k] = {
            "Sharpe": M.sharpe(rets[k], rf),
            "t-stat (excess return)": T.t_stat_mean(rets[k], rf),
            "PSR (SR > 0)": T.probabilistic_sharpe(rets[k], rf),
            "Trials": len(trials[k]),
            "Hurdle Sharpe": hurdle,
            "Deflated Sharpe": dsr,
            "Sharpe − SPY": bs["observed"],
            "95% CI": f"{bs['ci_low']:+.2f} to {bs['ci_high']:+.2f}",
            "P(boot ≤ 0)": bs["p_le_zero"],
        }
    bs_ew = T.bootstrap_sharpe_diff(rets["Sector momentum"], rets["Sector equal weight"], rf, n_boot=a.boot)
    stats_df = pd.DataFrame(stat_rows).T

    # ---------------------------------------------------------------- outputs
    perf.to_csv(RESULTS / "performance.csv")
    perf_is.to_csv(RESULTS / "performance_in_sample.csv")
    perf_oos.to_csv(RESULTS / "performance_out_of_sample.csv")
    sma_split.to_csv(RESULTS / "sma_grid.csv", index=False)
    xs_split.to_csv(RESULTS / "sector_momentum_grid.csv", index=False)
    wf_choices.to_csv(RESULTS / "walk_forward_choices.csv", index=False)
    wf_perf.to_csv(RESULTS / "walk_forward_performance.csv")
    cost_tbl.to_csv(RESULTS / "cost_sensitivity.csv", index=False)
    stats_df.to_csv(RESULTS / "statistical_tests.csv")
    M.annual_returns(rets).to_csv(RESULTS / "annual_returns.csv")

    main_series = {k: rets[k] for k in ["SPY buy & hold"] + ACTIVE}
    V.equity_curves(main_series, VISUALS / "equity_curves.png")
    V.drawdowns({k: rets[k] for k in ["SPY buy & hold", "SMA 50/200 trend", "Sector momentum"]},
                VISUALS / "drawdowns.png")
    V.sma_heatmap(sma_split, split, VISUALS / "sma_heatmap.png")
    V.cost_sensitivity(cost_tbl, ACTIVE, VISUALS / "cost_sensitivity.png")
    V.rolling_sharpes(main_series, rf, VISUALS / "rolling_sharpe.png")
    V.bootstrap_hist(draws, VISUALS / "bootstrap.png")

    # ---------------------------------------------------------------- write-up
    best_is = sma_split.iloc[0]
    xs_best = xs_split.iloc[0]
    rank_corr = sma_split[["IS Sharpe", "OOS Sharpe"]].corr(method="spearman").iloc[0, 1]
    s = {k: perf.loc[k].to_dict() for k in perf.index}
    so = {k: perf_oos.loc[k].to_dict() for k in perf_oos.index}
    st = stats_df.to_dict(orient="index")
    be = (cost_tbl.set_index("Cost per trade (bps)"))
    stats = {
        "data": info, "evaluation_start": str(start.date()), "split": split, "cost_bps": a.cost_bps, "lag": lag,
        "performance": s, "out_of_sample": so, "tests": {k: {kk: (vv if not isinstance(vv, np.generic) else vv.item())
                                                           for kk, vv in v.items()} for k, v in st.items()},
        "sector_mom_vs_equal_weight": {k: v for k, v in bs_ew.items() if k != "draws"},
        "sma_best_in_sample": {"fast": int(best_is["fast"]), "slow": int(best_is["slow"]),
                               "IS": float(best_is["IS Sharpe"]), "OOS": float(best_is["OOS Sharpe"]),
                               "OOS rank": int(best_is["OOS rank"])},
        "sma_rank_correlation": float(rank_corr),
        "walk_forward_start": str(wf_start.date()),
    }
    (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2, default=float))

    findings = build_findings(s, so, st, bs_ew, best_is, xs_best, rank_corr, sma_split, wf_perf, be, a.cost_bps,
                              split, len(sma_split))
    md = build_results_md(info, start, split, a.cost_bps, lag, perf, perf_oos, stats_df, sma_split, wf_perf,
                          wf_choices, cost_tbl, findings, wf_start)
    (RESULTS / "summary.md").write_text("# Results\n\n" + md)
    if not a.no_readme and not a.synthetic:
        update_readme(md)
    print(fmt_perf(perf).to_string())
    print("\n".join(findings))


def build_findings(s, so, st, bs_ew, best_is, xs_best, rank_corr, sma_split, wf_perf, be, cost_bps, split, n_grid):
    spy, sma, tsm, xs = (s[k] for k in ["SPY buy & hold"] + ACTIVE)
    out = []
    best_sr = max(ACTIVE, key=lambda k: s[k]["Sharpe"])
    out.append(
        f"- Over the full period, SPY buy-and-hold returned **{pct(spy['CAGR'])} a year** with a Sharpe of "
        f"{num(spy['Sharpe'])} and a **{pct(spy['Max drawdown'], 0)}** worst drawdown. "
        f"The best risk-adjusted strategy was **{best_sr}** (Sharpe {num(s[best_sr]['Sharpe'])}, "
        f"CAGR {pct(s[best_sr]['CAGR'])}, max drawdown {pct(s[best_sr]['Max drawdown'], 0)})."
    )
    trend_dd = [k for k in ["SMA 50/200 trend", "Time-series momentum"]
                if s[k]["Max drawdown"] > spy["Max drawdown"] + 0.05]
    if trend_dd:
        out.append(
            f"- The trend filters' clearest effect is on **risk**: {' and '.join(trend_dd)} cut the worst "
            f"drawdown from {pct(spy['Max drawdown'], 0)} to "
            f"{' / '.join(pct(s[k]['Max drawdown'], 0) for k in trend_dd)} by moving to T-bills in long declines, "
            f"while staying invested {pct(sma['Time invested'], 0)} / {pct(tsm['Time invested'], 0)} of the time. "
            f"Beta falls to {num(sma['Beta'])} / {num(tsm['Beta'])}."
        )
    beat = [k for k in ACTIVE if s[k]["CAGR"] > spy["CAGR"]]
    if not beat:
        out.append("- None of the strategies beat buy-and-hold on raw return after costs. Sitting in T-bills during "
                   "sharp V-shaped recoveries (whipsaws) is the price of avoiding long bear markets.")
    elif len(beat) < len(ACTIVE):
        out.append(f"- Only {', '.join(beat)} beat buy-and-hold on raw return ("
                   + ", ".join(f"{pct(s[k]['CAGR'])}" for k in beat) + f" vs {pct(spy['CAGR'])}).")
    else:
        out.append("- All three strategies beat buy-and-hold on raw return after costs over this window.")
    sig = [k for k in ACTIVE if st[k]["P(boot ≤ 0)"] < 0.05]
    out.append(
        "- **Statistical significance.** "
        + (f"{', '.join(sig)} improved Sharpe over SPY in more than 95% of bootstrap resamples. " if sig else
           "No strategy's Sharpe improvement over SPY is significant at 5%: ")
        + "; ".join(f"{k} {st[k]['Sharpe − SPY']:+.2f} (95% CI {st[k]['95% CI']})" for k in ACTIVE)
        + "."
        + ("" if len(sig) == len(ACTIVE) else
           " With about two decades of daily data, Sharpe differences of this size are hard to tell apart from luck.")
    )
    out.append(
        "- **Multiple-testing correction.** The deflated Sharpe ratio asks whether the result beats the best of all "
        "parameter sets tried, not zero: "
        + "; ".join(f"{k} {pct(st[k]['Deflated Sharpe'], 0)} (vs {pct(st[k]['PSR (SR > 0)'], 0)} unadjusted, "
                    f"{int(st[k]['Trials'])} trials)" for k in ACTIVE)
        + ". A deflated value below 95% means the edge does not survive the search."
    )
    out.append(
        f"- **Overfitting check.** On the {n_grid}-setting SMA grid, the best in-sample choice "
        f"({int(best_is['fast'])}/{int(best_is['slow'])}, Sharpe {num(best_is['IS Sharpe'])}) ranked "
        f"**#{int(best_is['OOS rank'])} of {n_grid}** after {split[:4]} (Sharpe {num(best_is['OOS Sharpe'])}). "
        f"The rank correlation between in-sample and out-of-sample Sharpe is {num(rank_corr)}: "
        + ("past rankings carried some information." if rank_corr > 0.3 else
           "in-sample rankings told you little about what would work next.")
    )
    wf, fixed = wf_perf.loc["SMA walk-forward"], wf_perf.loc["SMA 50/200 trend"]
    out.append(
        f"- **Walk-forward.** Re-optimising the SMA windows every January on the trailing five years earned a Sharpe of "
        f"{num(wf['Sharpe'])} out of sample, {'above' if wf['Sharpe'] > fixed['Sharpe'] else 'below'} the "
        f"fixed textbook 50/200 rule ({num(fixed['Sharpe'])}) over the same years. "
        + ("Optimisation added nothing reliable over a rule fixed in advance." if wf["Sharpe"] <= fixed["Sharpe"] + 0.05
           else "Adapting the windows helped over this period.")
    )
    xs_ew = s["Sector equal weight"]
    out.append(
        f"- **Sector momentum vs. its own benchmark.** Against an equal-weight portfolio of the same sectors "
        f"(Sharpe {num(xs_ew['Sharpe'])}), sector momentum's Sharpe difference is {bs_ew['observed']:+.2f} "
        f"(95% CI {bs_ew['ci_low']:+.2f} to {bs_ew['ci_high']:+.2f}). It trades {num(xs['Turnover / yr'], 1)}× its "
        f"capital a year, costing {pct(xs['Cost drag / yr'], 2)} annually at {cost_bps:.0f} bps."
    )
    hi = be.index.max()
    drops = {k: be.loc[0, f"{k} Sharpe"] - be.loc[hi, f"{k} Sharpe"] for k in ACTIVE}
    worst = max(drops, key=drops.get)
    out.append(
        f"- **Costs.** Moving from 0 to {hi:.0f} bps per trade lowers Sharpe by "
        + "; ".join(f"{k} {v:.2f}" for k, v in drops.items())
        + f". {worst} is the most cost-sensitive: it trades {num(s[worst]['Turnover / yr'], 1)}× its capital a year."
    )
    out.append(
        f"- **After {split[:4]}** (out of sample): SPY Sharpe {num(so['SPY buy & hold']['Sharpe'])}, "
        + ", ".join(f"{k} {num(so[k]['Sharpe'])}" for k in ACTIVE)
        + "."
        + (" Every strategy lagged buy-and-hold on a risk-adjusted basis in the untouched period."
           if all(so[k]["Sharpe"] < so["SPY buy & hold"]["Sharpe"] for k in ACTIVE) else "")
    )
    return out


def md_table(df, index=True):
    return df.to_markdown(index=index)


def build_results_md(info, start, split, cost_bps, lag, perf, perf_oos, stats_df, sma_split, wf_perf, wf_choices,
                     cost_tbl, findings, wf_start):
    st = stats_df.copy()
    for c in ["Sharpe", "t-stat (excess return)", "Hurdle Sharpe", "Sharpe − SPY"]:
        st[c] = st[c].map(lambda v: f"{v:+.2f}" if c == "Sharpe − SPY" else f"{v:.2f}")
    for c in ["PSR (SR > 0)", "Deflated Sharpe", "P(boot ≤ 0)"]:
        st[c] = st[c].map(lambda v: pct(v, 0))
    st["Trials"] = st["Trials"].astype(int)
    ct = cost_tbl.copy()
    for c in ct.columns[1:]:
        ct[c] = ct[c].map(num if "Sharpe" in c else pct)
    ct["Cost per trade (bps)"] = ct["Cost per trade (bps)"].astype(int)
    ct = ct[["Cost per trade (bps)"] + [f"{k} Sharpe" for k in ACTIVE] + [f"{k} CAGR" for k in ACTIVE]]
    grid = sma_split.copy()
    grid[["IS Sharpe", "OOS Sharpe"]] = grid[["IS Sharpe", "OOS Sharpe"]].map(num)
    oos = fmt_perf(perf_oos)[["CAGR", "Volatility", "Sharpe", "Max drawdown"]]
    wf = fmt_perf(wf_perf)[["CAGR", "Volatility", "Sharpe", "Max drawdown", "Turnover / yr"]]
    picks = ", ".join(f"{r.year}: {r.params}" for r in wf_choices.itertuples())
    return f"""_Daily adjusted closes from {info['source']}, {start.date()} to {info['end']}; cash earns the 3-month T-bill rate.
Every trade costs {cost_bps:.0f} bps of the traded amount (commission + slippage) and executes {lag} day after the signal.
In-sample / out-of-sample split at {split[:10]}. Full detail in [results/summary.md](results/summary.md)._

**Performance, {start.year}–{info['end'][:4]}, net of costs**

{md_table(fmt_perf(perf))}

**Out of sample only ({split[:4]}–{info['end'][:4]})**

{md_table(oos)}

**Is it real? Statistical tests (vs. SPY buy-and-hold)**

{md_table(st)}

_PSR: probability the true Sharpe is above zero, adjusted for skew and fat tails. Deflated Sharpe: the same probability
against the best Sharpe expected from the number of parameter sets tried. CI and P(boot ≤ 0): stationary block bootstrap
of the paired daily returns, 21-day mean blocks._

**Trading-cost sensitivity**

{md_table(ct, index=False)}

**SMA parameter grid: in-sample vs. out-of-sample Sharpe (sorted by in-sample rank)**

{md_table(grid, index=False)}

**Walk-forward SMA ({wf_start.year}–{info['end'][:4]}, re-optimised each January on the trailing 5 years)**

{md_table(wf)}

_Chosen windows: {picks}._

**Key findings**

""" + "\n".join(findings) + "\n"


def update_readme(md: str):
    path = ROOT / "README.md"
    if not path.exists():
        return
    txt = path.read_text()
    new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                 lambda _: "<!-- RESULTS:START -->\n" + md + "<!-- RESULTS:END -->", txt, flags=re.S)
    path.write_text(new)


if __name__ == "__main__":
    main()
