# Results

_Daily adjusted closes from Yahoo Finance (yfinance), 2000-02-02 to 2026-11-25; cash earns the 3-month T-bill rate.
Every trade costs 10 bps of the traded amount (commission + slippage) and executes 1 day after the signal.
In-sample / out-of-sample split at 2013-01-01. Full detail in [results/summary.md](results/summary.md)._

**Performance, 2000–2026, net of costs**

|                      | CAGR   | Volatility   |   Sharpe |   Sortino | Max drawdown   |   Calmar |   Beta | Alpha / yr   | Turnover / yr   | Cost drag / yr   | Time invested   |
|:---------------------|:-------|:-------------|---------:|----------:|:---------------|---------:|-------:|:-------------|:----------------|:-----------------|:----------------|
| SPY buy & hold       | 8.1%   | 19.2%        |     0.4  |      0.57 | -55%           |     0.15 |   1    | 0.0%         | 0.0×            | 0.00%            | 100%            |
| Sector equal weight  | 8.8%   | 18.0%        |     0.45 |      0.63 | -53%           |     0.17 |   0.9  | 1.1%         | 0.3×            | 0.03%            | 100%            |
| SMA 50/200 trend     | 8.1%   | 13.4%        |     0.51 |      0.7  | -34%           |     0.24 |   0.49 | 3.0%         | 1.0×            | 0.10%            | 74%             |
| Time-series momentum | 8.2%   | 14.3%        |     0.49 |      0.69 | -34%           |     0.24 |   0.56 | 2.7%         | 4.2×            | 0.42%            | 78%             |
| Sector momentum      | 7.9%   | 17.2%        |     0.42 |      0.58 | -35%           |     0.23 |   0.7  | 1.8%         | 5.1×            | 0.51%            | 95%             |

**Out of sample only (2013–2026)**

|                      | CAGR   | Volatility   |   Sharpe | Max drawdown   |
|:---------------------|:-------|:-------------|---------:|:---------------|
| SPY buy & hold       | 14.0%  | 16.8%        |     0.76 | -34%           |
| Sector equal weight  | 12.4%  | 15.7%        |     0.71 | -36%           |
| SMA 50/200 trend     | 9.5%   | 14.5%        |     0.58 | -34%           |
| Time-series momentum | 10.8%  | 15.5%        |     0.63 | -34%           |
| Sector momentum      | 11.3%  | 17.1%        |     0.61 | -31%           |

**Is it real? Statistical tests (vs. SPY buy-and-hold)**

|                      |   Sharpe |   t-stat (excess return) | PSR (SR > 0)   |   Trials |   Hurdle Sharpe | Deflated Sharpe   |   Sharpe − SPY | 95% CI         | P(boot ≤ 0)   |
|:---------------------|---------:|-------------------------:|:---------------|---------:|----------------:|:------------------|---------------:|:---------------|:--------------|
| SMA 50/200 trend     |     0.51 |                     2.63 | 100%           |       15 |            0.15 | 97%               |           0.1  | -0.18 to +0.41 | 24%           |
| Time-series momentum |     0.49 |                     2.54 | 99%            |        4 |            0.23 | 91%               |           0.09 | -0.17 to +0.35 | 26%           |
| Sector momentum      |     0.42 |                     2.16 | 98%            |       12 |            0.14 | 93%               |           0.01 | -0.20 to +0.22 | 47%           |

_PSR: probability the true Sharpe is above zero, adjusted for skew and fat tails. Deflated Sharpe: the same probability
against the best Sharpe expected from the number of parameter sets tried. CI and P(boot ≤ 0): stationary block bootstrap
of the paired daily returns, 21-day mean blocks._

**Trading-cost sensitivity**

|   Cost per trade (bps) |   SMA 50/200 trend Sharpe |   Time-series momentum Sharpe |   Sector momentum Sharpe | SMA 50/200 trend CAGR   | Time-series momentum CAGR   | Sector momentum CAGR   |
|-----------------------:|--------------------------:|------------------------------:|-------------------------:|:------------------------|:----------------------------|:-----------------------|
|                      0 |                      0.52 |                          0.51 |                     0.44 | 8.5%                    | 8.6%                        | 8.3%                   |
|                      5 |                      0.52 |                          0.49 |                     0.42 | 8.4%                    | 8.3%                        | 8.0%                   |
|                     10 |                      0.52 |                          0.48 |                     0.41 | 8.4%                    | 8.1%                        | 7.8%                   |
|                     25 |                      0.51 |                          0.44 |                     0.37 | 8.2%                    | 7.5%                        | 7.0%                   |
|                     50 |                      0.49 |                          0.36 |                     0.29 | 7.9%                    | 6.4%                        | 5.7%                   |

**SMA parameter grid: in-sample vs. out-of-sample Sharpe (sorted by in-sample rank)**

|   fast |   slow |   IS Sharpe |   OOS Sharpe |   IS rank |   OOS rank |
|-------:|-------:|------------:|-------------:|----------:|-----------:|
|     50 |    250 |        0.43 |         0.62 |         1 |          8 |
|     50 |    200 |        0.41 |         0.58 |         2 |         11 |
|     20 |    200 |        0.39 |         0.66 |         3 |          4 |
|     10 |    200 |        0.39 |         0.69 |         4 |          3 |
|     50 |    150 |        0.37 |         0.61 |         5 |         10 |
|     20 |    250 |        0.32 |         0.53 |         6 |         12 |
|     10 |    250 |        0.3  |         0.63 |         7 |          6 |
|     10 |    150 |        0.28 |         0.61 |         8 |          9 |
|     50 |    100 |        0.25 |         0.41 |         9 |         15 |
|    100 |    250 |        0.21 |         0.71 |        10 |          2 |
|    100 |    150 |        0.17 |         0.63 |        11 |          7 |
|    100 |    200 |        0.14 |         0.63 |        12 |          5 |
|     10 |    100 |        0.1  |         0.76 |        13 |          1 |
|     20 |    150 |        0.09 |         0.53 |        14 |         13 |
|     20 |    100 |        0.04 |         0.46 |        15 |         14 |

**Walk-forward SMA (2004–2026, re-optimised each January on the trailing 5 years)**

|                  | CAGR   | Volatility   |   Sharpe | Max drawdown   | Turnover / yr   |
|:-----------------|:-------|:-------------|---------:|:---------------|:----------------|
| SMA walk-forward | 6.7%   | 13.4%        |     0.42 | -34%           | 2.1×            |
| SMA 50/200 trend | 8.5%   | 13.7%        |     0.53 | -34%           | 1.0×            |
| SPY buy & hold   | 10.5%  | 18.6%        |     0.53 | -55%           | –               |

_Chosen windows: 2004: 50/100, 2005: 50/250, 2006: 50/250, 2007: 50/250, 2008: 50/250, 2009: 50/250, 2010: 50/200, 2011: 10/100, 2012: 20/200, 2013: 50/100, 2014: 50/100, 2015: 50/100, 2016: 50/150, 2017: 100/150, 2018: 100/150, 2019: 20/200, 2020: 100/250, 2021: 10/100, 2022: 10/100, 2023: 10/100, 2024: 10/200, 2025: 10/200, 2026: 20/200._

**Key findings**

- Over the full period, SPY buy-and-hold returned **8.1% a year** with a Sharpe of 0.40 and a **-55%** worst drawdown. The best risk-adjusted strategy was **SMA 50/200 trend** (Sharpe 0.51, CAGR 8.1%, max drawdown -34%).
- The trend filters' clearest effect is on **risk**: SMA 50/200 trend and Time-series momentum cut the worst drawdown from -55% to -34% / -34% by moving to T-bills in long declines, while staying invested 74% / 78% of the time. Beta falls to 0.49 / 0.56.
- Only Time-series momentum beat buy-and-hold on raw return (8.2% vs 8.1%).
- **Statistical significance.** No strategy's Sharpe improvement over SPY is significant at 5%: SMA 50/200 trend +0.10 (95% CI -0.18 to +0.41); Time-series momentum +0.09 (95% CI -0.17 to +0.35); Sector momentum +0.01 (95% CI -0.20 to +0.22). With about two decades of daily data, Sharpe differences of this size are hard to tell apart from luck.
- **Multiple-testing correction.** The deflated Sharpe ratio asks whether the result beats the best of all parameter sets tried, not zero: SMA 50/200 trend 97% (vs 100% unadjusted, 15 trials); Time-series momentum 91% (vs 99% unadjusted, 4 trials); Sector momentum 93% (vs 98% unadjusted, 12 trials). A deflated value below 95% means the edge does not survive the search.
- **Overfitting check.** On the 15-setting SMA grid, the best in-sample choice (50/250, Sharpe 0.43) ranked **#8 of 15** after 2013 (Sharpe 0.62). The rank correlation between in-sample and out-of-sample Sharpe is 0.10: in-sample rankings told you little about what would work next.
- **Walk-forward.** Re-optimising the SMA windows every January on the trailing five years earned a Sharpe of 0.42 out of sample, below the fixed textbook 50/200 rule (0.53) over the same years. Optimisation added nothing reliable over a rule fixed in advance.
- **Sector momentum vs. its own benchmark.** Against an equal-weight portfolio of the same sectors (Sharpe 0.45), sector momentum's Sharpe difference is -0.03 (95% CI -0.25 to +0.18). It trades 5.1× its capital a year, costing 0.51% annually at 10 bps.
- **Costs.** Moving from 0 to 50 bps per trade lowers Sharpe by SMA 50/200 trend 0.04; Time-series momentum 0.14; Sector momentum 0.15. Sector momentum is the most cost-sensitive: it trades 5.1× its capital a year.
- **After 2013** (out of sample): SPY Sharpe 0.76, SMA 50/200 trend 0.58, Time-series momentum 0.63, Sector momentum 0.61. Every strategy lagged buy-and-hold on a risk-adjusted basis in the untouched period.
