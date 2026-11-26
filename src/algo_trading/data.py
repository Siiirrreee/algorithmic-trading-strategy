"""Market data: download, cache and validate daily prices and the T-bill rate.

Prices are dividend- and split-adjusted closes. The universe is fixed in advance
(SPY plus the Select Sector SPDR ETFs), so no ticker is chosen because it survived.
Funds that launched later (XLRE in 2015, XLC in 2018) simply have no price before
their launch date, and strategies treat them as ineligible until they have enough
history: a point-in-time universe.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK = "SPY"
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
UNIVERSE = [BENCHMARK] + SECTORS
START = "1999-01-01"
TRADING_DAYS = 252

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PRICE_FILE = DATA_DIR / "prices.csv"
TBILL_FILE = DATA_DIR / "tbill.csv"


# --------------------------------------------------------------------------- download
def _download_yfinance(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _download_stooq(tickers: list[str], start: str) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        url = f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        frames[t] = df["Close"]
    out = pd.DataFrame(frames)
    return out.loc[out.index >= start]


def download_prices(tickers: list[str] = UNIVERSE, start: str = START) -> tuple[pd.DataFrame, str]:
    """Adjusted daily closes from Yahoo Finance, with Stooq as a fallback."""
    errors = []
    for name, fn in (("Yahoo Finance (yfinance)", _download_yfinance), ("Stooq", _download_stooq)):
        try:
            px = fn(tickers, start)
            px = px.reindex(columns=tickers)
            if px[BENCHMARK].notna().sum() > TRADING_DAYS * 5:
                return px, name
            errors.append(f"{name}: too little data")
        except Exception as exc:  # network or parsing failure: try the next source
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Could not download prices. " + " | ".join(errors))


def download_tbill(start: str = START) -> pd.Series:
    """3-month T-bill secondary-market rate (FRED DTB3), annualised percent."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
    try:
        df = pd.read_csv(url, na_values=".")
        df.columns = ["date", "rate"]
        s = df.set_index(pd.to_datetime(df["date"]))["rate"].astype(float)
    except Exception:
        import yfinance as yf

        s = yf.download("^IRX", start=start, progress=False, auto_adjust=False)["Close"].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "tbill_pct"
    return s.loc[s.index >= start].dropna()


# --------------------------------------------------------------------------- cache
def load_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return (prices, daily risk-free return, info). Uses the cached CSVs unless refresh."""
    DATA_DIR.mkdir(exist_ok=True)
    source = "cached CSV"
    if refresh or not PRICE_FILE.exists():
        px, source = download_prices()
        px.to_csv(PRICE_FILE, float_format="%.6f")
        download_tbill().to_csv(TBILL_FILE)
    px = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    tb = pd.read_csv(TBILL_FILE, index_col=0, parse_dates=True).iloc[:, 0]
    px, rf = clean(px, tb)
    info = {
        "source": source,
        "start": str(px.index[0].date()),
        "end": str(px.index[-1].date()),
        "days": int(len(px)),
        "first_valid": {c: str(px[c].first_valid_index().date()) for c in px.columns},
    }
    return px, rf, info


def clean(px: pd.DataFrame, tbill_pct: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Align to SPY trading days, forward-fill only small interior gaps, build daily rf."""
    px = px.sort_index()
    px = px.loc[px[BENCHMARK].notna()]
    px = px[~px.index.duplicated()]
    # Fill at most 3 consecutive missing interior quotes; never back-fill before launch.
    first = {c: px[c].first_valid_index() for c in px.columns}
    px = px.ffill(limit=3)
    for c, d in first.items():
        if d is not None:
            px.loc[px.index < d, c] = np.nan
    tb = tbill_pct.reindex(px.index, method="ffill").fillna(0.0)
    rf = (1 + tb / 100) ** (1 / TRADING_DAYS) - 1
    rf.name = "rf"
    return px, rf


# --------------------------------------------------------------------------- synthetic
def synthetic_data(n_days: int = 252 * 12, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Correlated GBM prices with a launch-date gap, for tests and offline runs."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    k = len(UNIVERSE)
    corr = 0.6 * np.ones((k, k)) + 0.4 * np.eye(k)
    vols = rng.uniform(0.15, 0.30, k) / np.sqrt(TRADING_DAYS)
    mus = rng.uniform(0.02, 0.10, k) / TRADING_DAYS
    z = rng.standard_normal((n_days, k)) @ np.linalg.cholesky(corr).T
    rets = mus + vols * z
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=UNIVERSE)
    px.loc[px.index < idx[n_days // 2], "XLC"] = np.nan  # launches mid-sample
    rf = pd.Series(0.02 / TRADING_DAYS, index=idx, name="rf")
    return px, rf
