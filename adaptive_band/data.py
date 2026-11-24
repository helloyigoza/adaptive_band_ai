from __future__ import annotations

import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Yahoo Finance natively supported intervals.
YAHOO_NATIVE_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}

# Custom intervals built by resampling from a base interval.
# value = (base_interval_to_fetch, pandas_resample_rule)
RESAMPLE_MAP: dict[str, tuple[str, str]] = {
    "2h": ("1h", "2h"),
    "3h": ("1h", "3h"),
    "4h": ("1h", "4h"),
    "6h": ("1h", "6h"),
    "8h": ("1h", "8h"),
    "12h": ("1h", "12h"),
    "2d": ("1d", "2D"),
    "3d": ("1d", "3D"),
}

# Cache TTL per interval class (seconds).
_INTRADAY_TTL = 30 * 60  # 30 minutes
_DAILY_TTL = 6 * 60 * 60  # 6 hours

CACHE_DIR = Path("data_cache")


def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    aliases = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "BNB": "BNB-USD",
        "XRP": "XRP-USD",
        "DOGE": "DOGE-USD",
    }
    return aliases.get(s, s)


def default_start_for_interval(interval: str, requested_start: str = "2022-01-01") -> str:
    """Yahoo often limits intraday history. We still request what user asked for,
    but this helper can be used by callers to pick a safer fallback.
    """
    intraday = interval.endswith("m") or interval in {"60m", "90m", "1h"}
    if not intraday:
        return requested_start
    # Yahoo caps: ~60 days for sub-hour, ~730 days for 1h/60m/90m.
    if interval in {"1h", "60m", "90m"}:
        days = 729
    else:
        days = 59
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return max(requested_start, dt.strftime("%Y-%m-%d"))


def _is_intraday(interval: str) -> bool:
    return interval.endswith("m") or interval in {"60m", "90m", "1h"} or interval in RESAMPLE_MAP and RESAMPLE_MAP[interval][0] == "1h"


def _cache_path(ticker: str, interval: str, start: str, end: Optional[str]) -> Path:
    end_part = end or "now"
    safe = f"{ticker}_{interval}_{start}_{end_part}".replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.pkl"


def _ttl_for(interval: str) -> int:
    return _INTRADAY_TTL if _is_intraday(interval) else _DAILY_TTL


def _load_cache(path: Path, ttl: int) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl:
        return None
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"[cache] {path.name} kullanılıyor (yaş={int(age)}s, ttl={ttl}s)")
            return df
    except Exception as exc:
        print(f"[uyarı] cache okunamadı ({path.name}): {exc}")
    return None


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(df, f)
    except Exception as exc:
        print(f"[uyarı] cache yazılamadı ({path.name}): {exc}")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule, label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    return out


def _fetch_native(ticker: str, interval: str, start: str, end: Optional[str]) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance kurulu değil. `pip install -r requirements.txt` çalıştır.") from exc

    kwargs = dict(tickers=ticker, interval=interval, auto_adjust=False, progress=False, group_by="column")
    if end:
        kwargs.update(start=start, end=end)
    else:
        kwargs.update(start=start)

    df = yf.download(**kwargs)

    if df.empty and (interval.endswith("m") or interval in {"60m", "90m", "1h"}):
        fallback_start = default_start_for_interval(interval, start)
        if fallback_start != start:
            print(f"[uyarı] Yahoo intraday limiti nedeniyle start={start} -> {fallback_start} olarak güncellendi ({interval}).")
        kwargs["start"] = fallback_start
        df = yf.download(**kwargs)

    if df.empty:
        raise RuntimeError(f"Yahoo Finance veri döndürmedi: symbol={ticker}, interval={interval}, start={start}, end={end}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns=rename)
    keep = ["open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"Eksik kolonlar: {missing}")

    out = df[keep].copy()
    out.index = pd.to_datetime(out.index)
    out = out.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def download_ohlcv(symbol: str, interval: str, start: str = "2022-01-01", end: Optional[str] = None) -> pd.DataFrame:
    ticker = normalize_symbol(symbol)

    # Resolve fetch interval (resample if needed).
    if interval in RESAMPLE_MAP:
        base_interval, rule = RESAMPLE_MAP[interval]
    elif interval in YAHOO_NATIVE_INTERVALS:
        base_interval, rule = interval, None
    else:
        raise RuntimeError(
            f"Desteklenmeyen interval: {interval}. "
            f"Native: {sorted(YAHOO_NATIVE_INTERVALS)} | Resample: {sorted(RESAMPLE_MAP)}"
        )

    # Cache key uses the *requested* interval so resampled data is cached separately too.
    cache_path = _cache_path(ticker, interval, start, end)
    ttl = _ttl_for(interval)
    cached = _load_cache(cache_path, ttl)
    if cached is not None:
        cached.attrs["symbol"] = ticker
        cached.attrs["interval"] = interval
        return cached

    base_df = _fetch_native(ticker, base_interval, start, end)

    if rule is not None:
        out = _resample_ohlcv(base_df, rule)
        if out.empty:
            raise RuntimeError(f"Resample sonrası veri boş: {ticker} {interval}")
        print(f"[resample] {base_interval} -> {interval} ({len(base_df)} bar -> {len(out)} bar)")
    else:
        out = base_df

    out.attrs["symbol"] = ticker
    out.attrs["interval"] = interval
    _save_cache(cache_path, out)
    return out
