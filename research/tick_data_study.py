"""
Tick data study: can every trade of a past minute be fetched, and is it complete?
=================================================================================

WHY (AGENTS.md §2.6)
--------------------
Trades will be resolved on 1-minute candles. Two kinds of minute need the exchange's
individual trades ("ticks") to know what happened first:
  * the minute that contains the entry (only trades after the entry time count), and
  * any minute in which both the stop and the target were touched.
Before that design is planned, we must know, for the main exchange and the backups:
  1. how far back individual trades can be fetched, and through which source, and
  2. whether they are complete, i.e. they rebuild the exchange's own 1-minute candle
     exactly: first trade = open, last = close, highest = high, lowest = low, and the
     traded quantities add up to the candle's volume.

WHAT IT DOES
------------
Public market data only: no API key, no Gemini calls, nothing written outside
research/results/. For each source it fetches trades for some past minutes (including
the busiest minute it can find, the hardest case) and compares them with the 1-minute
candles of the same exchange. For the daily archives it checks every minute of a whole
day. Downloads: about 60 MB in total (two daily archive files).

HOW TO RUN IT (from the project folder)
---------------------------------------
    uv run research/tick_data_study.py

It takes a few minutes. Results are printed and saved as a dated Markdown report in
research/results/. Re-run it if the exchange, the symbol or the resolution design changes.
"""

import gzip
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"
MINUTE_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
TIMEOUT_MS = 20_000
MAX_PAGES = 300  # per minute; the busiest minutes need a few dozen
# Prices must match exactly. Quantities are compared with a tiny tolerance for decimal rounding.
VOLUME_TOLERANCE = 1e-6

PERP = "BTC/USDT:USDT"
OKX_INSTRUMENT = "BTC-USDT-SWAP"


# =============================================================================
# Helpers
# =============================================================================

def retry(call: Callable[[], Any], what: str) -> Any:
    """Retry temporary network failures 3 times with growing waits (AGENTS.md §5)."""
    for attempt in range(3):
        try:
            return call()
        except (ccxt.NetworkError, urllib.error.URLError, TimeoutError) as e:
            if attempt == 2:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  temporary error on {what} ({type(e).__name__}), retrying in {wait}s")
            time.sleep(wait)


def utc(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def download(url: str) -> tuple[bytes, str]:
    """File content and its Last-Modified header."""
    def get() -> tuple[bytes, str]:
        with urllib.request.urlopen(url, timeout=TIMEOUT_MS / 1000 * 6) as response:
            return response.read(), response.headers.get("Last-Modified", "unknown")
    return retry(get, url)


def exists(url: str) -> bool:
    def head() -> bool:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=TIMEOUT_MS / 1000) as response:
                return response.status == 200
        except urllib.error.HTTPError:
            return False
    return retry(head, url)


def candles_1m(exchange: ccxt.Exchange, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Closed 1-minute candles from start_ms (inclusive) to end_ms (exclusive), indexed by open time."""
    rows: list[list[float]] = []
    since = start_ms
    while since < end_ms:
        page = retry(lambda: exchange.fetch_ohlcv(symbol, "1m", since=since, limit=1000), f"{exchange.id} candles")
        page = [r for r in page if since <= r[0] < end_ms]
        if not page:
            break
        rows.extend(page)
        since = int(page[-1][0]) + MINUTE_MS
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    return frame.set_index("ts").sort_index()


@dataclass
class MinuteCheck:
    """How well one minute's trades rebuild its candle."""

    minute_ms: int
    trades: int
    fields: tuple[str, ...]  # fields that did not match; empty when the minute is exact
    detail: str

    @property
    def complete(self) -> bool:
        return not self.fields


def compare_minutes(trades: pd.DataFrame, candles: pd.DataFrame, volume_scale: float = 1.0, open_is_previous_close: bool = False) -> list[MinuteCheck]:
    """Rebuild each candle from trades (columns ts, price, amount) and compare field by field.

    volume_scale converts the trade quantity into the candle's volume unit (OKX: contracts → BTC).
    open_is_previous_close tests the convention where a candle opens at the previous minute's last
    trade, so its high and low include that price too.
    """
    trades = trades.sort_values("ts", kind="stable")
    trades["minute"] = (trades["ts"] // MINUTE_MS) * MINUTE_MS
    grouped = trades.groupby("minute").agg(
        n=("price", "size"), open=("price", "first"), close=("price", "last"),
        high=("price", "max"), low=("price", "min"), volume=("amount", "sum"),
    )
    if open_is_previous_close:
        previous_close = grouped["close"].shift(1)
        has_previous = previous_close.notna() & (grouped.index.to_series().diff() == MINUTE_MS)
        grouped.loc[has_previous, "open"] = previous_close[has_previous]
        grouped["high"] = grouped[["high", "open"]].max(axis=1)
        grouped["low"] = grouped[["low", "open"]].min(axis=1)
    checks = []
    for minute, candle in candles.iterrows():
        if minute not in grouped.index:
            fields = () if candle["volume"] == 0 else ("missing",)
            checks.append(MinuteCheck(int(minute), 0, fields, "no trades"))
            continue
        rebuilt = grouped.loc[minute]
        fields = [field for field in ("open", "close", "high", "low") if rebuilt[field] != candle[field]]
        problems = [f"{field} {rebuilt[field]} vs {candle[field]}" for field in fields]
        volume = rebuilt["volume"] * volume_scale
        if abs(volume - candle["volume"]) > VOLUME_TOLERANCE * max(1.0, candle["volume"]):
            fields.append("volume")
            problems.append(f"volume {volume:.6f} vs {candle['volume']:.6f}")
        checks.append(MinuteCheck(int(minute), int(rebuilt["n"]), tuple(fields), "; ".join(problems) or "exact"))
    return checks


def volume_totals(trades: pd.DataFrame, candles: pd.DataFrame, volume_scale: float = 1.0) -> str:
    """Total traded quantity vs total candle volume over the candles' period."""
    start, end = candles.index.min(), candles.index.max() + MINUTE_MS
    inside = trades[(trades["ts"] >= start) & (trades["ts"] < end)]
    return f"{inside['amount'].sum() * volume_scale:,.4f} (trades) vs {candles['volume'].sum():,.4f} (candles)"


def busiest_minute(candles: pd.DataFrame) -> int:
    return int(candles["volume"].idxmax())


# =============================================================================
# Sources
# =============================================================================

def binance_rest_minute(exchange: ccxt.Exchange, minute_ms: int) -> pd.DataFrame:
    """Every aggregated trade of one minute through the REST API, paging by trade id."""
    end = minute_ms + MINUTE_MS
    rows: list[dict] = []
    page = retry(lambda: exchange.fetch_trades(PERP, since=minute_ms, limit=1000, params={"endTime": end - 1}), "binance trades")
    for _ in range(MAX_PAGES):
        rows.extend(page)
        if len(page) < 1000 or page[-1]["timestamp"] >= end:
            break
        next_id = int(page[-1]["id"]) + 1
        page = retry(lambda: exchange.fetch_trades(PERP, limit=1000, params={"fromId": next_id}), "binance trades")
    frame = pd.DataFrame({"ts": [t["timestamp"] for t in rows], "price": [t["price"] for t in rows], "amount": [t["amount"] for t in rows]})
    return frame[(frame["ts"] >= minute_ms) & (frame["ts"] < end)]


def binance_rest_window(exchange: ccxt.Exchange) -> dict[str, str]:
    """Probe how far back the REST trade endpoint answers."""
    now = int(time.time() * 1000)
    result = {}
    for hours in (1, 24, 47, 49, 24 * 7):
        start = (now - hours * HOUR_MS) // MINUTE_MS * MINUTE_MS
        try:
            trades = retry(lambda: exchange.fetch_trades(PERP, since=start, limit=5), "binance window")
            result[f"{hours} h ago"] = f"answers ({len(trades)} trades)"
        except ccxt.ExchangeError as e:
            result[f"{hours} h ago"] = f"refused: {str(e)[:100]}"
    return result


def binance_archive_day(day: date) -> tuple[pd.DataFrame, str]:
    """A whole day of USDT-M aggregated trades from data.binance.vision."""
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{day}.zip"
    content, published = download(url)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        frame = pd.read_csv(archive.open(archive.namelist()[0]))
    ts = frame["transact_time"]
    if ts.max() > 10**14:  # microseconds in some newer archives
        ts = ts // 1000
    return pd.DataFrame({"ts": ts, "price": frame["price"], "amount": frame["quantity"]}), published


def okx_rest_minute(exchange: ccxt.Exchange, minute_ms: int) -> pd.DataFrame:
    """Every trade of one minute from OKX's history endpoint.

    The first page is found by time (type=2: trades before a timestamp); later pages go backwards by
    trade id (type=1). Paging by time alone skips trades that share the boundary millisecond.
    """
    end = minute_ms + MINUTE_MS
    first = retry(lambda: exchange.publicGetMarketHistoryTrades({"instId": OKX_INSTRUMENT, "type": "2", "after": str(end), "limit": "100"}), "okx trades")
    rows: list[dict] = list(first.get("data", []))
    for _ in range(MAX_PAGES):
        if not rows:
            break
        oldest = min(rows, key=lambda t: int(t["tradeId"]))
        if int(oldest["ts"]) < minute_ms:
            break
        response = retry(lambda: exchange.publicGetMarketHistoryTrades({"instId": OKX_INSTRUMENT, "type": "1", "after": oldest["tradeId"], "limit": "100"}), "okx trades")
        page = response.get("data", [])
        if not page:
            break
        rows.extend(page)
    frame = pd.DataFrame({
        "ts": [int(t["ts"]) for t in rows], "price": [float(t["px"]) for t in rows],
        "amount": [float(t["sz"]) for t in rows], "id": [t["tradeId"] for t in rows],
    }).drop_duplicates("id")
    # Same-millisecond trades in exchange order (trade ids increase).
    frame["id"] = frame["id"].astype("int64")
    frame = frame.sort_values(["ts", "id"])
    return frame[(frame["ts"] >= minute_ms) & (frame["ts"] < end)].drop(columns="id")


def okx_rest_depth(exchange: ccxt.Exchange) -> dict[str, str]:
    now = int(time.time() * 1000)
    result = {}
    for days in (1, 30, 85, 95, 120):
        target = now - days * DAY_MS
        response = retry(lambda: exchange.publicGetMarketHistoryTrades({"instId": OKX_INSTRUMENT, "type": "2", "after": str(target), "limit": "5"}), "okx depth")
        rows = response.get("data", [])
        result[f"{days} days ago"] = f"answers (newest {utc(int(rows[0]['ts']))})" if rows else "empty"
    return result


def bybit_archive_day(day: date) -> tuple[pd.DataFrame, str]:
    url = f"https://public.bybit.com/trading/BTCUSDT/BTCUSDT{day}.csv.gz"
    content, published = download(url)
    frame = pd.read_csv(io.BytesIO(gzip.decompress(content)))
    return pd.DataFrame({"ts": (frame["timestamp"] * 1000).round().astype("int64"), "price": frame["price"], "amount": frame["size"]}), published


def bybit_rest_ignores_since(exchange: ccxt.Exchange) -> str:
    target = int(time.time() * 1000) - DAY_MS
    trades = retry(lambda: exchange.fetch_trades(PERP, since=target, limit=50), "bybit trades")
    if trades and trades[0]["timestamp"] > target + HOUR_MS:
        return f"asked for {utc(target)}, got trades from {utc(trades[0]['timestamp'])}: recent trades only"
    return "answered with trades near the requested time"


# =============================================================================
# Report
# =============================================================================

def summarise(checks: list[MinuteCheck]) -> tuple[int, int, list[MinuteCheck]]:
    bad = [c for c in checks if not c.complete]
    return len(checks), len(checks) - len(bad), bad


def field_summary(checks: list[MinuteCheck]) -> str:
    """Exact minutes, minutes whose high and low match (what decides a stop or target hit), and mismatches per field."""
    total = len(checks)
    exact = sum(c.complete for c in checks)
    range_exact = sum(not ({"high", "low", "missing"} & set(c.fields)) for c in checks)
    counts = {f: sum(f in c.fields for c in checks) for f in ("open", "close", "high", "low", "volume", "missing")}
    per_field = ", ".join(f"{f} {n}" for f, n in counts.items() if n) or "none"
    return f"{exact} of {total} minutes exact; high and low exact in {range_exact} of {total}; mismatches by field: {per_field}"


def mismatch_table(label: str, bad: list[MinuteCheck]) -> list[str]:
    """Every minute whose high or low differs (those decide stop and target hits), then the first 10 others."""
    price_range = [c for c in bad if {"high", "low", "missing"} & set(c.fields)]
    others = [c for c in bad if c not in price_range]
    header = ["| Source | Minute | Trades | Complete | Detail |", "|---|---|---|---|---|"]
    lines = ["", f"Minutes whose high or low differ (all {len(price_range)}):", ""]
    lines += header + minute_lines(label, price_range) if price_range else ["none"]
    lines += ["", f"Other mismatches (first 10 of {len(others)}):", ""]
    lines += header + minute_lines(label, others[:10]) if others else ["none"]
    return lines


def minute_lines(label: str, checks: list[MinuteCheck]) -> list[str]:
    return [f"| {label} | {utc(c.minute_ms)} | {c.trades:,} | {'yes' if c.complete else 'NO'} | {c.detail} |" for c in checks]


def main() -> None:
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    now = int(time.time() * 1000)
    lines = [f"# Tick data study ({today})", "", "Generated by `research/tick_data_study.py` (public data only).", ""]

    # --- Binance USDT-M (main exchange) ---
    binance = ccxt.binanceusdm({"enableRateLimit": True, "timeout": TIMEOUT_MS})
    print("Binance USDT-M: REST window ...")
    window = binance_rest_window(binance)

    print("Binance USDT-M: REST completeness ...")
    recent = candles_1m(binance, PERP, now - 46 * HOUR_MS, now - 10 * MINUTE_MS)
    rest_minutes = [now - 1 * HOUR_MS, now - 24 * HOUR_MS, now - 46 * HOUR_MS + 30 * MINUTE_MS]
    rest_minutes = [m // MINUTE_MS * MINUTE_MS for m in rest_minutes] + [busiest_minute(recent)]
    rest_checks: list[MinuteCheck] = []
    for minute in rest_minutes:
        candle = candles_1m(binance, PERP, minute, minute + MINUTE_MS)
        rest_checks += compare_minutes(binance_rest_minute(binance, minute), candle)

    print(f"Binance USDT-M: archive for {yesterday} (every minute) ...")
    archive_trades, archive_published = binance_archive_day(yesterday)
    day_start = int(datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc).timestamp() * 1000)
    day_candles = candles_1m(binance, PERP, day_start, day_start + DAY_MS)
    archive_checks = compare_minutes(archive_trades, day_candles)
    archive_depth = {f"{days} days ago": exists(f"https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{today - timedelta(days=days)}.zip") for days in (1, 365, 3 * 365)}
    today_file = exists(f"https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{today}.zip")

    lines += ["## Binance USDT-M perpetual (main exchange)", "", "**REST `aggTrades` window** (by time; paging by trade id hits the same limit):", ""]
    lines += [f"- {k}: {v}" for k, v in window.items()]
    lines += ["", f"**REST completeness:** {field_summary(rest_checks)}. The last row is the busiest minute of the last 46 h.", "",
              "| Source | Minute | Trades | Complete | Detail |", "|---|---|---|---|---|"]
    lines += minute_lines("REST", rest_checks)
    _, _, bad = summarise(archive_checks)
    lines += ["", f"**Daily archive** (`data.binance.vision`, {yesterday}, published {archive_published}): {field_summary(archive_checks)}.",
              f"Whole-day volume: {volume_totals(archive_trades, day_candles)}. Busiest minute that day: {utc(busiest_minute(day_candles))}. Today's file already published: {today_file}.", ""]
    lines += [f"- archive for {k}: {'available' if v else 'missing'}" for k, v in archive_depth.items()]
    if bad:
        lines += mismatch_table("archive", bad)

    # --- OKX (backup) ---
    print("OKX: REST history depth and completeness ...")
    okx = ccxt.okx({"enableRateLimit": True, "timeout": TIMEOUT_MS})
    retry(okx.load_markets, "okx markets")
    contract_size = float(okx.market(PERP)["contractSize"])  # trades are in contracts, ccxt candles in BTC
    okx_depth = okx_rest_depth(okx)
    okx_checks: list[MinuteCheck] = []
    okx_recent = candles_1m(okx, PERP, now - 6 * HOUR_MS, now - 10 * MINUTE_MS)
    okx_minutes = [(now - DAY_MS) // MINUTE_MS * MINUTE_MS, (now - 60 * DAY_MS) // MINUTE_MS * MINUTE_MS, busiest_minute(okx_recent)]
    for minute in okx_minutes:
        okx_candle = candles_1m(okx, PERP, minute, minute + MINUTE_MS)
        okx_checks += compare_minutes(okx_rest_minute(okx, minute), okx_candle, volume_scale=contract_size)
    lines += ["", "## OKX perpetual (backup)", "", "**REST `history-trades` (type=2, paging by time) depth:**", ""]
    lines += [f"- {k}: {v}" for k, v in okx_depth.items()]
    lines += ["", f"**Completeness:** {field_summary(okx_checks)} (trades in contracts × {contract_size} BTC). The last row is the busiest minute of the last 6 h.", "",
              "| Source | Minute | Trades | Complete | Detail |", "|---|---|---|---|---|"]
    lines += minute_lines("REST", okx_checks)
    lines += ["", "Note: ccxt's plain `fetch_trades` ignores `since` on OKX (it returns the latest trades); the raw endpoint is needed."]

    # --- Bybit (backup) ---
    print(f"Bybit: REST behaviour and archive for {yesterday} (every minute) ...")
    bybit = ccxt.bybit({"enableRateLimit": True, "timeout": TIMEOUT_MS})
    bybit_rest = bybit_rest_ignores_since(bybit)
    bybit_trades, bybit_published = bybit_archive_day(yesterday)
    bybit_candles = candles_1m(bybit, PERP, day_start, day_start + DAY_MS)
    bybit_checks = compare_minutes(bybit_trades, bybit_candles)
    bybit_prev = compare_minutes(bybit_trades, bybit_candles, open_is_previous_close=True)
    _, _, bad = summarise(bybit_prev)
    lines += ["", "## Bybit perpetual (backup)", "", f"- REST `fetch_trades` with `since`: {bybit_rest}.",
              f"- Daily archive (`public.bybit.com`, {yesterday}, published {bybit_published}). Whole-day volume: {volume_totals(bybit_trades, bybit_candles)}.",
              f"  - Candle opens at its first trade: {field_summary(bybit_checks)}.",
              f"  - Candle opens at the previous minute's last trade: {field_summary(bybit_prev)}."]
    if bad:
        lines += mismatch_table("archive", bad)

    report = "\n".join(lines) + "\n"
    print("\n" + report)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    path = RESULTS_FOLDER / f"tick_data_{today}.md"
    path.write_text(report, encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    try:
        main()
    except (ccxt.BaseError, urllib.error.URLError, OSError) as e:
        print(f"Study failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
