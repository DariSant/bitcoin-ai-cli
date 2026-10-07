"""Market data from the exchange (public endpoints only, no keys)."""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import ccxt
import pandas as pd

from btc_cli import config

T = TypeVar("T")

# One exchange object per process: ccxt downloads the market list once per object.
_exchange: ccxt.Exchange | None = None
# Patched by the tests so retries don't really wait.
_sleep = time.sleep


class MarketDataError(RuntimeError):
    """Market data could not be fetched, after retrying temporary failures."""


def create_exchange() -> ccxt.Exchange:
    """The exchange every command reads from (config.EXCHANGE_ID: Binance spot until the Phase 1 futures item).

    Built once from the config value, so records can't name a different exchange than the one used,
    with a request timeout and ccxt's own rate limiting.
    """
    global _exchange
    if _exchange is None:
        _exchange = getattr(ccxt, config.EXCHANGE_ID)({"timeout": config.EXCHANGE_TIMEOUT_SECONDS * 1000, "enableRateLimit": True})
    return _exchange


def with_retries(call: Callable[[], T], what: str) -> T:
    """Run an exchange call, retrying temporary network errors after 2 s and 4 s (AGENTS.md §5).

    ccxt.NetworkError covers timeouts, rate limits (DDoSProtection) and maintenance. ExchangeError
    (bad symbol, bad request) is never retried. Raises MarketDataError when it gives up.
    """
    attempts = config.EXCHANGE_MAX_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ccxt.NetworkError as e:
            if attempt == attempts:
                raise MarketDataError(f"Network error fetching {what} after {attempts} attempts: {e}") from e
            wait = 2.0 ** attempt
            logging.warning(f"Network error fetching {what} ({type(e).__name__}); retry {attempt + 1} of {attempts} in {wait:.0f}s")
            _sleep(wait)
        except ccxt.ExchangeError as e:
            raise MarketDataError(f"Exchange error fetching {what}: {e}") from e
    raise AssertionError("unreachable")


def fetch_ohlcv_data(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given symbol and timeframe and return a Pandas DataFrame.
    """
    try:
        ohlcv = with_retries(lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=config.ANALYSIS_CANDLES), f"{timeframe} candles")
    except MarketDataError:
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching data for {timeframe} timeframe: {e}")

    if not ohlcv:
        raise RuntimeError(f"No data returned for {timeframe} timeframe.")

    # Convert to Pandas DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df


def fetch_resolution_candles(symbol: str) -> list[list]:
    """Raw 15m candles used to resolve open trades: the last RESOLUTION_CANDLES (100 = 25 hours, known P0 limit).

    Raises MarketDataError if they can't be fetched.
    """
    exchange = create_exchange()
    return with_retries(lambda: exchange.fetch_ohlcv(symbol, timeframe="15m", limit=config.RESOLUTION_CANDLES), "15m candles to check open trades")
