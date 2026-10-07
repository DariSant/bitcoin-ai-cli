"""Market data from the exchange (public endpoints only, no keys)."""

import ccxt
import pandas as pd

from btc_cli import config


def create_exchange() -> ccxt.Exchange:
    """The exchange every command reads from (config.EXCHANGE_ID: Binance spot until the Phase 1 futures item)."""
    # Built from the config value, so records can't name a different exchange than the one used.
    return getattr(ccxt, config.EXCHANGE_ID)()


def fetch_ohlcv_data(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given symbol and timeframe and return a Pandas DataFrame.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=config.ANALYSIS_CANDLES)
    except ccxt.NetworkError as e:
        raise RuntimeError(f"Network error fetching data for {timeframe} timeframe: {e}")
    except ccxt.ExchangeError as e:
        raise RuntimeError(f"Exchange error fetching data for {timeframe} timeframe: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching data for {timeframe} timeframe: {e}")

    if not ohlcv:
        raise RuntimeError(f"No data returned for {timeframe} timeframe.")

    # Convert to Pandas DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df


def fetch_resolution_candles(symbol: str) -> list[list]:
    """Raw 15m candles used to resolve open trades: the last RESOLUTION_CANDLES (100 = 25 hours, known P0 limit)."""
    exchange = create_exchange()
    return exchange.fetch_ohlcv(symbol, timeframe="15m", limit=config.RESOLUTION_CANDLES)
