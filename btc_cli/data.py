"""Market data from the exchange (public endpoints only, no keys)."""

import ccxt
import pandas as pd


def create_exchange() -> ccxt.Exchange:
    """The exchange every command reads from (Binance spot until the Phase 1 futures item)."""
    return ccxt.binance()


def fetch_ohlcv_data(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given symbol and timeframe and return a Pandas DataFrame.
    """
    try:
        # Fetch the last 200 candles
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
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
    """Raw 15m candles used to resolve open trades: the last 100, i.e. 25 hours (known P0 limit)."""
    exchange = create_exchange()
    return exchange.fetch_ohlcv(symbol, timeframe="15m", limit=100)
