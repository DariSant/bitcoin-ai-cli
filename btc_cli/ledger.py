"""WIN/LOSS resolution and PnL. Pure: takes a trade and candles, returns the result.

Moved unchanged from app.py. Known issues (fixed separately, see TODO.md Phase 1):
the candle containing the entry is skipped, a candle touching both levels is a
LOSS, and there are no fees, slippage or time limit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Exit:
    """The first candle that touched a level."""

    result: str  # "WIN" or "LOSS"
    close_timestamp: float  # that candle's open time, in seconds


def _levels(trade: dict) -> tuple[float, float, float, float]:
    """Entry, stop, target and size as floats; raises on a damaged value, as app.py always did."""
    return (
        float(trade.get("entry_price", 0)),
        float(trade.get("stop_loss", 0)),
        float(trade.get("take_profit", 0)),
        float(trade.get("position_size_usd", 0)),
    )


def find_exit(trade: dict, ohlcv: list[list]) -> Exit | None:
    """Scan candles in order and return the first stop or target hit after entry, or None."""
    entry_timestamp = trade.get("entry_timestamp")
    verdict = trade.get("verdict", "")
    _, stop_loss, take_profit, _ = _levels(trade)

    for candle in ohlcv:
        # candle: [timestamp, open, high, low, close, volume]
        ts = candle[0] / 1000.0  # Convert to seconds
        if ts <= entry_timestamp:
            continue

        high = float(candle[2])
        low = float(candle[3])

        # The stop is checked first, so a candle touching both levels is a LOSS.
        if verdict == "GO LONG":
            if low <= stop_loss:
                return Exit("LOSS", ts)
            elif high >= take_profit:
                return Exit("WIN", ts)
        elif verdict == "GO SHORT":
            if high >= stop_loss:
                return Exit("LOSS", ts)
            elif low <= take_profit:
                return Exit("WIN", ts)

    return None


def calculate_pnl(trade: dict, result: str) -> float:
    """PnL in USD for a trade closed exactly at its stop or target (no fees)."""
    verdict = trade.get("verdict", "")
    entry_price, stop_loss, take_profit, pos_size_usd = _levels(trade)

    if result == "WIN":
        if verdict == "GO LONG":
            return pos_size_usd * ((take_profit - entry_price) / entry_price)
        return pos_size_usd * ((entry_price - take_profit) / entry_price)
    # LOSS: strict % move to the stop, which is about -$100 by construction
    if verdict == "GO LONG":
        return pos_size_usd * ((stop_loss - entry_price) / entry_price)
    return pos_size_usd * ((entry_price - stop_loss) / entry_price)


def close_trade(trade: dict, exit_: Exit, pnl_usd: float) -> dict:
    """The history record: the open trade plus its result, rounded PnL and close time."""
    return {
        **trade,
        "status": "CLOSED",
        "result": exit_.result,
        "pnl_usd": round(pnl_usd, 2),
        "close_timestamp": exit_.close_timestamp,
    }
