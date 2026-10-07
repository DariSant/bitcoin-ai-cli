"""Unit tests for btc_cli.ledger (today's resolution rules, unchanged by the split)."""

import pytest

from btc_cli.ledger import Exit, calculate_pnl, close_trade, find_exit

ENTRY = 1_000_000.0  # seconds

LONG = {
    "status": "OPEN",
    "entry_timestamp": ENTRY,
    "verdict": "GO LONG",
    "entry_price": 100.0,
    "stop_loss": 90.0,
    "take_profit": 120.0,
    "position_size_usd": 1000.0,
}
SHORT = {**LONG, "verdict": "GO SHORT", "stop_loss": 110.0, "take_profit": 80.0}


def candle(offset_s: float, high: float, low: float) -> list[float]:
    return [(ENTRY + offset_s) * 1000, 100.0, high, low, 100.0, 1.0]


def test_no_hit_returns_none():
    assert find_exit(LONG, [candle(60, 101, 99), candle(960, 119, 91)]) is None


def test_first_hit_wins_and_close_time_is_the_candle_open():
    candles = [candle(60, 101, 99), candle(960, 120, 99), candle(1860, 101, 80)]

    assert find_exit(LONG, candles) == Exit("WIN", ENTRY + 960)


def test_candles_at_or_before_entry_are_skipped():
    assert find_exit(LONG, [candle(-300, 130, 80), candle(0, 130, 80)]) is None


@pytest.mark.parametrize("trade", [LONG, SHORT], ids=["long", "short"])
def test_candle_touching_both_levels_is_a_loss(trade):
    assert find_exit(trade, [candle(60, 130, 70)]).result == "LOSS"


def test_short_win_and_loss():
    assert find_exit(SHORT, [candle(60, 101, 80)]).result == "WIN"
    assert find_exit(SHORT, [candle(60, 110, 99)]).result == "LOSS"


def test_damaged_level_raises_even_without_a_hit():
    with pytest.raises(ValueError):
        find_exit({**LONG, "position_size_usd": "n/a"}, [])


@pytest.mark.parametrize(
    ("trade", "result", "pnl"),
    [(LONG, "WIN", 200.0), (LONG, "LOSS", -100.0), (SHORT, "WIN", 200.0), (SHORT, "LOSS", -100.0)],
)
def test_pnl_is_the_move_to_the_level_times_size(trade, result, pnl):
    assert calculate_pnl(trade, result) == pytest.approx(pnl)


def test_close_trade_keeps_fields_in_order_and_rounds_pnl():
    closed = close_trade(LONG, Exit("WIN", ENTRY + 960), 199.999)

    assert list(closed) == [*LONG, "result", "pnl_usd", "close_timestamp"]
    assert closed["status"] == "CLOSED"
    assert closed["pnl_usd"] == 200.0
    assert LONG["status"] == "OPEN", "the open trade passed in is not modified"
