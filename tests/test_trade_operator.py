"""Unit tests for btc_cli.trade_operator (today's math, unchanged by the split)."""

import pytest

from btc_cli.trade_operator import OrderCalc, build_operator_report, compute_order, parse_magnet_target


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("TARGET: 67450.00 | DISTANCE: 0.71%", 67450.0),
        ("TARGET:67450", 67450.0),
        ("TARGET: $67,450 | DISTANCE: 0.71%", None),
        ("67450", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_magnet_target(raw, expected):
    assert parse_magnet_target(raw) == expected


def test_long_uses_the_lower_of_structural_and_one_atr_stop():
    # Structural stop 68500 - 500 = 68000 is below the 1-ATR stop 69000, so it wins.
    order = compute_order("GO LONG", 70000.0, 1000.0, 68500.0, 74000.0)

    assert order == OrderCalc(True, 68000.0, 74000.0, 2.0, 3500.0)


def test_long_one_atr_floor_applies_when_structure_is_close():
    order = compute_order("GO LONG", 70000.0, 1000.0, 69800.0, 74000.0)

    assert order.stop_loss == 69000.0
    assert order.risk_reward_ratio == 4.0
    assert order.position_size_usd == 7000.0  # $100 risk / $1000 distance * price


def test_short_uses_the_higher_of_structural_and_one_atr_stop():
    order = compute_order("GO SHORT", 70000.0, 1000.0, 70200.0, 66000.0)

    assert order == OrderCalc(True, 71000.0, 66000.0, 4.0, 7000.0)


@pytest.mark.parametrize(
    ("verdict", "magnet"),
    [("GO LONG", 69000.0), ("GO LONG", 70000.0), ("GO SHORT", 71000.0), ("GO SHORT", 70000.0)],
)
def test_target_on_the_wrong_side_is_invalid(verdict, magnet):
    order = compute_order(verdict, 70000.0, 1000.0, 70000.0, magnet)

    assert order.valid is False
    assert order.take_profit == magnet
    assert order.risk_reward_ratio is None


def test_unknown_verdict_gives_no_order():
    assert compute_order("SIT ON HANDS", 70000.0, 1000.0, 68500.0, 74000.0) is None


def test_the_one_atr_floor_sets_the_stop_when_the_threat_is_close():
    """The case where the old mock math disagreed with operate (fixed 2026-10-07: mock now uses compute_order)."""
    order = compute_order("GO LONG", 70000.0, 1000.0, 69800.0, 74000.0)

    assert order.stop_loss == 69000.0


def test_operator_report_rounds_to_cents():
    order = OrderCalc(True, 66732.115, 67450.0, 1.9361, 27392.7409)

    assert build_operator_report(66976.624, order) == {
        "order_type": "MARKET",
        "entry_price": 66976.62,
        "stop_loss": 66732.12,
        "take_profit": 67450.0,
        "risk_reward_ratio": 1.94,
        "position_size_usd": 27392.74,
    }
