"""Characterization: how an open paper trade is resolved into WIN/LOSS.

Several of these pin known Phase 1/2 bugs (TODO.md) exactly as they behave today:
the entry candle is skipped, a candle touching both levels is a LOSS, only the
last 100 15m candles (25 h) are checked. Damaged ledger and history files are kept
untouched and block the strategy (fixed on branch fix/safe-storage).
They will change on purpose when those bugs are fixed, with owner approval.
"""

import json

import ccxt
import pytest

from btc_cli.config import STRATEGY_VERSION
from tests.conftest import FROZEN_EPOCH
from tests.characterization.support import read_json

# Entry at 11:05 UTC, five minutes into the 11:00 candle and 55 minutes before the frozen clock.
ENTRY = FROZEN_EPOCH - 3300
ENTRY_CANDLE = FROZEN_EPOCH - 3600
NEXT_CANDLE = ENTRY_CANDLE + 900

LONG = {
    "symbol": "BTC/USDT",
    "status": "OPEN",
    "entry_timestamp": ENTRY,
    "verdict": "GO LONG",
    "order_type": "MARKET",
    "entry_price": 67000.0,
    "stop_loss": 66700.0,
    "take_profit": 67600.0,
    "risk_reward_ratio": 2.0,
    "position_size_usd": 22333.33,
}
SHORT = {**LONG, "verdict": "GO SHORT", "stop_loss": 67300.0, "take_profit": 66400.0}


def candle(open_ts: float, high: float, low: float) -> list[float]:
    return [int(open_ts * 1000), 67000.0, high, low, 67000.0, 100.0]


QUIET = (67050.0, 66950.0)


def write_ledger(tmp_path, strategy: str, content: dict | str):
    path = tmp_path / "output_alpha" / strategy / "BTC_USDT_paper_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content if isinstance(content, str) else json.dumps(content, indent=2)
    path.write_text(text, encoding="utf-8")
    return path


def history_path(tmp_path, strategy: str = "defensive"):
    return tmp_path / "output_alpha" / strategy / "BTC_USDT_trade_history.json"


@pytest.mark.parametrize(
    ("ledger", "hit", "result", "pnl"),
    [
        (LONG, (67600.0, 66950.0), "WIN", 200.0),
        (LONG, (67050.0, 66700.0), "LOSS", -100.0),
        (SHORT, (67050.0, 66400.0), "WIN", 200.0),
        (SHORT, (67300.0, 66950.0), "LOSS", -100.0),
    ],
    ids=["long-win", "long-loss", "short-win", "short-loss"],
)
def test_first_candle_touching_a_level_closes_the_trade(ledger, hit, result, pnl, run_cli, exchange, tmp_path):
    ledger_file = write_ledger(tmp_path, "defensive", ledger)
    exchange.candles["15m"] = [candle(NEXT_CANDLE, *QUIET), candle(NEXT_CANDLE + 900, *hit), candle(NEXT_CANDLE + 1800, 70000.0, 60000.0)]

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    assert not ledger_file.exists()
    # PnL is the move to the level times position size; close time is the hit candle's open time.
    # The ledger is in the pre-versioning format: it still resolves, and is not backfilled with version fields.
    assert read_json(history_path(tmp_path)) == [
        {
            **ledger, "status": "CLOSED", "result": result, "pnl_usd": pnl, "close_timestamp": NEXT_CANDLE + 900,
            "resolved_at_utc": "2026-10-01T12:00:00Z", "resolved_by_strategy_version": STRATEGY_VERSION,
        }
    ]
    assert f"Trade Closed (DEFENSIVE): {result}" in out.output
    assert exchange.calls == [{"symbol": "BTC/USDT", "timeframe": "15m", "since": None, "limit": 100}]


@pytest.mark.parametrize("ledger", [LONG, SHORT], ids=["long", "short"])
def test_candle_touching_both_levels_is_scored_loss(ledger, run_cli, exchange, tmp_path):
    exchange.candles["15m"] = [candle(NEXT_CANDLE, 67700.0, 66300.0)]
    write_ledger(tmp_path, "defensive", ledger)

    run_cli("operate", "--def")

    assert read_json(history_path(tmp_path))[0]["result"] == "LOSS"


def test_entry_candle_is_ignored_even_if_it_hits_a_level(run_cli, exchange, tmp_path):
    ledger_file = write_ledger(tmp_path, "defensive", LONG)
    before = ledger_file.read_bytes()
    exchange.candles["15m"] = [candle(ENTRY_CANDLE, 67050.0, 66000.0), candle(NEXT_CANDLE, *QUIET)]

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    assert ledger_file.read_bytes() == before
    assert not history_path(tmp_path).exists()
    assert "Active position detected for DEFENSIVE" in out.output


def test_only_the_last_100_candles_are_checked(run_cli, exchange, tmp_path):
    hit_then_quiet = [candle(NEXT_CANDLE, 67600.0, 66950.0)] + [candle(NEXT_CANDLE + 900 * i, *QUIET) for i in range(1, 101)]
    exchange.candles["15m"] = hit_then_quiet
    ledger_file = write_ledger(tmp_path, "defensive", LONG)

    run_cli("operate", "--def")

    assert ledger_file.exists()
    assert not history_path(tmp_path).exists()


def test_open_trade_blocks_the_strategy_and_prints_its_ticket(run_cli, exchange, gemini, tmp_path, snapshot):
    exchange.candles["15m"] = [candle(NEXT_CANDLE, *QUIET)]
    write_ledger(tmp_path, "defensive", LONG)
    write_ledger(tmp_path, "greedy", SHORT)

    out = run_cli("operate")

    assert out.exit_code == 0, out.output
    assert len(exchange.calls) == 2
    assert gemini.calls == []
    snapshot("resolution_open_console.txt", out.output)


def test_closed_trade_is_appended_to_existing_history(run_cli, exchange, tmp_path):
    old_trade = {**SHORT, "status": "CLOSED", "result": "WIN", "pnl_usd": 200.0, "close_timestamp": ENTRY - 7200}
    history_path(tmp_path).parent.mkdir(parents=True)
    history_path(tmp_path).write_text(json.dumps([old_trade]), encoding="utf-8")
    write_ledger(tmp_path, "defensive", LONG)
    exchange.candles["15m"] = [candle(NEXT_CANDLE, 67600.0, 66950.0)]

    run_cli("operate", "--def")

    history = read_json(history_path(tmp_path))
    assert [t["result"] for t in history] == ["WIN", "WIN"]
    assert history[0] == old_trade


@pytest.mark.parametrize("damaged", ["[{not valid json", '{"a JSON object": "not a list"}'], ids=["invalid-json", "not-a-list"])
def test_unreadable_history_is_kept_and_the_trade_stays_open(damaged, run_cli, exchange, tmp_path):
    """AGENTS.md §2.3: a damaged history keeps its bytes; nothing is written and the trade is not lost."""
    history_path(tmp_path).parent.mkdir(parents=True)
    history_path(tmp_path).write_text(damaged, encoding="utf-8")
    ledger_file = write_ledger(tmp_path, "defensive", LONG)
    ledger_before = ledger_file.read_bytes()
    exchange.candles["15m"] = [candle(NEXT_CANDLE, 67600.0, 66950.0)]

    out = run_cli("operate", "--def")

    assert out.exit_code == 1, out.output
    assert history_path(tmp_path).read_text(encoding="utf-8") == damaged
    assert ledger_file.read_bytes() == ledger_before
    assert "trade history file for DEFENSIVE can't be read" in out.output
    assert "New trades for DEFENSIVE are blocked" in out.output
    assert not (tmp_path / "output_alpha" / "operate").exists()


@pytest.mark.parametrize("damaged", ["{not valid json", "[1, 2]", ""], ids=["invalid-json", "not-an-object", "empty"])
def test_unreadable_ledger_is_kept_and_blocks_the_strategy(damaged, run_cli, exchange, tmp_path):
    """AGENTS.md §2.3: a damaged ledger keeps its bytes and is never treated as "no open trade"."""
    ledger_file = write_ledger(tmp_path, "defensive", damaged)

    out = run_cli("operate", "--def")

    assert out.exit_code == 1, out.output
    assert exchange.calls == []
    assert ledger_file.read_text(encoding="utf-8") == damaged
    assert "ledger file for DEFENSIVE can't be read" in out.output
    assert "No recent analysis found" not in out.output


@pytest.mark.parametrize("ledger", [{**LONG, "status": "CLOSED"}, {**LONG, "entry_timestamp": None}], ids=["not-open", "no-entry-time"])
def test_ledger_that_is_not_open_is_ignored(ledger, run_cli, exchange, tmp_path):
    write_ledger(tmp_path, "defensive", ledger)

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    assert exchange.calls == []


def test_exchange_error_while_checking_stops_with_exit_1(run_cli, exchange, tmp_path):
    ledger_file = write_ledger(tmp_path, "defensive", LONG)
    exchange.error = ccxt.NetworkError("simulated outage")

    out = run_cli("operate", "--def")

    assert out.exit_code == 1
    assert "Error fetching data to verify open positions" in out.output
    assert ledger_file.exists()
