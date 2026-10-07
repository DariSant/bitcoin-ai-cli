"""Characterization: the Python Operator inside `operate` (ticket math, rejections, freshness)."""

import json
import os

from tests.characterization.support import LONG_MAGNET, analysis_replies, read_json

LEDGER = ("output_alpha", "defensive", "BTC_USDT_paper_ledger.json")


def analyze_def(run_cli, gemini, verdict: str, magnet: str, bias: str = "BULLISH"):
    """Write one defensive analysis through the real `analyze --def` command."""
    gemini.script = analysis_replies(bias, magnet, defensive=verdict, greedy=None)
    result = run_cli("analyze", "--def")
    assert result.exit_code == 0, result.output
    return result


def test_long_ticket_with_target_below_price_is_rejected_and_logged(run_cli, gemini, clock, tmp_path, snapshot):
    analyze_def(run_cli, gemini, "GO LONG", "TARGET: 66000.00 | DISTANCE: -1.46%")

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    assert not tmp_path.joinpath(*LEDGER).exists()
    assert not (tmp_path / "output_alpha" / "operate").exists()
    log = (tmp_path / "output_alpha" / "operator_errors.log").read_text(encoding="utf-8")
    snapshot("operate_invalid_long_errors.log", log)
    snapshot("operate_invalid_long_console.txt", out.output)


def test_short_ticket_with_target_above_price_is_rejected_and_logged(run_cli, gemini, tmp_path):
    analyze_def(run_cli, gemini, "GO SHORT", "TARGET: 68000.00 | DISTANCE: 1.53%", bias="BEARISH")

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    assert not tmp_path.joinpath(*LEDGER).exists()
    log = (tmp_path / "output_alpha" / "operator_errors.log").read_text(encoding="utf-8")
    assert log == (
        "[2026-10-01 12:00:00 UTC] ERROR: INVALID TICKET LOGIC | Verdict: GO SHORT | Price: 66976.62 | "
        "Threat: 67282.57 | Magnet: 68000.0 | SL: 67352.845 | TP: 68000.0\n"
    )


def test_unparsable_magnet_falls_back_to_poc(run_cli, gemini, tmp_path):
    analyze_def(run_cli, gemini, "GO SHORT", "TARGET: $66,500 | DISTANCE: 0.71%", bias="BEARISH")

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    assert "Could not parse Agent 2 Magnet" in out.output
    ledger = read_json(tmp_path.joinpath(*LEDGER))
    assert ledger["take_profit"] == 64475.44  # 15m POC
    assert ledger["stop_loss"] == 67352.85


def test_analysis_exactly_600_seconds_old_is_still_traded(run_cli, gemini, clock, tmp_path):
    analyze_def(run_cli, gemini, "GO LONG", LONG_MAGNET)
    clock.advance(600)

    run_cli("operate", "--def")

    assert tmp_path.joinpath(*LEDGER).exists()


def test_analysis_older_than_600_seconds_is_skipped(run_cli, gemini, clock, tmp_path):
    analyze_def(run_cli, gemini, "GO LONG", LONG_MAGNET)
    clock.advance(601)

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    assert not tmp_path.joinpath(*LEDGER).exists()
    assert "No recent analysis has been run in the last 10 minutes for DEFENSIVE." in out.output


def test_same_analysis_can_be_traded_twice(run_cli, gemini, exchange, clock, tmp_path):
    """Known P1 bug: once the first trade closes, the same fresh analysis opens a second one."""
    analyze_def(run_cli, gemini, "GO LONG", LONG_MAGNET)
    run_cli("operate", "--def")
    first = read_json(tmp_path.joinpath(*LEDGER))
    exchange.candles["15m"] = [[int((first["entry_timestamp"] + 60) * 1000), 67000.0, 67500.0, 66950.0, 67000.0, 100.0]]
    clock.advance(240)

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    history = read_json(tmp_path / "output_alpha" / "defensive" / "BTC_USDT_trade_history.json")
    assert [t["result"] for t in history] == ["WIN"]
    second = read_json(tmp_path.joinpath(*LEDGER))
    assert second["status"] == "OPEN"
    assert second["entry_timestamp"] == clock.epoch
    entry_time_fields = ("entry_timestamp", "entry_time_utc", "trade_id")
    assert {k: v for k, v in second.items() if k not in entry_time_fields} == {k: v for k, v in first.items() if k not in entry_time_fields}
    # Since records are versioned, the duplicate is at least visible: two trades, one analysis.
    assert second["trade_id"] != first["trade_id"]
    assert second["analysis_run_id"] == first["analysis_run_id"]


def test_operate_requires_gemini_key_it_never_uses(run_cli, gemini, monkeypatch, tmp_path):
    """Known P2 issue: `operate` makes no AI calls but still exits 1 without GEMINI_API_KEY."""
    monkeypatch.delenv("GEMINI_API_KEY")

    out = run_cli("operate")

    assert out.exit_code == 1
    assert "GEMINI_API_KEY environment variable is missing" in out.output
    assert gemini.calls == []


def test_operate_picks_the_analysis_by_file_mtime_not_by_name(run_cli, gemini, clock, tmp_path):
    analyze_def(run_cli, gemini, "SIT ON HANDS", LONG_MAGNET)
    clock.advance(60)
    analyze_def(run_cli, gemini, "GO LONG", LONG_MAGNET)
    older_by_name = sorted((tmp_path / "output_alpha" / "analyze" / "defensive").rglob("*.json"))[0]
    future = older_by_name.stat().st_mtime + 100
    os.utime(older_by_name, (future, future))

    out = run_cli("operate", "--def")

    assert "Defensive Verdict: SIT ON HANDS. Bypassing execution." in out.output
    assert not tmp_path.joinpath(*LEDGER).exists()


def test_flags_select_one_strategy_and_both_flags_are_refused(run_cli, gemini, tmp_path):
    gemini.script = analysis_replies("BULLISH", LONG_MAGNET, defensive=None, greedy="GO LONG")

    greed_only = run_cli("analyze", "--greed")
    both = run_cli("operate", "--def", "--greed")

    assert greed_only.exit_code == 0, greed_only.output
    assert [c["response_schema"] for c in gemini.calls] == ["Agent1TechSchema", "Agent2VolumeSchema", "Agent3ManagerSchema"]
    assert not (tmp_path / "output_alpha" / "analyze" / "defensive").exists()
    assert both.exit_code == 1
    assert "Cannot pass both flags" in both.output


def test_analyze_skips_ai_when_every_strategy_has_an_open_trade(run_cli, gemini, exchange, tmp_path):
    for strategy in ("defensive", "greedy"):
        path = tmp_path / "output_alpha" / strategy / "BTC_USDT_paper_ledger.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"status": "OPEN", "entry_timestamp": 1.0, "verdict": "GO LONG", "entry_price": 1.0, "stop_loss": 0.0, "take_profit": 1e9, "position_size_usd": 1.0}), encoding="utf-8")
    exchange.candles["15m"] = []

    out = run_cli("analyze")

    assert out.exit_code == 0, out.output
    assert "All requested strategies have open positions. Skipping AI analysis" in out.output
    assert gemini.calls == []
    assert [c["timeframe"] for c in exchange.calls] == ["15m", "15m"]
