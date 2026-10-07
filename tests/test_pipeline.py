"""Every record the pipeline writes is versioned and traceable (AGENTS.md §6)."""

import json

from btc_cli import config
from tests.conftest import FALLBACK_MODEL, PRIMARY_MODEL
from tests.characterization.support import LONG_MAGNET, analysis_replies, files_under, read_json

HEADER = {
    "schema_version": config.SCHEMA_VERSION,
    "strategy_version": config.STRATEGY_VERSION,
    "exchange": config.EXCHANGE_ID,
    "market_type": config.MARKET_TYPE,
}


def only(tmp_path, pattern):
    matches = sorted(tmp_path.glob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


def test_status_record_is_versioned_without_models(run_cli, tmp_path):
    assert run_cli("status").exit_code == 0

    metadata = read_json(only(tmp_path, "output_alpha/status/system/*/*.json"))["metadata"]

    assert HEADER.items() <= metadata.items()
    assert metadata["timestamp_utc"] == "2026-10-01T12:00:00Z"
    assert metadata["run_id"] == "20261001T120000Z-SYSTEM-BTCUSDT"
    assert "models_used" not in metadata


def test_analysis_ticket_ledger_and_history_are_versioned_and_linked(run_cli, gemini, exchange, clock, tmp_path):
    gemini.script = analysis_replies("BULLISH", LONG_MAGNET, "GO LONG", None)
    assert run_cli("analyze", "--def").exit_code == 0
    assert gemini.script == []
    analysis_path = only(tmp_path, "output_alpha/analyze/defensive/*/*.json")
    analysis = read_json(analysis_path)["metadata"]
    clock.advance(60)
    assert run_cli("operate", "--def").exit_code == 0

    footprint = read_json(only(tmp_path, "output_alpha/operate/defensive/*/*.json"))["metadata"]
    ledger = read_json(tmp_path / "output_alpha" / "defensive" / "BTC_USDT_paper_ledger.json")

    assert HEADER.items() <= analysis.items()
    assert analysis["models_used"] == {"agent_1_technical": PRIMARY_MODEL, "agent_2_volume": PRIMARY_MODEL, "agent_3_defensive": PRIMARY_MODEL}
    assert analysis["models_configured"] == {"primary": PRIMARY_MODEL, "fallback": FALLBACK_MODEL}
    link = {
        "analysis_file": analysis_path.relative_to(tmp_path / "output_alpha").as_posix(),
        "analysis_run_id": analysis["run_id"],
        "models_used": analysis["models_used"],
    }
    for record in (footprint, ledger):
        assert HEADER.items() <= record.items()
        assert link.items() <= record.items()
        assert record["trade_id"] == "20261001T120100Z-DEF-BTCUSDT"
    assert footprint["timestamp_utc"] == ledger["entry_time_utc"] == "2026-10-01T12:01:00Z"

    # Close it: the target is hit in the next candle.
    exchange.candles["15m"] = [[int((ledger["entry_timestamp"] + 60) * 1000), 67000.0, ledger["take_profit"] + 1, ledger["entry_price"], 67000.0, 1.0]]
    clock.advance(60)
    assert run_cli("operate", "--def").exit_code == 0

    (closed,) = read_json(tmp_path / "output_alpha" / "defensive" / "BTC_USDT_trade_history.json")
    assert {k: v for k, v in closed.items() if k in ledger and k != "status"} == {k: v for k, v in ledger.items() if k != "status"}
    assert closed["resolved_at_utc"] == "2026-10-01T12:02:00Z"
    assert closed["resolved_by_strategy_version"] == config.STRATEGY_VERSION


def test_models_used_names_the_model_for_each_agent_call(run_cli, gemini, tmp_path):
    """Only Agent 2 falls back; the record must say so per call, not per run."""
    a1, a2, defensive, greedy = analysis_replies("BULLISH", LONG_MAGNET, "GO LONG", "GO LONG")
    gemini.script = [a1, RuntimeError("primary unavailable"), a2, defensive, greedy]

    assert run_cli("analyze").exit_code == 0

    for strategy in ("defensive", "greedy"):
        metadata = read_json(only(tmp_path, f"output_*/analyze/{strategy}/*/*.json"))["metadata"]
        assert metadata["models_used"] == {
            "agent_1_technical": PRIMARY_MODEL,
            "agent_2_volume": FALLBACK_MODEL,
            f"agent_3_{strategy}": PRIMARY_MODEL,
        }


def test_a_fresh_legacy_analysis_still_trades_with_an_empty_link(run_cli, clock, tmp_path):
    analysis = {
        "metadata": {"timestamp": clock.local_naive().isoformat(), "symbol": "BTC/USDT", "command_run": "analyze"},
        "raw_market_data": {"15m": {"price": 67000.0, "atr_14": 140.0, "poc_price": 67500.0, "calculated_support": 66900.0, "calculated_resistance": 67300.0}},
        "agent_2_volume": {"magnet_target": "TARGET: 67600.00 | DISTANCE: 0.9%"},
        "agent_3_synthesis": {"final_verdict": "GO LONG"},
    }
    path = tmp_path / "output_alpha" / "analyze" / "defensive" / "2026-10" / "legacy_BTCUSDT_DEF_analysis.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(analysis), encoding="utf-8")

    assert run_cli("operate", "--def").exit_code == 0

    ledger = read_json(tmp_path / "output_alpha" / "defensive" / "BTC_USDT_paper_ledger.json")
    assert HEADER.items() <= ledger.items()
    assert ledger["analysis_file"] == "analyze/defensive/2026-10/legacy_BTCUSDT_DEF_analysis.json"
    assert ledger["analysis_run_id"] is None and ledger["models_used"] is None
    # The legacy analysis itself is only read, never rewritten.
    assert read_json(path) == analysis
    assert "output_alpha/defensive/BTC_USDT_paper_ledger.json" in files_under(tmp_path)


def _damage_defensive_ledger(tmp_path):
    damaged = tmp_path / "output_alpha" / "defensive" / "BTC_USDT_paper_ledger.json"
    damaged.parent.mkdir(parents=True, exist_ok=True)
    damaged.write_text("{broken", encoding="utf-8")
    return damaged


def test_a_damaged_ledger_blocks_only_its_own_strategy_and_operate_exits_1(run_cli, gemini, clock, tmp_path):
    gemini.script = analysis_replies("BULLISH", LONG_MAGNET, "GO LONG", "GO LONG")
    assert run_cli("analyze").exit_code == 0
    damaged = _damage_defensive_ledger(tmp_path)
    clock.advance(60)

    out = run_cli("operate")

    assert out.exit_code == 1, out.output
    assert damaged.read_text(encoding="utf-8") == "{broken"
    assert not (tmp_path / "output_alpha" / "operate" / "defensive").exists()
    assert read_json(tmp_path / "output_alpha" / "greedy" / "BTC_USDT_paper_ledger.json")["status"] == "OPEN"


def test_analyze_still_runs_the_healthy_strategy_then_exits_1(run_cli, gemini, tmp_path):
    damaged = _damage_defensive_ledger(tmp_path)
    gemini.script = analysis_replies("BULLISH", LONG_MAGNET, None, "GO LONG")

    out = run_cli("analyze")

    assert out.exit_code == 1, out.output
    assert gemini.script == []
    assert [f for f in files_under(tmp_path) if "/analyze/" in f] == [only(tmp_path, "output_alpha/analyze/greedy/*/*.json").relative_to(tmp_path).as_posix()]
    assert damaged.read_text(encoding="utf-8") == "{broken"


def test_analyze_with_every_strategy_damaged_makes_no_ai_calls(run_cli, gemini, tmp_path):
    _damage_defensive_ledger(tmp_path)

    out = run_cli("analyze", "--def")

    assert out.exit_code == 1, out.output
    assert gemini.calls == []
    assert "All requested strategies have open positions" not in out.output


def test_auto_runs_every_step_for_the_healthy_strategy_then_exits_1(run_cli, gemini, tmp_path):
    _damage_defensive_ledger(tmp_path)
    gemini.script = analysis_replies("BULLISH", LONG_MAGNET, None, "GO LONG")

    out = run_cli("auto")

    assert out.exit_code == 1, out.output
    assert read_json(tmp_path / "output_alpha" / "greedy" / "BTC_USDT_paper_ledger.json")["status"] == "OPEN"
    assert (tmp_path / "output_alpha" / "defensive" / "BTC_USDT_paper_ledger.json").read_text(encoding="utf-8") == "{broken"
