"""Unit tests for the record-versioning helpers in btc_cli.storage (AGENTS.md §6)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from btc_cli import config, storage

MOMENT = datetime(2026, 10, 1, 12, 0, 5, tzinfo=timezone.utc)


def test_record_header_carries_versions_and_data_source():
    assert storage.record_header() == {
        "schema_version": config.SCHEMA_VERSION,
        "strategy_version": config.STRATEGY_VERSION,
        "exchange": config.EXCHANGE_ID,
        "market_type": config.MARKET_TYPE,
    }


def test_versions_are_set_for_the_warm_up_period():
    assert config.SCHEMA_VERSION == 1
    # "0.x" is warm-up; "1.0" is the first official version (owner decision 2026-10-07).
    assert config.STRATEGY_VERSION.startswith("0.")


def test_utc_iso_converts_any_offset_to_z():
    madrid = MOMENT.astimezone(timezone(timedelta(hours=2)))
    assert storage.utc_iso(madrid) == "2026-10-01T12:00:05Z"


def test_record_id_is_utc_based_and_names_strategy_and_symbol():
    assert storage.record_id(MOMENT, "defensive", "BTC/USDT") == "20261001T120005Z-DEF-BTCUSDT"
    assert storage.record_id(MOMENT, "greedy", "ETH/USDT") == "20261001T120005Z-GREED-ETHUSDT"


def test_local_now_is_the_same_instant_as_naive_local_time():
    local = storage.local_now(MOMENT)
    assert local.tzinfo is None
    assert local.astimezone(timezone.utc) == MOMENT


def test_legacy_records_read_as_schema_0_and_strategy_0_0():
    legacy_analysis = {"metadata": {"timestamp": "2026-10-02T22:34:43.1", "symbol": "BTC/USDT", "command_run": "analyze"}}
    legacy_ledger = {"symbol": "BTC/USDT", "status": "OPEN"}
    for record in (legacy_analysis, legacy_ledger, {}):
        assert storage.schema_version_of(record) == 0
        assert storage.strategy_version_of(record) == "0.0"


def test_versions_are_read_from_metadata_or_top_level():
    assert storage.schema_version_of({"metadata": {"schema_version": 1}}) == 1
    assert storage.strategy_version_of({"metadata": {"strategy_version": "0.1"}}) == "0.1"
    assert storage.schema_version_of({"schema_version": 1}) == 1
    assert storage.strategy_version_of({"strategy_version": "0.2"}) == "0.2"


def test_analysis_link_points_inside_the_data_folder():
    path = Path(config.BASE_DIR) / "analyze" / "defensive" / "2026-10" / "x_analysis.json"
    analysis = {"metadata": {"run_id": "20261001T120000Z-DEF-BTCUSDT", "models_used": {"agent_1_technical": "m"}}}
    assert storage.analysis_link(path, analysis) == {
        "analysis_file": "analyze/defensive/2026-10/x_analysis.json",
        "analysis_run_id": "20261001T120000Z-DEF-BTCUSDT",
        "models_used": {"agent_1_technical": "m"},
    }


def test_analysis_link_to_a_legacy_analysis_has_no_run_id_or_models():
    path = Path("elsewhere") / "x_analysis.json"
    assert storage.analysis_link(path, {"metadata": {"timestamp": "2026-10-02T22:34:43"}}) == {
        "analysis_file": "elsewhere/x_analysis.json",
        "analysis_run_id": None,
        "models_used": None,
    }


def test_closing_a_trade_leaves_legacy_history_entries_untouched(tmp_path):
    legacy_entry = {"symbol": "BTC/USDT", "status": "CLOSED", "result": "LOSS", "pnl_usd": -100.0}
    history = tmp_path / "history.json"
    history.write_text(json.dumps([legacy_entry]), encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}", encoding="utf-8")
    new_entry = {**storage.record_header(), "status": "CLOSED", "result": "WIN"}

    storage.move_to_history(str(ledger), str(history), new_entry)

    assert json.loads(history.read_text(encoding="utf-8")) == [legacy_entry, new_entry]
