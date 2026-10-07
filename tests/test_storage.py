"""Unit tests for the record-versioning helpers in btc_cli.storage (AGENTS.md §6)."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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


# --- Atomic writes and damaged files (AGENTS.md §2.3, §5) ---

def test_write_json_atomic_writes_the_same_bytes_as_a_plain_dump(tmp_path):
    target = tmp_path / "ledger.json"
    data = {"price": 67000.5, "note": "ünïcode"}

    storage.write_json_atomic(target, data)

    assert target.read_text(encoding="utf-8") == json.dumps(data, indent=2)
    assert [p.name for p in tmp_path.iterdir()] == ["ledger.json"]


def test_a_crash_during_replace_keeps_the_old_file_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "ledger.json"
    target.write_text('{"old": true}', encoding="utf-8")

    def crash(*args, **kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(storage.os, "replace", crash)
    with pytest.raises(OSError, match="simulated crash"):
        storage.write_json_atomic(target, {"new": True})

    assert target.read_text(encoding="utf-8") == '{"old": true}'
    assert [p.name for p in tmp_path.iterdir()] == ["ledger.json"]


def _ledger_and_history(tmp_path, history: str | None):
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}", encoding="utf-8")
    history_file = tmp_path / "history.json"
    if history is not None:
        history_file.write_text(history, encoding="utf-8")
    return ledger, history_file


def test_a_trade_already_in_history_is_not_appended_twice(tmp_path):
    """Crash between 'append to history' and 'delete ledger': the next run must not duplicate the trade."""
    closed = {"trade_id": "20261001T120000Z-DEF-BTCUSDT", "result": "WIN"}
    ledger, history_file = _ledger_and_history(tmp_path, json.dumps([closed]))

    storage.move_to_history(str(ledger), str(history_file), closed)

    assert json.loads(history_file.read_text(encoding="utf-8")) == [closed]
    assert not ledger.exists()


def test_a_legacy_trade_without_trade_id_is_not_appended_twice(tmp_path):
    closed = {"symbol": "BTC/USDT", "entry_timestamp": 1.0, "verdict": "GO LONG", "entry_price": 67000.0, "result": "WIN"}
    other = {**closed, "entry_timestamp": 2.0}
    ledger, history_file = _ledger_and_history(tmp_path, json.dumps([closed]))

    storage.move_to_history(str(ledger), str(history_file), closed)
    assert json.loads(history_file.read_text(encoding="utf-8")) == [closed]

    ledger.write_text("{}", encoding="utf-8")
    storage.move_to_history(str(ledger), str(history_file), other)
    assert json.loads(history_file.read_text(encoding="utf-8")) == [closed, other]


def test_a_damaged_history_raises_before_anything_changes(tmp_path):
    ledger, history_file = _ledger_and_history(tmp_path, "[{broken")

    with pytest.raises(storage.DamagedRecordError) as raised:
        storage.move_to_history(str(ledger), str(history_file), {"trade_id": "x"})

    assert raised.value.path == str(history_file)

    assert history_file.read_text(encoding="utf-8") == "[{broken"
    assert ledger.exists()


def test_read_ledger_distinguishes_missing_from_damaged(tmp_path):
    path = tmp_path / "ledger.json"
    assert storage.read_ledger(str(path)) is None
    path.write_text('"just a string"', encoding="utf-8")
    with pytest.raises(storage.DamagedRecordError, match="expected a JSON dict"):
        storage.read_ledger(str(path))


# --- Run lock (AGENTS.md §5) ---

# A separate process that takes the lock for the data folder in argv[1], says "locked", then
# either holds it until its stdin closes, or dies abruptly without releasing it ("crash").
HOLDER = """
import os, sys
os.environ["BTC_CLI_DATA_DIR"] = sys.argv[1]
from btc_cli import storage
with storage.run_lock("holder process"):
    print("locked", os.getpid(), flush=True)
    if sys.argv[2] == "crash":
        os._exit(9)
    sys.stdin.readline()
"""


def start_holder(data_dir, mode: str) -> tuple[subprocess.Popen, int]:
    """The process and its real PID (on Windows a venv's python.exe is a launcher with its own PID)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(data_dir), mode],
        cwd=config.PROJECT_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    word, pid = proc.stdout.readline().split()
    assert word == "locked"
    return proc, int(pid)


@pytest.fixture
def lock_in(tmp_path, monkeypatch):
    """Point the run lock at tmp_path, where a holder process with BTC_CLI_DATA_DIR=tmp_path also looks."""
    monkeypatch.setattr(config, "LOCK_FILE", tmp_path / "run.lock")
    return tmp_path


def test_the_lock_records_its_holder_and_the_file_stays(lock_in):
    with storage.run_lock("status"):
        pass

    text = (lock_in / "run.lock").read_text(encoding="utf-8")
    assert f"pid {os.getpid()}, command 'status', started " in text


def test_a_second_lock_in_the_same_data_folder_is_refused(lock_in):
    with storage.run_lock("first"):
        with pytest.raises(storage.RunLockedError) as raised:
            with storage.run_lock("second"):
                pass
    assert "command 'first'" in raised.value.holder
    with storage.run_lock("third"):
        pass


def test_a_lock_held_by_another_process_is_refused_until_it_ends(lock_in):
    holder, holder_pid = start_holder(lock_in, "hold")
    try:
        with pytest.raises(storage.RunLockedError) as raised:
            with storage.run_lock("second"):
                pass
        assert f"pid {holder_pid}, command 'holder process'" in raised.value.holder
    finally:
        holder.stdin.close()
        assert holder.wait(timeout=30) == 0

    with storage.run_lock("after the holder ended"):
        pass


def test_a_crashed_holder_leaves_no_stale_lock(lock_in):
    holder, _ = start_holder(lock_in, "crash")
    assert holder.wait(timeout=30) == 9
    assert (lock_in / "run.lock").exists()

    with storage.run_lock("after the crash"):
        pass
