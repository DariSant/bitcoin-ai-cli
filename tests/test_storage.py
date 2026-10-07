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


# --- Which analysis operate trades (TODO.md "operate scans every saved analysis") ---

MAX_AGE = 600


def local_to_utc(naive_local: datetime) -> datetime:
    """The UTC instant of a naive local wall time (how the CLI names its files)."""
    return naive_local.astimezone(timezone.utc)


def write_analysis(at_utc: datetime, symbol: str = "BTCUSDT", recorded_utc: datetime | None = None, legacy: bool = False) -> Path:
    """An analysis file named like the CLI names it (local time), recording `recorded_utc` (default: at_utc)."""
    local = storage.local_now(at_utc)
    folder = Path(config.BASE_DIR) / "analyze" / "defensive" / local.strftime("%Y-%m")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{local.strftime('%Y%m%d_%H%M%S')}_{symbol}_DEF_analysis.json"
    recorded = recorded_utc or at_utc
    metadata = {"timestamp": storage.local_now(recorded).isoformat()}
    if not legacy:
        metadata["timestamp_utc"] = storage.utc_iso(recorded)
    path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    return path


def test_an_analysis_from_just_before_midnight_on_the_last_day_of_the_month_is_found():
    now = local_to_utc(datetime(2026, 11, 1, 0, 3))
    written = write_analysis(now - timedelta(minutes=5))
    assert written.parent.name == storage.local_now(now - timedelta(minutes=5)).strftime("%Y-%m")

    assert storage.latest_analysis("defensive", "BTC/USDT", now, MAX_AGE) == written


def test_only_files_named_inside_the_window_are_opened(monkeypatch):
    now = local_to_utc(datetime(2026, 10, 20, 12, 0))
    for day in range(1, 19):
        write_analysis(local_to_utc(datetime(2026, 10, day, 9, 0)))
    recent = [write_analysis(now - timedelta(minutes=m)) for m in (2, 4)]
    opened = []
    original = storage.read_analysis
    monkeypatch.setattr(storage, "read_analysis", lambda p: opened.append(p) or original(p))

    assert storage.latest_analysis("defensive", "BTC/USDT", now, MAX_AGE) == recent[0]
    assert sorted(opened) == sorted(recent)


def test_the_recorded_time_wins_over_the_file_name():
    """E.g. the hour repeated when clocks go back: a later analysis can get an earlier-sorting name."""
    now = local_to_utc(datetime(2026, 10, 20, 12, 0))
    later_name_earlier_record = write_analysis(now - timedelta(minutes=2), recorded_utc=now - timedelta(minutes=8))
    earlier_name_later_record = write_analysis(now - timedelta(minutes=6), recorded_utc=now - timedelta(minutes=1))
    assert later_name_earlier_record.name > earlier_name_later_record.name

    assert storage.latest_analysis("defensive", "BTC/USDT", now, MAX_AGE) == earlier_name_later_record


def test_legacy_analyses_are_compared_by_their_local_timestamp():
    now = local_to_utc(datetime(2026, 10, 20, 12, 0))
    newer = write_analysis(now - timedelta(minutes=1), legacy=True)
    write_analysis(now - timedelta(minutes=5), legacy=True)

    assert storage.latest_analysis("defensive", "BTC/USDT", now, MAX_AGE) == newer


def test_when_nothing_is_recent_the_newest_by_name_is_returned_for_the_stale_message():
    now = local_to_utc(datetime(2026, 10, 20, 12, 0))
    write_analysis(now - timedelta(hours=5))
    newest = write_analysis(now - timedelta(hours=3))

    assert storage.latest_analysis("defensive", "BTC/USDT", now, MAX_AGE) == newest


def test_older_months_and_other_symbols_are_not_considered():
    now = local_to_utc(datetime(2026, 10, 20, 12, 0))
    write_analysis(local_to_utc(datetime(2026, 8, 31, 23, 59)))
    write_analysis(now - timedelta(minutes=1), symbol="ETHUSDT")

    assert storage.latest_analysis("defensive", "BTC/USDT", now, MAX_AGE) is None
