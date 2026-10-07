"""Output paths and every file the program writes: analyses, footprints, ledgers, history, logs.

Every path is built from config.BASE_DIR / config.DATA_DIR, which are absolute (AGENTS.md §5).
JSON records are written atomically, and a recorded file that can't be read is never
overwritten (AGENTS.md §2.3). Known remaining issue (TODO.md): file names use naive local time.
"""

import contextlib
import json
import os
import pathlib
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

from btc_cli import config

if os.name == "nt":
    import msvcrt
else:
    import fcntl

# What readers assume for records written before versioning (AGENTS.md §6: legacy, warm-up).
LEGACY_SCHEMA_VERSION = 0
LEGACY_STRATEGY_VERSION = "0.0"


class DamagedRecordError(Exception):
    """A recorded file exists but can't be read. It is left untouched for the owner to inspect."""

    def __init__(self, path: str | os.PathLike, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = str(path)
        self.reason = reason


def write_json_atomic(path: str | os.PathLike, data: dict | list) -> None:
    """Write JSON so a crash leaves either the old file or the new one, never half of one.

    The temp file sits in the same folder, because os.replace is only atomic within one file system.
    """
    target = pathlib.Path(path)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(temp, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, target)
    finally:
        # Only left behind if the write or the replace failed.
        temp.unlink(missing_ok=True)


def _strategy_prefix(strategy: str) -> str:
    return "DEF" if strategy == "defensive" else "GREED" if strategy == "greedy" else strategy.upper()


# --- Record versioning (AGENTS.md §6) ---

def utc_iso(moment: datetime) -> str:
    """An aware datetime as an ISO 8601 UTC string ending in Z."""
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def record_id(now_utc: datetime, strategy: str, symbol: str) -> str:
    """A readable unique id such as 20261001T120000Z-DEF-BTCUSDT (one record per strategy per second)."""
    return f"{now_utc.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{_strategy_prefix(strategy)}-{symbol.replace('/', '')}"


def record_header() -> dict:
    """The version and data-source fields every record carries."""
    return {
        "schema_version": config.SCHEMA_VERSION,
        "strategy_version": config.STRATEGY_VERSION,
        "exchange": config.EXCHANGE_ID,
        "market_type": config.MARKET_TYPE,
    }


def _version_field(record: dict, key: str, default):
    """A field from `metadata` (analysis, footprint) or the top level (ledger, history)."""
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return record.get(key, default)


def schema_version_of(record: dict) -> int:
    """The record's schema version; 0 for legacy records."""
    return _version_field(record, "schema_version", LEGACY_SCHEMA_VERSION)


def strategy_version_of(record: dict) -> str:
    """The strategy version the record was written under; "0.0" for legacy records."""
    return _version_field(record, "strategy_version", LEGACY_STRATEGY_VERSION)


def display_path(path: str | os.PathLike) -> str:
    """A path as shown to the user: relative to the data folder when inside it, e.g. output_alpha/defensive/..."""
    path = pathlib.Path(path)
    try:
        return path.relative_to(config.DATA_DIR).as_posix()
    except ValueError:
        return str(path)


def local_now(now_utc: datetime) -> datetime:
    """The naive local wall time of the same instant, as `datetime.now()` gives it (file names, legacy fields)."""
    return now_utc.astimezone().replace(tzinfo=None)


# --- Analyses (and the status footprint) ---

def log_execution(command_name: str, strategy: str, symbol: str, data_4h: dict, data_15m: dict, agent1_report: dict = None, agent2_report: dict = None, agent3_report: dict = None, models_used: dict[str, str] | None = None) -> pathlib.Path:
    """
    Universally log execution state to a JSON footprint in BASE_DIR/.
    `models_used` maps each agent call to the model that answered it; None when no AI ran (status).
    """
    # One instant for the file name, the legacy local `timestamp` and the UTC fields.
    now_utc = datetime.now(timezone.utc)
    now = local_now(now_utc)
    directory = pathlib.Path(config.BASE_DIR) / command_name / strategy / now.strftime('%Y-%m')
    directory.mkdir(parents=True, exist_ok=True)

    strategy_prefix = _strategy_prefix(strategy)
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{symbol.replace('/', '')}_{strategy_prefix}_analysis.json"
    filepath = directory / filename

    payload = {
        "metadata": {
            "timestamp": now.isoformat(),
            "symbol": symbol,
            "command_run": command_name,
            **record_header(),
            "timestamp_utc": utc_iso(now_utc),
            "strategy": strategy,
            "run_id": record_id(now_utc, strategy, symbol),
        },
        "raw_market_data": {
            "4h": data_4h,
            "15m": data_15m
        }
    }

    if models_used is not None:
        payload["metadata"]["models_used"] = models_used
        payload["metadata"]["models_configured"] = {"primary": config.PRIMARY_MODEL, "fallback": config.FALLBACK_MODEL}

    # Only add AI reports if they were passed to the function
    if agent1_report: payload["agent_1_technical"] = agent1_report
    if agent2_report: payload["agent_2_volume"] = agent2_report
    if agent3_report: payload["agent_3_synthesis"] = agent3_report

    write_json_atomic(filepath, payload)

    return filepath


# File names start with the naive local time they were written; this margin covers a clock change.
_NAME_TIME_MARGIN = timedelta(hours=1)


def _recorded_time(path: pathlib.Path) -> datetime:
    """When an analysis says it was written, as an aware UTC time; the earliest possible time if unreadable.

    Uses `timestamp_utc`, or the legacy local `timestamp` for records written before versioning.
    """
    try:
        metadata = read_analysis(path).get("metadata", {})
        if metadata.get("timestamp_utc"):
            return datetime.fromisoformat(metadata["timestamp_utc"].replace("Z", "+00:00"))
        stamp = datetime.fromisoformat(metadata["timestamp"])
        return stamp if stamp.tzinfo else stamp.astimezone()
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        # operate reads the chosen file again and reports a damaged one; here it just loses the comparison.
        return datetime.min.replace(tzinfo=timezone.utc)


def latest_analysis(strategy: str, symbol: str, now_utc: datetime, max_age_seconds: int) -> pathlib.Path | None:
    """The symbol's newest analysis by its own recorded time, or None if there is none recently.

    Bounded work on the server (TODO.md "operate scans every saved analysis"): only the month folders
    the freshness window touches are listed (at most two), and only files whose name falls inside the
    window (plus an hour for clock changes) are opened. If none is that recent, the newest by name is
    returned, so operate can report it as stale. File modification times are never used: a backup
    restore or copy changes them.
    """
    now_local = local_now(now_utc)
    oldest_local = local_now(now_utc - timedelta(seconds=max_age_seconds))
    base = pathlib.Path(config.BASE_DIR) / "analyze" / strategy
    months = sorted({oldest_local.strftime("%Y-%m"), now_local.strftime("%Y-%m")})
    files = [f for month in months for f in (base / month).glob("*.json") if symbol.replace("/", "") in f.name]
    if not files:
        return None
    cutoff = (oldest_local - _NAME_TIME_MARGIN).strftime("%Y%m%d_%H%M%S")
    recent = [f for f in files if f.name[:15] >= cutoff]
    if not recent:
        return max(files, key=lambda f: f.name)
    return max(recent, key=lambda f: (_recorded_time(f), f.name))


def read_analysis(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analysis_link(path: pathlib.Path, analysis: dict) -> dict:
    """Fields that tie a ticket or trade to the analysis that opened it (None for legacy analyses)."""
    try:
        relative = pathlib.Path(path).relative_to(config.BASE_DIR).as_posix()
    except ValueError:
        relative = path.as_posix()
    metadata = analysis.get("metadata", {})
    return {
        "analysis_file": relative,
        "analysis_run_id": metadata.get("run_id"),
        "models_used": metadata.get("models_used"),
    }


# --- Operator outputs ---

def write_execution_footprint(now: datetime, now_utc: datetime, strategy: str, symbol: str, operator_payload: dict, operator_report: dict, source: dict) -> pathlib.Path:
    """Save the operator's inputs and ticket under BASE_DIR/operate/; returns the path.

    `now` is the naive local time used in the file name, `now_utc` the same instant,
    and `source` links to the analysis (see `analysis_link`).
    """
    directory = pathlib.Path(config.BASE_DIR) / "operate" / strategy / now.strftime('%Y-%m')
    directory.mkdir(parents=True, exist_ok=True)

    strategy_prefix = "DEF" if strategy == "defensive" else "GREED"
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{symbol.replace('/', '')}_{strategy_prefix}_EXECUTION.json"
    filepath = directory / filename

    execution_footprint = {
        "metadata": {
            "timestamp": now.isoformat(),
            "symbol": symbol,
            "command_run": "operate",
            "strategy": strategy,
            **record_header(),
            "timestamp_utc": utc_iso(now_utc),
            "trade_id": record_id(now_utc, strategy, symbol),
            **source,
        },
        "operator_payload": operator_payload,
        "operator_execution": operator_report
    }

    write_json_atomic(filepath, execution_footprint)

    return filepath


def append_operator_error(log_entry: str) -> None:
    """Record a rejected ticket in BASE_DIR/operator_errors.log."""
    log_path = pathlib.Path(config.BASE_DIR) / "operator_errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)


# --- Ledger and history ---

def ensure_strategy_dir(strategy: str) -> None:
    (pathlib.Path(config.BASE_DIR) / strategy).mkdir(parents=True, exist_ok=True)


def ledger_path(strategy: str, symbol: str) -> pathlib.Path:
    clean_symbol = symbol.replace("/", "_")
    return pathlib.Path(config.BASE_DIR) / strategy / f"{clean_symbol}_paper_ledger.json"


def history_path(strategy: str, symbol: str) -> pathlib.Path:
    clean_symbol = symbol.replace("/", "_")
    return pathlib.Path(config.BASE_DIR) / strategy / f"{clean_symbol}_trade_history.json"


def _read_recorded_json(path: str | os.PathLike, expected: type) -> dict | list:
    """Parse a recorded file; raise DamagedRecordError if it isn't valid JSON of the expected type."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise DamagedRecordError(path, str(e)) from e
    if not isinstance(content, expected):
        raise DamagedRecordError(path, f"expected a JSON {expected.__name__}, found {type(content).__name__}")
    return content


def read_ledger(path: str | os.PathLike) -> dict | None:
    """The ledger, or None if there is none. A damaged ledger raises DamagedRecordError and is left untouched."""
    if not os.path.exists(path):
        return None
    return _read_recorded_json(path, dict)


def write_ledger(path: str | os.PathLike, ledger_entry: dict) -> None:
    write_json_atomic(path, ledger_entry)


def _trade_key(trade: dict) -> tuple:
    """Identity of a trade: its trade_id, or entry details for legacy trades written before trade_id existed."""
    if trade.get("trade_id"):
        return ("trade_id", trade["trade_id"])
    return ("legacy", trade.get("symbol"), trade.get("entry_timestamp"), trade.get("verdict"), trade.get("entry_price"))


def move_to_history(ledger_file: str | os.PathLike, history_file: str | os.PathLike, closed_trade: dict) -> None:
    """Append the closed trade to history, then delete the ledger.

    A damaged history raises DamagedRecordError before anything is written, so the
    trade stays OPEN in its ledger. If a crash left the trade in history but the
    ledger in place, the trade is not appended a second time.
    """
    history = _read_recorded_json(history_file, list) if os.path.exists(history_file) else []

    if not any(isinstance(t, dict) and _trade_key(t) == _trade_key(closed_trade) for t in history):
        write_json_atomic(history_file, [*history, closed_trade])

    # Clear ledger
    os.remove(ledger_file)


# --- Diagnostics ---

def append_system_health(health_payload: dict) -> None:
    """Append one JSON line to logs/system_health.log in the data folder."""
    log_dir = pathlib.Path(config.LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "system_health.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(health_payload) + "\n")


# --- Run lock (AGENTS.md §5: a scheduled run and a manual one must never overlap) ---

# Windows locks byte ranges, and a locked range can't be read by other processes. The lock sits far
# past the short "who holds it" text at the start of the file, so a blocked run can still read that text.
_WINDOWS_LOCK_OFFSET = 1 << 20


class RunLockedError(Exception):
    """Another process holds the run lock for this data folder."""

    def __init__(self, path: pathlib.Path, holder: str) -> None:
        super().__init__(f"{path} is held by {holder}")
        self.path = path
        self.holder = holder


def _try_lock(fd: int) -> None:
    """Take the OS lock without waiting; raise OSError if another process holds it."""
    if os.name == "nt":
        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if os.name == "nt":
        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _lock_holder(path: pathlib.Path) -> str:
    """The "who holds it" text a running process wrote, for the message (best effort)."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text or "another process"


@contextlib.contextmanager
def run_lock(command: str) -> Iterator[None]:
    """Hold the data folder's run lock while the block runs; raise RunLockedError if another run holds it.

    This is an OS lock on an open file, not "the file exists": the OS releases it when the process
    ends, even after a crash or a kill, so a stale lock can never block later runs. The file itself
    stays, holding the last holder's details, and is never deleted.
    """
    path = pathlib.Path(config.LOCK_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            _try_lock(fd)
        except OSError as e:
            raise RunLockedError(path, _lock_holder(path)) from e
        try:
            holder = f"pid {os.getpid()}, command '{command}', started {utc_iso(datetime.now(timezone.utc))}"
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, (holder + "\n").encode("utf-8"))
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)
