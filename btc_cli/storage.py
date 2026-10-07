"""Output paths and every file the program writes: analyses, footprints, ledgers, history, logs.

Moved unchanged from app.py, including known Phase 2 issues (TODO.md): writes are
not atomic, paths are relative to the current folder, file names use naive local
time, and a damaged history file is silently replaced.
"""

import json
import os
import pathlib
from datetime import datetime, timezone

from btc_cli import config

# What readers assume for records written before versioning (AGENTS.md §6: legacy, warm-up).
LEGACY_SCHEMA_VERSION = 0
LEGACY_STRATEGY_VERSION = "0.0"


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


def local_now(now_utc: datetime) -> datetime:
    """The naive local wall time of the same instant, as `datetime.now()` gives it (file names, legacy fields)."""
    return now_utc.astimezone().replace(tzinfo=None)


# --- Analyses (and the status footprint) ---

def log_execution(command_name: str, strategy: str, symbol: str, data_4h: dict, data_15m: dict, agent1_report: dict = None, agent2_report: dict = None, agent3_report: dict = None, models_used: dict[str, str] | None = None) -> str:
    """
    Universally log execution state to a JSON footprint in BASE_DIR/.
    `models_used` maps each agent call to the model that answered it; None when no AI ran (status).
    """
    # One instant for the file name, the legacy local `timestamp` and the UTC fields.
    now_utc = datetime.now(timezone.utc)
    now = local_now(now_utc)
    directory_path = f"{config.BASE_DIR}/{command_name}/{strategy}/{now.strftime('%Y-%m')}/"
    os.makedirs(directory_path, exist_ok=True)

    strategy_prefix = _strategy_prefix(strategy)
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{symbol.replace('/', '')}_{strategy_prefix}_analysis.json"
    filepath = os.path.join(directory_path, filename)

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

    with open(filepath, "w") as file:
        json.dump(payload, file, indent=2)

    return filepath


def analysis_files(strategy: str) -> list[pathlib.Path]:
    """Every saved analysis for a strategy, across all months (unbounded, see TODO.md)."""
    analyze_dir = pathlib.Path(f"{config.BASE_DIR}/analyze/{strategy}")
    if not analyze_dir.exists():
        return []
    return list(analyze_dir.rglob("*.json"))


def latest_analysis_for_symbol(json_files: list[pathlib.Path], symbol: str) -> pathlib.Path | None:
    """The symbol's analysis with the newest file modification time, or None."""
    # Filter files for the requested symbol
    symbol_files = [f for f in json_files if symbol.replace("/", "") in f.name]
    if not symbol_files:
        return None

    # Sort files by modification time descending to get the most recent one
    symbol_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return symbol_files[0]


def read_analysis(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def analysis_link(path: pathlib.Path, analysis: dict) -> dict:
    """Fields that tie a ticket or trade to the analysis that opened it (None for legacy analyses)."""
    try:
        relative = path.relative_to(config.BASE_DIR).as_posix()
    except ValueError:
        relative = path.as_posix()
    metadata = analysis.get("metadata", {})
    return {
        "analysis_file": relative,
        "analysis_run_id": metadata.get("run_id"),
        "models_used": metadata.get("models_used"),
    }


# --- Operator outputs ---

def write_execution_footprint(now: datetime, now_utc: datetime, strategy: str, symbol: str, operator_payload: dict, operator_report: dict, source: dict) -> str:
    """Save the operator's inputs and ticket under BASE_DIR/operate/; returns the path.

    `now` is the naive local time used in the file name, `now_utc` the same instant,
    and `source` links to the analysis (see `analysis_link`).
    """
    directory_path = f"{config.BASE_DIR}/operate/{strategy}/{now.strftime('%Y-%m')}/"
    os.makedirs(directory_path, exist_ok=True)

    strategy_prefix = "DEF" if strategy == "defensive" else "GREED"
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{symbol.replace('/', '')}_{strategy_prefix}_EXECUTION.json"
    filepath = os.path.join(directory_path, filename)

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

    with open(filepath, "w") as file:
        json.dump(execution_footprint, file, indent=2)

    return filepath


def append_operator_error(log_entry: str) -> None:
    """Record a rejected ticket in BASE_DIR/operator_errors.log."""
    log_path = f"{config.BASE_DIR}/operator_errors.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(log_entry)


# --- Ledger and history ---

def ensure_strategy_dir(strategy: str) -> None:
    os.makedirs(f"{config.BASE_DIR}/{strategy}", exist_ok=True)


def ledger_path(strategy: str, symbol: str) -> str:
    clean_symbol = symbol.replace("/", "_")
    return f"{config.BASE_DIR}/{strategy}/{clean_symbol}_paper_ledger.json"


def history_path(strategy: str, symbol: str) -> str:
    clean_symbol = symbol.replace("/", "_")
    return f"{config.BASE_DIR}/{strategy}/{clean_symbol}_trade_history.json"


def read_ledger(path: str) -> dict | None:
    """The ledger, or None if it is missing or not valid JSON (known P0 bug: a damaged ledger reads as no trade)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def write_ledger(path: str, ledger_entry: dict) -> None:
    with open(path, "w") as file:
        json.dump(ledger_entry, file, indent=2)


def move_to_history(ledger_file: str, history_file: str, closed_trade: dict) -> None:
    """Append the closed trade to history, then delete the ledger."""
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except:  # noqa: E722 — known P0 bug kept by the split: a damaged history is silently replaced
            pass

    history.append(closed_trade)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    # Clear ledger
    os.remove(ledger_file)


# --- Diagnostics ---

def append_system_health(health_payload: dict) -> None:
    """Append one JSON line to logs/system_health.log."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "system_health.log")
    with open(log_path, "a") as f:
        f.write(json.dumps(health_payload) + "\n")
