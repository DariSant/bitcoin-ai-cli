"""Output paths and every file the program writes: analyses, footprints, ledgers, history, logs.

Moved unchanged from app.py, including known Phase 2 issues (TODO.md): writes are
not atomic, paths are relative to the current folder, file names use naive local
time, and a damaged history file is silently replaced.
"""

import json
import os
import pathlib
from datetime import datetime

from btc_cli import config


def _strategy_prefix(strategy: str) -> str:
    return "DEF" if strategy == "defensive" else "GREED" if strategy == "greedy" else strategy.upper()


# --- Analyses (and the status footprint) ---

def log_execution(command_name: str, strategy: str, symbol: str, data_4h: dict, data_15m: dict, agent1_report: dict = None, agent2_report: dict = None, agent3_report: dict = None) -> str:
    """
    Universally log execution state to a JSON footprint in BASE_DIR/.
    """
    now = datetime.now()
    directory_path = f"{config.BASE_DIR}/{command_name}/{strategy}/{now.strftime('%Y-%m')}/"
    os.makedirs(directory_path, exist_ok=True)

    strategy_prefix = _strategy_prefix(strategy)
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{symbol.replace('/', '')}_{strategy_prefix}_analysis.json"
    filepath = os.path.join(directory_path, filename)

    payload = {
        "metadata": {
            "timestamp": now.isoformat(),
            "symbol": symbol,
            "command_run": command_name
        },
        "raw_market_data": {
            "4h": data_4h,
            "15m": data_15m
        }
    }

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


# --- Operator outputs ---

def write_execution_footprint(now: datetime, strategy: str, symbol: str, operator_payload: dict, operator_report: dict) -> str:
    """Save the operator's inputs and ticket under BASE_DIR/operate/; returns the path."""
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
            "strategy": strategy
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
