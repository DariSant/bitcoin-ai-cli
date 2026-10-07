"""Settings: read from config.toml at the project root, checked, and exposed as module constants.

Always read a setting as `config.NAME` at call time, never `from btc_cli.config import NAME`:
tests rebind these names, and the output_beta fallback rebinds BASE_DIR.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.toml"
# Overrides [paths] data_dir, e.g. a temporary folder for development runs (AGENTS.md §2.7).
DATA_DIR_ENV = "BTC_CLI_DATA_DIR"


class ConfigError(Exception):
    """config.toml is missing, unreadable, or holds a missing or impossible value."""


@dataclass(frozen=True)
class Settings:
    """Every value in config.toml, checked."""

    data_dir: str
    exchange_id: str
    market_type: str
    analysis_candles: int
    resolution_candles: int
    primary_model: str
    fallback_model: str
    gemini_timeout_seconds: int
    gemini_max_attempts: int
    gemini_max_retry_wait_seconds: int
    swing_lookback: int
    volume_profile_bins: int
    value_area_share: float
    risk_usd: float
    threat_buffer_atr: float
    min_stop_atr: float
    analysis_max_age_seconds: int
    account_balance_usdt: float
    risk_per_trade_percent: float


_KIND_NAMES = {int: "a whole number", float: "a number", str: "text in quotes"}


def _value(raw: dict[str, Any], section: str, key: str, kind: type) -> Any:
    """One value of the expected type; an int is accepted where a float is expected."""
    try:
        value = raw[section][key]
    except (KeyError, TypeError) as e:
        raise ConfigError(f"[{section}] {key} is missing") from e
    if kind is float and type(value) is int:
        value = float(value)
    if type(value) is not kind:
        raise ConfigError(f"[{section}] {key} must be {_KIND_NAMES[kind]}, found {value!r}")
    if kind is str and not value.strip():
        raise ConfigError(f"[{section}] {key} must not be empty")
    return value


def _in_range(raw: dict[str, Any], section: str, key: str, kind: type, low: float, high: float | None = None, low_inclusive: bool = False) -> Any:
    value = _value(raw, section, key, kind)
    too_low = value < low if low_inclusive else value <= low
    if too_low or (high is not None and value > high):
        bound = f"at least {low}" if low_inclusive else f"greater than {low}"
        if high is not None:
            bound += f" and at most {high}"
        raise ConfigError(f"[{section}] {key} must be {bound}, found {value!r}")
    return value


def parse_settings(raw: dict[str, Any]) -> Settings:
    """Check parsed TOML and return the settings; raises ConfigError naming the bad value."""
    return Settings(
        data_dir=_value(raw, "paths", "data_dir", str),
        exchange_id=_value(raw, "market", "exchange_id", str),
        market_type=_value(raw, "market", "market_type", str),
        analysis_candles=_in_range(raw, "market", "analysis_candles", int, 0),
        resolution_candles=_in_range(raw, "market", "resolution_candles", int, 0),
        primary_model=_value(raw, "gemini", "primary_model", str),
        fallback_model=_value(raw, "gemini", "fallback_model", str),
        gemini_timeout_seconds=_in_range(raw, "gemini_requests", "timeout_seconds", int, 0),
        gemini_max_attempts=_in_range(raw, "gemini_requests", "max_attempts", int, 1, 3, low_inclusive=True),
        gemini_max_retry_wait_seconds=_in_range(raw, "gemini_requests", "max_retry_wait_seconds", int, 0, low_inclusive=True),
        swing_lookback=_in_range(raw, "indicators", "swing_lookback", int, 0),
        volume_profile_bins=_in_range(raw, "indicators", "volume_profile_bins", int, 0),
        value_area_share=_in_range(raw, "indicators", "value_area_share", float, 0, 1),
        risk_usd=_in_range(raw, "operator", "risk_usd", float, 0),
        threat_buffer_atr=_in_range(raw, "operator", "threat_buffer_atr", float, 0, low_inclusive=True),
        min_stop_atr=_in_range(raw, "operator", "min_stop_atr", float, 0),
        analysis_max_age_seconds=_in_range(raw, "operator", "analysis_max_age_seconds", int, 0),
        account_balance_usdt=_in_range(raw, "operator", "account_balance_usdt", float, 0),
        risk_per_trade_percent=_in_range(raw, "operator", "risk_per_trade_percent", float, 0, 100),
    )


def resolve_data_dir(setting: str, override: str | None) -> Path:
    """The data folder as an absolute path: the override if set, else the setting; relative paths start at the project root."""
    chosen = Path(override if override and override.strip() else setting).expanduser()
    if not chosen.is_absolute():
        chosen = PROJECT_ROOT / chosen
    return chosen.resolve()


def load_settings(path: Path = CONFIG_FILE) -> Settings:
    """Read and check a settings file."""
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"settings file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name} is not valid TOML: {e}") from e
    return parse_settings(raw)


try:
    _settings = load_settings()
except ConfigError as e:
    # Raised while the program starts: a one-line message instead of a traceback.
    raise SystemExit(f"Settings error in {CONFIG_FILE.name}: {e}") from e

# [paths]: every path is absolute, built from the project root (AGENTS.md §5), never from
# the folder the command was started in.
DATA_DIR = resolve_data_dir(_settings.data_dir, os.environ.get(DATA_DIR_ENV))
# Output root for recorded data. The analyze pipeline rebinds this to BETA_DIR for the rest
# of the process once any agent uses the fallback model (known P0 bug, TODO.md "Two ledgers
# per strategy").
BASE_DIR = DATA_DIR / "output_alpha"
BETA_DIR = DATA_DIR / "output_beta"
LOGS_DIR = DATA_DIR / "logs"
ERROR_LOG = DATA_DIR / "error.log"
# One run at a time per data folder (see storage.run_lock). Never delete it to "unstick" a run.
LOCK_FILE = DATA_DIR / "run.lock"
# Inputs for the `mock` command: part of the project, not data.
MOCK_DIR = PROJECT_ROOT / "mock_json"

# [market]: data.create_exchange() builds the exchange from EXCHANGE_ID, so the source
# written into every record is the one actually used.
EXCHANGE_ID = _settings.exchange_id
MARKET_TYPE = _settings.market_type
ANALYSIS_CANDLES = _settings.analysis_candles
RESOLUTION_CANDLES = _settings.resolution_candles

# [gemini]
PRIMARY_MODEL = _settings.primary_model
FALLBACK_MODEL = _settings.fallback_model

# [gemini_requests]
GEMINI_TIMEOUT_SECONDS = _settings.gemini_timeout_seconds
GEMINI_MAX_ATTEMPTS = _settings.gemini_max_attempts
GEMINI_MAX_RETRY_WAIT_SECONDS = _settings.gemini_max_retry_wait_seconds

# [indicators]
SWING_LOOKBACK = _settings.swing_lookback
VOLUME_PROFILE_BINS = _settings.volume_profile_bins
VALUE_AREA_SHARE = _settings.value_area_share

# [operator]
RISK_USD = _settings.risk_usd
THREAT_BUFFER_ATR = _settings.threat_buffer_atr
MIN_STOP_ATR = _settings.min_stop_atr
ANALYSIS_MAX_AGE_SECONDS = _settings.analysis_max_age_seconds
ACCOUNT_BALANCE_USDT = _settings.account_balance_usdt
RISK_PER_TRADE_PERCENT = _settings.risk_per_trade_percent

# Record versions (AGENTS.md §6), written into every record. Kept in code, not config.toml,
# because they describe the code and settings together.
# SCHEMA_VERSION: bump when a record's structure changes. Records without it are schema 0.
# STRATEGY_VERSION: bump for any §2.4 change, including a [frozen] value in config.toml
# (owner decision 2026-10-07). "0.x" versions are warm-up data; "1.0" is the first official
# version, once Phase 1 is complete. Records without it are legacy "0.0".
SCHEMA_VERSION = 1
STRATEGY_VERSION = "0.1"
