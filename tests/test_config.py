"""Tests for btc_cli.config: loading and checking config.toml."""

import copy
import dataclasses
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from btc_cli import config

# The frozen values (AGENTS.md §2.4) as they were hardcoded before config.toml (strategy_version "0.1").
FROZEN_0_1 = dict(
    exchange_id="binance",
    market_type="spot",
    analysis_candles=200,
    resolution_candles=100,
    primary_model="gemini-3.5-flash-lite",
    fallback_model="gemini-2.5-flash",
    swing_lookback=20,
    volume_profile_bins=10,
    value_area_share=0.70,
    risk_usd=100.0,
    threat_buffer_atr=0.5,
    min_stop_atr=1.0,
    analysis_max_age_seconds=600,
    account_balance_usdt=10000.0,
    risk_per_trade_percent=1.0,
)
# Settings that change where files go or how calls are retried, not which trades are taken.
NOT_FROZEN = {"data_dir", "gemini_timeout_seconds", "gemini_max_attempts", "gemini_max_retry_wait_seconds"}


def raw_settings() -> dict:
    with open(config.CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def test_config_toml_holds_the_strategy_0_1_values():
    """Strategy freeze guard (AGENTS.md §2.4).

    If this fails because a value in config.toml was changed on purpose, that change needs the
    owner's approval, a STRATEGY_VERSION bump and a CHANGELOG entry. Then update this test.
    """
    assert config.STRATEGY_VERSION == "0.1"
    settings = dataclasses.asdict(config.load_settings())
    assert {k: v for k, v in settings.items() if k not in NOT_FROZEN} == FROZEN_0_1


def test_every_setting_is_classified_as_frozen_or_not():
    """A new setting must be added to FROZEN_0_1 or NOT_FROZEN, so the guard can't silently miss it."""
    fields = {f.name for f in dataclasses.fields(config.Settings)}
    assert fields == set(FROZEN_0_1) | NOT_FROZEN
    assert not set(FROZEN_0_1) & NOT_FROZEN


def test_module_constants_mirror_the_settings():
    assert (config.EXCHANGE_ID, config.MARKET_TYPE) == ("binance", "spot")
    assert (config.PRIMARY_MODEL, config.FALLBACK_MODEL) == (FROZEN_0_1["primary_model"], FROZEN_0_1["fallback_model"])
    assert config.RISK_USD == 100.0 and type(config.RISK_USD) is float
    assert config.VALUE_AREA_SHARE == 0.70
    assert config.ANALYSIS_MAX_AGE_SECONDS == 600


def test_an_int_is_accepted_where_a_float_is_expected():
    raw = raw_settings()
    raw["operator"]["risk_usd"] = 100
    settings = config.parse_settings(raw)
    assert settings.risk_usd == 100.0 and type(settings.risk_usd) is float


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("operator", "risk_usd", 0, "greater than 0"),
        ("operator", "risk_usd", -5.0, "greater than 0"),
        ("operator", "risk_usd", "100", "must be a number"),
        ("operator", "threat_buffer_atr", -0.1, "at least 0"),
        ("operator", "risk_per_trade_percent", 150.0, "at most 100"),
        ("operator", "analysis_max_age_seconds", 600.5, "must be a whole number"),
        ("market", "analysis_candles", True, "must be a whole number"),
        ("market", "exchange_id", "  ", "must not be empty"),
        ("indicators", "value_area_share", 1.5, "at most 1"),
        ("indicators", "volume_profile_bins", 0, "greater than 0"),
        ("gemini_requests", "max_attempts", 4, "at most 3"),
        ("gemini_requests", "max_attempts", 0, "at least 1"),
        ("gemini_requests", "timeout_seconds", 0, "greater than 0"),
        ("gemini_requests", "max_retry_wait_seconds", -1, "at least 0"),
    ],
)
def test_impossible_values_are_rejected_with_a_clear_message(section, key, value, message):
    raw = copy.deepcopy(raw_settings())
    raw[section][key] = value
    with pytest.raises(config.ConfigError, match=message) as raised:
        config.parse_settings(raw)
    assert f"[{section}] {key}" in str(raised.value)


def test_a_missing_value_is_named():
    raw = raw_settings()
    del raw["gemini"]["fallback_model"]
    with pytest.raises(config.ConfigError, match=r"\[gemini\] fallback_model is missing"):
        config.parse_settings(raw)


def test_a_missing_section_is_named():
    raw = raw_settings()
    del raw["operator"]
    with pytest.raises(config.ConfigError, match=r"\[operator\] risk_usd is missing"):
        config.parse_settings(raw)


def test_unreadable_files_raise_config_error(tmp_path):
    with pytest.raises(config.ConfigError, match="not found"):
        config.load_settings(tmp_path / "missing.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("[market\nexchange_id = ", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="not valid TOML"):
        config.load_settings(broken)


# --- Data folder (AGENTS.md §5: paths from the project root, not the current folder) ---

def test_the_default_data_folder_is_the_project_root():
    assert config.load_settings().data_dir == "."
    assert config.resolve_data_dir(".", None) == config.PROJECT_ROOT


def test_a_relative_data_folder_starts_at_the_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert config.resolve_data_dir("data", None) == config.PROJECT_ROOT / "data"


def test_an_absolute_data_folder_is_used_as_is(tmp_path):
    assert config.resolve_data_dir(str(tmp_path), None) == tmp_path.resolve()


def test_the_environment_override_wins_unless_blank(tmp_path):
    assert config.resolve_data_dir(".", str(tmp_path)) == tmp_path.resolve()
    assert config.resolve_data_dir(".", "  ") == config.PROJECT_ROOT


def test_every_data_path_is_absolute_and_inside_the_data_folder():
    for path in (config.BASE_DIR, config.BETA_DIR, config.LOGS_DIR, config.ERROR_LOG):
        assert Path(path).is_absolute()
        assert Path(path).parent == Path(config.DATA_DIR)


def test_the_environment_override_reaches_a_fresh_process(tmp_path):
    """BTC_CLI_DATA_DIR is read when the program starts. Only btc_cli.config is imported, so no .env is read."""
    env = {**os.environ, config.DATA_DIR_ENV: str(tmp_path)}
    script = "from btc_cli import config; print(config.DATA_DIR); print(config.BASE_DIR)"
    out = subprocess.run([sys.executable, "-c", script], cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.splitlines() == [str(tmp_path.resolve()), str(tmp_path.resolve() / "output_alpha")]
