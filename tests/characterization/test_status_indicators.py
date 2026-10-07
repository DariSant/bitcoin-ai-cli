"""Characterization: indicator values, via the `status` footprint."""

import json

import ccxt

from tests.characterization.support import local_stamp, normalize_local_time, only_file, read_json


def test_status_writes_indicator_footprint(run_cli, exchange, clock, tmp_path, snapshot):
    result = run_cli("status")

    assert result.exit_code == 0, result.output
    footprint = only_file(tmp_path / "output_alpha" / "status" / "system", "*/*.json")
    assert footprint.parent.name == "2026-10"
    assert footprint.name == f"{local_stamp(clock.epoch)}_BTCUSDT_SYSTEM_analysis.json"
    # 200 candles per timeframe, fetched once each, including the still-forming last candle.
    assert exchange.calls == [
        {"symbol": "BTC/USDT", "timeframe": "4h", "since": None, "limit": 200},
        {"symbol": "BTC/USDT", "timeframe": "15m", "since": None, "limit": 200},
    ]

    record = read_json(footprint)
    snapshot("status_footprint.json", normalize_local_time(json.dumps(record, indent=2), clock.epoch))
    snapshot("status_console.txt", normalize_local_time(result.output, clock.epoch))


def test_status_exchange_failure_exits_1(run_cli, exchange, tmp_path):
    exchange.error = ccxt.NetworkError("simulated outage")

    result = run_cli("status")

    assert result.exit_code == 1
    assert "Could not calculate metrics" in result.output
    assert not (tmp_path / "output_alpha").exists()

