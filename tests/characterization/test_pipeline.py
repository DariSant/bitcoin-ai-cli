"""Characterization: a full analyze -> operate -> resolve cycle for long, short and SIT ON HANDS.

Captures the exact prompts, the analysis JSON, the execution footprint, the
ledger and the history record, plus the console output of each step.
"""

import json

import pytest

from tests.characterization.support import (
    files_under,
    local_stamp,
    normalize_local_time,
    prompts_text,
    read_json,
    scenario_replies,
)

STRATEGIES = {"defensive": "DEF", "greedy": "GREED"}


def resolution_candles(ledger: dict, entry_candle_open: float) -> list[list[float]]:
    """15m candles after an entry: the entry candle sweeps both levels, then a quiet candle, then a hit.

    Long trades hit the target (WIN); short trades hit the stop (LOSS), so both PnL branches run.
    """
    entry, stop, target = ledger["entry_price"], ledger["stop_loss"], ledger["take_profit"]
    if ledger["verdict"] == "GO LONG":
        hit = (target + 5, entry - 1)
    else:
        hit = (stop + 5, entry - 1)
    rows = [
        (entry_candle_open, max(stop, target) + 100, min(stop, target) - 100),
        (entry_candle_open + 900, entry + 1, entry - 1),
        (entry_candle_open + 1800, *hit),
    ]
    return [[int(ts * 1000), entry, high, low, entry, 100.0] for ts, high, low in rows]


@pytest.mark.parametrize("scenario", ["long", "short", "sit"])
def test_full_cycle(scenario, run_cli, gemini, exchange, clock, tmp_path, snapshot):
    out = tmp_path / "output_alpha"
    t_analyze = clock.epoch
    gemini.script = scenario_replies(scenario)

    analyze = run_cli("analyze")

    assert analyze.exit_code == 0, analyze.output
    assert gemini.script == [], "every canned reply should be used"
    snapshot(f"{scenario}_prompts.txt", prompts_text(gemini))

    records: dict = {"analysis": {}, "execution": {}, "ledger": {}, "history": {}}
    for strategy, prefix in STRATEGIES.items():
        path = out / "analyze" / strategy / "2026-10" / f"{local_stamp(t_analyze)}_BTCUSDT_{prefix}_analysis.json"
        records["analysis"][strategy] = read_json(path)

    # Operate one minute later, inside the 10-minute freshness window.
    clock.advance(60)
    t_operate = clock.epoch
    operate = run_cli("operate")
    assert operate.exit_code == 0, operate.output

    for strategy, prefix in STRATEGIES.items():
        execution = out / "operate" / strategy / "2026-10" / f"{local_stamp(t_operate)}_BTCUSDT_{prefix}_EXECUTION.json"
        ledger = out / strategy / "BTC_USDT_paper_ledger.json"
        if scenario == "sit":
            assert not execution.exists() and not ledger.exists()
            continue
        records["execution"][strategy] = read_json(execution)
        records["ledger"][strategy] = read_json(ledger)
        assert records["ledger"][strategy]["entry_timestamp"] == t_operate

    # Resolve 16 minutes later: the analysis is now stale, so nothing new is opened.
    resolve = None
    if scenario != "sit":
        exchange.candles["15m"] = resolution_candles(records["ledger"]["defensive"], entry_candle_open=t_analyze)
        clock.advance(16 * 60)
        resolve = run_cli("operate")
        assert resolve.exit_code == 0, resolve.output
        for strategy in STRATEGIES:
            assert not (out / strategy / "BTC_USDT_paper_ledger.json").exists()
            records["history"][strategy] = read_json(out / strategy / "BTC_USDT_trade_history.json")

    records["files_written"] = files_under(tmp_path)
    records["exchange_calls"] = exchange.calls
    epochs = (t_analyze, t_operate, clock.epoch)
    snapshot(f"{scenario}_records.json", normalize_local_time(json.dumps(records, indent=2), *epochs))

    console = f"$ analyze\n{analyze.output}\n$ operate\n{operate.output}"
    if resolve is not None:
        console += f"\n$ operate (resolution)\n{resolve.output}"
    snapshot(f"{scenario}_console.txt", normalize_local_time(console, *epochs))
