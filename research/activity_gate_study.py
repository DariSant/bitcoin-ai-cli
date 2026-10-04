"""
Market Activity Gate - threshold study
======================================

WHAT THIS SCRIPT DOES (in plain English)
----------------------------------------
The trading tool should NOT trade when the market is too quiet, because:
  * price moves less, so the take-profit is harder to reach within 24 hours, and
  * the stop loss is tighter, so trading fees eat a bigger share of each trade.

This script checks whether the "too quiet" thresholds are still sensible.
It downloads about one year of real BTC/USDT perpetual-futures candles from
Binance (public data: no API key, no Gemini calls, costs nothing), then
pretends to open a trade every 30 minutes using the planned operator rules
(stop loss, 1.5R target, 24-hour time limit, fees and slippage).

For every candidate threshold it compares the moments the gate would BLOCK
against the moments it would ALLOW. A threshold is worth keeping when the
blocked moments are clearly worse than the allowed ones in BOTH halves of
the year (so we know the result is not a fluke of one period).

Because the trade direction is chosen at random in this simulation, the
average result before costs is about 0R everywhere. The differences you see
are therefore pure costs and missed movement - exactly what the gate is
meant to avoid. The gate does not create profit on its own; the AI still
has to pick the right direction.

HOW TO RUN IT (from the project folder)
---------------------------------------
    uv run research/activity_gate_study.py

It takes 1-3 minutes. The results are printed AND saved as a dated
Markdown report in research/results/, so you can compare re-checks over time.

WHEN TO RUN IT
--------------
Once a month (see TODO.md / CHANGELOG.md), and also straight away if you
change the fee rate, the exchange, the symbol, or the stop/target rules.
Only change a threshold when the report says REVIEW in two monthly checks
in a row - changing it every month would just chase noise.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# =============================================================================
# 1. SETTINGS
# These mirror the values agreed in TODO.md ("Market activity gate" item).
# When the real config.toml exists, keep these numbers in sync with it.
# =============================================================================

# --- Market and data window ---
EXCHANGE_ID = "binanceusdm"          # Binance USDT-M perpetual futures (public data)
SYMBOL = "BTC/USDT:USDT"             # the perpetual BTC contract
DAYS_TO_DOWNLOAD = 400               # ~1 year to study + ~1 month of warm-up history

# --- Current gate thresholds (what we are checking) ---
CURRENT_RVOL_MIN = 0.50              # last hour volume / median of same hour over 20 days
CURRENT_ATR15_MIN_PERCENT = 0.15     # 15m ATR as % of price
CURRENT_ATR4H_MIN_PERCENT = 0.60     # 4H ATR as % of price (safety net)
CURRENT_MAX_COST_R = 0.30            # fees + slippage as a share of the risk

RVOL_LOOKBACK_DAYS = 20              # how many past days build the "normal volume" baseline

# --- Operator rules used in the simulation (from TODO.md) ---
FEE_RATE_PER_SIDE = 0.0005           # 0.05% futures taker fee, paid on entry and on exit
SLIPPAGE_PER_SIDE = 0.0002           # 0.02% (2 basis points) worse price on entry and exit
TARGET_REWARD_TO_RISK = 1.5          # minimum agreed R:R
MAX_HOLD_CANDLES_15M = 96            # 24 hours = 96 candles of 15 minutes
DECISION_EVERY_MINUTES = 30          # the planned schedule: one decision every 30 minutes

# --- What counts as a "clear" difference between blocked and allowed moments ---
MEANINGFUL_GAP_IN_R = 0.05           # blocked trades must be at least 0.05R worse, in both halves
MIN_BLOCKED_CASES_PER_HALF = 50      # fewer blocked cases than this in a half = too few to judge

# --- Where to save the report ---
RESULTS_FOLDER = Path(__file__).resolve().parent / "results"

FIFTEEN_MINUTES_MS = 15 * 60 * 1000
FOUR_HOURS_MS = 4 * 60 * 60 * 1000


# =============================================================================
# 2. DOWNLOAD CANDLES
# =============================================================================

def download_closed_candles(exchange, timeframe: str, candle_length_ms: int) -> pd.DataFrame:
    """
    Download candles page by page (Binance returns at most 1,500 per request),
    then drop the last candle if it has not finished yet. Only finished
    ("closed") candles are used, so the numbers never change between runs.
    """
    now_ms = exchange.milliseconds()
    next_start_ms = now_ms - DAYS_TO_DOWNLOAD * 24 * 60 * 60 * 1000
    all_rows = []

    while next_start_ms < now_ms:
        # Try each page up to 3 times, waiting a little longer each time,
        # so one network hiccup does not ruin the whole study.
        for attempt_number in range(1, 4):
            try:
                page = exchange.fetch_ohlcv(SYMBOL, timeframe, since=next_start_ms, limit=1500)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as error:
                if attempt_number == 3:
                    raise RuntimeError(f"Could not download {timeframe} candles: {error}") from error
                time.sleep(2 * attempt_number)
        if not page:
            break
        all_rows.extend(page)
        next_start_ms = page[-1][0] + candle_length_ms
        time.sleep(0.2)  # be polite to the exchange

    candles = pd.DataFrame(all_rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    candles = candles.drop_duplicates("open_time").reset_index(drop=True)
    candles["close_time"] = candles["open_time"] + candle_length_ms
    # Keep only candles that have already closed.
    return candles[candles["close_time"] <= now_ms].reset_index(drop=True)


# =============================================================================
# 3. INDICATORS (closed candles only)
# =============================================================================

def wilder_atr(candles: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Average True Range, the standard way (Wilder's smoothing, as on TradingView).
    "True range" = the biggest of: high-low, |high - previous close|, |low - previous close|.
    Written by hand so this script does not depend on pandas-ta.
    """
    previous_close = candles["close"].shift(1)
    true_range = pd.concat([
        candles["high"] - candles["low"],
        (candles["high"] - previous_close).abs(),
        (candles["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)

    atr_values = np.full(len(candles), np.nan)
    if len(candles) > length:
        # The first ATR value is a simple average of the first `length` true ranges...
        atr_values[length] = true_range.iloc[1:length + 1].mean()
        # ...after that, each new value moves 1/length of the way towards the new true range.
        for index in range(length + 1, len(candles)):
            atr_values[index] = (atr_values[index - 1] * (length - 1) + true_range.iloc[index]) / length
    return pd.Series(atr_values, index=candles.index)


def add_indicators(candles_15m: pd.DataFrame, candles_4h: pd.DataFrame) -> pd.DataFrame:
    """Add every value the gate and the simulated operator need to the 15m table."""
    data = candles_15m.copy()

    # Volatility of the 15m candles, in dollars and as % of price.
    data["atr_15m"] = wilder_atr(data)
    data["atr_15m_percent"] = data["atr_15m"] / data["close"] * 100

    # Swing levels used for the stop loss (lowest low / highest high of the last 20 candles).
    data["swing_low_20"] = data["low"].rolling(20).min()
    data["swing_high_20"] = data["high"].rolling(20).max()

    # Seasonal relative volume (RVOL):
    #   volume of the last hour (4 closed candles) divided by the median volume
    #   of that SAME clock hour over the previous 20 days.
    # Comparing with the same hour removes the daily rhythm of BTC volume
    # (quiet late at night UTC, busy at the US market open).
    data["volume_last_hour"] = data["volume"].rolling(4).sum()
    data["time_of_day_slot"] = (data["close_time"] // FIFTEEN_MINUTES_MS) % 96
    data["normal_volume_this_hour"] = data.groupby("time_of_day_slot")["volume_last_hour"].transform(
        lambda same_slot: same_slot.shift(1).rolling(RVOL_LOOKBACK_DAYS, min_periods=15).median()
    )
    data["rvol"] = data["volume_last_hour"] / data["normal_volume_this_hour"]

    # 4H volatility: attach the most recent 4H candle that had ALREADY closed.
    four_hour = candles_4h.copy()
    four_hour["atr_4h"] = wilder_atr(four_hour)
    data = pd.merge_asof(
        data.sort_values("close_time"),
        four_hour[["close_time", "atr_4h"]].dropna().sort_values("close_time"),
        on="close_time", direction="backward",
    )
    data["atr_4h_percent"] = data["atr_4h"] / data["close"] * 100
    return data


# =============================================================================
# 4. SIMULATE THE OPERATOR AT EVERY DECISION POINT
# =============================================================================

def simulate_one_side(direction, entry_price, stop_price, target_price, future_highs, future_lows, close_after_24h):
    """
    Walk forward through the next 24 hours of candles and report what happened.
    If one candle touches both the stop and the target, we assume the stop came
    first (the cautious choice, same as the real tool).
    Returns (result in R before costs, exit price, True if the 24h limit closed it).
    """
    stop_distance = abs(entry_price - stop_price)
    if direction == "long":
        stop_hits, target_hits = future_lows <= stop_price, future_highs >= target_price
    else:
        stop_hits, target_hits = future_highs >= stop_price, future_lows <= target_price

    first_stop = np.argmax(stop_hits) if stop_hits.any() else len(future_highs)
    first_target = np.argmax(target_hits) if target_hits.any() else len(future_highs)

    if first_stop == len(future_highs) and first_target == len(future_highs):
        move = (close_after_24h - entry_price) if direction == "long" else (entry_price - close_after_24h)
        return move / stop_distance, close_after_24h, True
    if first_stop <= first_target:
        return -1.0, stop_price, False
    return TARGET_REWARD_TO_RISK, target_price, False


def simulate_decisions(data: pd.DataFrame) -> pd.DataFrame:
    """Every 30 minutes, simulate a long AND a short with the planned operator rules."""
    highs, lows, closes = data["high"].to_numpy(), data["low"].to_numpy(), data["close"].to_numpy()
    cost_rate_per_side = FEE_RATE_PER_SIDE + SLIPPAGE_PER_SIDE
    decision_rows = []

    # Skip the first ~30 days: the 20-day volume baseline needs history to warm up.
    first_usable_index = 30 * 96
    for index in range(first_usable_index, len(data) - MAX_HOLD_CANDLES_15M):
        row = data.iloc[index]
        is_decision_time = (row["close_time"] // 60_000) % DECISION_EVERY_MINUTES == 0
        if not is_decision_time or pd.isna(row["rvol"]) or pd.isna(row["atr_4h"]) or pd.isna(row["atr_15m"]):
            continue

        entry_price = row["close"]
        # Stop floor agreed in TODO.md: the larger of 1 x 15m ATR and 0.5 x 4H ATR.
        stop_floor = max(row["atr_15m"], 0.5 * row["atr_4h"])
        future_highs = highs[index + 1:index + 1 + MAX_HOLD_CANDLES_15M]
        future_lows = lows[index + 1:index + 1 + MAX_HOLD_CANDLES_15M]
        close_after_24h = closes[index + MAX_HOLD_CANDLES_15M]

        results_by_side = {}
        for direction in ("long", "short"):
            if direction == "long":
                stop_price = min(row["swing_low_20"] - 0.5 * row["atr_15m"], entry_price - stop_floor)
                target_price = entry_price + TARGET_REWARD_TO_RISK * (entry_price - stop_price)
            else:
                stop_price = max(row["swing_high_20"] + 0.5 * row["atr_15m"], entry_price + stop_floor)
                target_price = entry_price - TARGET_REWARD_TO_RISK * (stop_price - entry_price)
            stop_distance = abs(entry_price - stop_price)

            result_r, exit_price, timed_out = simulate_one_side(
                direction, entry_price, stop_price, target_price, future_highs, future_lows, close_after_24h)
            # The cost the operator can ESTIMATE before trading (used by the gate)...
            estimated_cost_r = cost_rate_per_side * 2 * entry_price / stop_distance
            # ...and the cost actually paid (exit price may differ from entry).
            actual_cost_r = cost_rate_per_side * (entry_price + exit_price) / stop_distance
            results_by_side[direction] = dict(
                result_r_after_costs=result_r - actual_cost_r, timed_out=timed_out,
                estimated_cost_r=estimated_cost_r, stop_percent=stop_distance / entry_price * 100)

        close_time = pd.to_datetime(row["close_time"], unit="ms", utc=True)
        decision_rows.append(dict(
            close_time=close_time,
            is_weekend=close_time.weekday() >= 5,
            rvol=row["rvol"],
            atr_15m_percent=row["atr_15m_percent"],
            atr_4h_percent=row["atr_4h_percent"],
            # Average of long and short = a trade in a random direction.
            result_r_after_costs=np.mean([s["result_r_after_costs"] for s in results_by_side.values()]),
            estimated_cost_r=np.mean([s["estimated_cost_r"] for s in results_by_side.values()]),
            timed_out_percent=np.mean([s["timed_out"] for s in results_by_side.values()]) * 100,
            stop_percent=np.mean([s["stop_percent"] for s in results_by_side.values()]),
            move_next_24h_percent=(future_highs.max() - future_lows.min()) / entry_price * 100,
        ))

    decisions = pd.DataFrame(decision_rows)
    # Split the period in two halves to check that conclusions hold in both.
    decisions["half"] = np.where(decisions.index < len(decisions) / 2, "first half", "second half")
    return decisions


# =============================================================================
# 5. COMPARE BLOCKED vs ALLOWED MOMENTS
# =============================================================================

def compare_blocked_and_allowed(decisions: pd.DataFrame, blocked_mask: pd.Series) -> dict:
    """Summarise one rule: how often it blocks, and how blocked moments compare with allowed ones."""
    summary = {
        "blocked_%": blocked_mask.mean() * 100,
        "blocked_avg_R": decisions.loc[blocked_mask, "result_r_after_costs"].mean(),
        "allowed_avg_R": decisions.loc[~blocked_mask, "result_r_after_costs"].mean(),
        "blocked_move_24h_%": decisions.loc[blocked_mask, "move_next_24h_percent"].median(),
        "allowed_move_24h_%": decisions.loc[~blocked_mask, "move_next_24h_percent"].median(),
        "blocked_cost_R": decisions.loc[blocked_mask, "estimated_cost_r"].median(),
        "allowed_cost_R": decisions.loc[~blocked_mask, "estimated_cost_r"].median(),
    }
    # The "gap" is how much better the allowed moments were than the blocked ones.
    fewest_blocked_in_a_half = len(decisions)
    for half_name in ("first half", "second half"):
        in_half = decisions["half"] == half_name
        blocked_in_half = blocked_mask & in_half
        allowed_in_half = ~blocked_mask & in_half
        fewest_blocked_in_a_half = min(fewest_blocked_in_a_half, int(blocked_in_half.sum()))
        gap = (decisions.loc[allowed_in_half, "result_r_after_costs"].mean()
               - decisions.loc[blocked_in_half, "result_r_after_costs"].mean())
        summary[f"gap_R_{half_name.split()[0]}"] = gap

    # A rule that almost never fires (like the 4H safety net) cannot be judged
    # half by half - say so instead of pretending it failed.
    if fewest_blocked_in_a_half < MIN_BLOCKED_CASES_PER_HALF:
        summary["verdict"] = "TOO FEW CASES"
    elif min(summary["gap_R_first"], summary["gap_R_second"]) >= MEANINGFUL_GAP_IN_R:
        summary["verdict"] = "KEEP"
    else:
        summary["verdict"] = "REVIEW"
    return summary


def sweep_thresholds(decisions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Try a range of values for each threshold, so a better value is easy to spot."""
    candidate_rules = {
        "Low volume: RVOL below": ("rvol", [0.3, 0.4, 0.5, 0.6, 0.7], CURRENT_RVOL_MIN),
        "Low volatility: 15m ATR % below": ("atr_15m_percent", [0.10, 0.125, 0.15, 0.175, 0.20], CURRENT_ATR15_MIN_PERCENT),
        "Low volatility: 4H ATR % below": ("atr_4h_percent", [0.5, 0.6, 0.7, 0.8], CURRENT_ATR4H_MIN_PERCENT),
    }
    tables = {}
    for title, (column, values, current_value) in candidate_rules.items():
        rows = []
        for value in values:
            row = compare_blocked_and_allowed(decisions, decisions[column] < value)
            row = {"threshold": value, "current": "<-- current" if value == current_value else "", **row}
            rows.append(row)
        tables[title] = pd.DataFrame(rows)

    cost_rows = []
    for value in [0.20, 0.25, 0.30, 0.35]:
        row = compare_blocked_and_allowed(decisions, decisions["estimated_cost_r"] > value)
        cost_rows.append({"threshold": value, "current": "<-- current" if value == CURRENT_MAX_COST_R else "", **row})
    tables["Costs too high: fees + slippage above (R)"] = pd.DataFrame(cost_rows)
    return tables


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    print("Downloading about one year of BTC perpetual-futures candles (public data)...")
    try:
        exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True, "timeout": 20000})
        candles_15m = download_closed_candles(exchange, "15m", FIFTEEN_MINUTES_MS)
        candles_4h = download_closed_candles(exchange, "4h", FOUR_HOURS_MS)
    except Exception as error:  # any download problem -> short, friendly message
        print(f"\nERROR: could not download market data from {EXCHANGE_ID}.\nDetails: {error}")
        print("Check your internet connection and try again in a few minutes.")
        sys.exit(1)

    print(f"  {len(candles_15m):,} candles of 15m and {len(candles_4h):,} candles of 4H downloaded.")
    print("Calculating indicators and simulating a decision every 30 minutes (1-2 minutes)...")
    decisions = simulate_decisions(add_indicators(candles_15m, candles_4h))

    # --- Current rules, one by one and combined ---
    low_volume = decisions["rvol"] < CURRENT_RVOL_MIN
    low_volatility = ((decisions["atr_15m_percent"] < CURRENT_ATR15_MIN_PERCENT)
                      | (decisions["atr_4h_percent"] < CURRENT_ATR4H_MIN_PERCENT))
    costs_too_high = decisions["estimated_cost_r"] > CURRENT_MAX_COST_R
    whole_gate = low_volume | low_volatility | costs_too_high

    current_rules = pd.DataFrame([
        {"rule": f"LOW_VOLUME (RVOL < {CURRENT_RVOL_MIN})", **compare_blocked_and_allowed(decisions, low_volume)},
        {"rule": f"LOW_VOLATILITY (15m ATR < {CURRENT_ATR15_MIN_PERCENT}% or 4H ATR < {CURRENT_ATR4H_MIN_PERCENT}%)",
         **compare_blocked_and_allowed(decisions, low_volatility)},
        {"rule": f"COSTS_TOO_HIGH (> {CURRENT_MAX_COST_R}R)", **compare_blocked_and_allowed(decisions, costs_too_high)},
        {"rule": "WHOLE GATE (any of the above)", **compare_blocked_and_allowed(decisions, whole_gate)},
    ])
    weekday_block = whole_gate[~decisions["is_weekend"]].mean() * 100
    weekend_block = whole_gate[decisions["is_weekend"]].mean() * 100
    threshold_tables = sweep_thresholds(decisions)

    # --- Build the report text (printed and saved) ---
    study_start = decisions["close_time"].min().strftime("%Y-%m-%d")
    study_end = decisions["close_time"].max().strftime("%Y-%m-%d")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    def as_text(table: pd.DataFrame) -> str:
        return "```\n" + table.round(3).to_string(index=False) + "\n```"

    report_lines = [
        f"# Activity gate study - {today}",
        "",
        f"- Data: {SYMBOL} on {EXCHANGE_ID}, decisions from {study_start} to {study_end} "
        f"({len(decisions):,} decision points, one every {DECISION_EVERY_MINUTES} minutes).",
        f"- Simulated rules: stop floor = max(1 x 15m ATR, 0.5 x 4H ATR), target {TARGET_REWARD_TO_RISK}R, "
        f"24 h time limit, fee {FEE_RATE_PER_SIDE * 100:.3f}% + slippage {SLIPPAGE_PER_SIDE * 100:.3f}% per side.",
        "- Trades are taken in a random direction, so results before costs are about 0R; the differences "
        "come from costs and missed movement.",
        f"- Verdict KEEP = blocked moments were at least {MEANINGFUL_GAP_IN_R}R worse than allowed ones in BOTH halves. "
        f"TOO FEW CASES = fewer than {MIN_BLOCKED_CASES_PER_HALF} blocked moments in one half.",
        "",
        "## Current thresholds",
        as_text(current_rules),
        f"Whole gate blocks {weekday_block:.1f}% of weekday decisions and {weekend_block:.1f}% of weekend decisions.",
        "",
        "## Threshold sweeps (to spot a better value)",
    ]
    for title, table in threshold_tables.items():
        report_lines += [f"### {title}", as_text(table), ""]
    report_lines += [
        "## How to read this",
        "- `blocked_avg_R` should be clearly lower (worse) than `allowed_avg_R`.",
        "- `gap_R_first` / `gap_R_second` = allowed minus blocked, in each half of the period.",
        "- Prefer the threshold where the gap stays clear in both halves without blocking much more time.",
        "- Change a threshold only if the current one shows REVIEW in two monthly checks in a row.",
        "- TOO FEW CASES is normal for the 4H ATR safety net: it rarely fires. Keep it unless it starts "
        "blocking a lot (more than about 5% of the time).",
    ]
    report_text = "\n".join(report_lines)

    print("\n" + report_text)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_FOLDER / f"activity_gate_{today}.md"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
