"""Technical indicators. Pure: no network, no disk, no printing."""

import pandas as pd
import pandas_ta as ta

from btc_cli import config


def calculate_indicators(df: pd.DataFrame, timeframe: str) -> dict:
    """
    Calculate technical indicators (EMAs, RSIs, and RSI Delta) using pandas-ta,
    and return the latest row's values.
    """
    # These lengths are part of the output field names (ema_34, rsi_13, vma_20, atr_14), so they are not settings.
    # Calculate EMAs
    df['EMA_34'] = ta.ema(df['close'], length=34)
    df['EMA_89'] = ta.ema(df['close'], length=89)
    df['EMA_144'] = ta.ema(df['close'], length=144)

    # Calculate RSIs
    df['RSI_13'] = ta.rsi(df['close'], length=13)
    df['RSI_47'] = ta.rsi(df['close'], length=47)

    # Calculate RSI Delta
    df['RSI_Delta'] = df['RSI_13'] - df['RSI_47']

    # Calculate Volume Moving Average (VMA)
    df['VMA_20'] = ta.sma(df['volume'], length=20)

    # Calculate ATR (Volatility)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Calculate Structural Levels (Swing High / Swing Low)
    df['swing_high'] = df['high'].rolling(config.SWING_LOOKBACK).max()
    df['swing_low'] = df['low'].rolling(config.SWING_LOOKBACK).min()

    # Calculate Volume Profile Point of Control (POC) and Value Area
    # Create equal price bins based on the close column
    bins = pd.cut(df['close'], bins=config.VOLUME_PROFILE_BINS)
    # Group by bins and sum the volume for each bin
    volume_by_bin = df.groupby(bins, observed=False)['volume'].sum()
    # Find the bin with the maximum volume
    max_volume_bin = volume_by_bin.idxmax()
    # Extract the midpoint price of that highest-volume bin
    poc_price = float(max_volume_bin.mid)

    # Calculate Value Area (VAH / VAL) - 70% True Distribution
    total_volume = volume_by_bin.sum()
    target_volume = total_volume * config.VALUE_AREA_SHARE

    # Sort bins by volume descending
    sorted_bins = volume_by_bin.sort_values(ascending=False)

    accumulated_volume = 0
    selected_bins = []

    # Accumulate volume until we reach >= 70%
    for bin_interval, vol in sorted_bins.items():
        accumulated_volume += vol
        selected_bins.append(bin_interval)
        if accumulated_volume >= target_volume:
            break

    # Calculate VAL and VAH based on selected bins
    val = float(min(b.left for b in selected_bins))
    vah = float(max(b.right for b in selected_bins))

    # Check for NaNs on required indicators in the last row
    last_row = df.iloc[-1]

    if pd.isna(last_row['EMA_144']) or pd.isna(last_row['RSI_13']) or pd.isna(last_row['RSI_47']) or pd.isna(last_row['VMA_20']) or pd.isna(last_row['ATR_14']) or pd.isna(last_row['swing_high']) or pd.isna(last_row['swing_low']):
        raise ValueError(f"Insufficient candle data fetched to calculate required technical and volume metrics for {timeframe} timeframe.")

    current_price = float(last_row['close'])
    ema_144 = float(last_row['EMA_144'])

    # Calculate Distance Percentages
    distance_to_144_ema_percent = ((current_price - ema_144) / ema_144) * 100
    distance_to_poc_percent = ((current_price - poc_price) / poc_price) * 100

    # Logic for Value Area Status
    if current_price > vah:
        price_to_va_status = "BREAKING_ABOVE_VAH"
    elif current_price < val:
        price_to_va_status = "BREAKING_BELOW_VAL"
    else:
        price_to_va_status = "INSIDE_VALUE"

    # Return rounded values for the latest row
    return {
        'price': round(current_price, 2),
        'calculated_resistance': round(float(last_row['swing_high']), 2),
        'calculated_support': round(float(last_row['swing_low']), 2),
        'volume': round(float(last_row['volume']), 2),
        'vma_20': round(float(last_row['VMA_20']), 2),
        'poc_price': round(poc_price, 2),
        'ema_34': round(float(last_row['EMA_34']), 2),
        'ema_89': round(float(last_row['EMA_89']), 2),
        'ema_144': round(ema_144, 2),
        'rsi_13': round(float(last_row['RSI_13']), 2),
        'rsi_47': round(float(last_row['RSI_47']), 2),
        'rsi_delta': round(float(last_row['RSI_Delta']), 2),
        'atr_14': round(float(last_row['ATR_14']), 2),
        'vah': round(vah, 2),
        'val': round(val, 2),
        'dist_144_percent': round(distance_to_144_ema_percent, 2),
        'dist_poc_percent': round(distance_to_poc_percent, 2),
        'va_status': price_to_va_status,
    }
