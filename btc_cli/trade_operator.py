"""The Python Operator: entry, stop, target and size. Pure: no network, no disk, no printing.

Moved unchanged from app.py. Known issues (fixed separately, see TODO.md Phase 1):
the risk is a fixed $100, and there is no minimum R:R, target distance limit or
leverage cap. `mock` and `operate` share `compute_order` (since 2026-10-07).
"""

from dataclasses import dataclass

from btc_cli import config


@dataclass(frozen=True)
class OrderCalc:
    """Operator output before rounding. When `valid` is False, only stop and target are set."""

    valid: bool
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float | None = None
    position_size_usd: float | None = None


def parse_magnet_target(raw_magnet_string) -> float | None:
    """Read the price from Agent 2's "TARGET: <price> | DISTANCE: <x>%" text; None if unparsable."""
    try:
        target_section = raw_magnet_string.split('|')[0]
        number_str = target_section.split(':')[1]
        return float(number_str.strip())
    except (IndexError, ValueError, AttributeError):
        return None


def compute_order(verdict: str, current_price, atr_14, threat_level, magnet_target) -> OrderCalc | None:
    """Operate's ticket math. None for a verdict that is neither GO LONG nor GO SHORT."""
    if verdict == "GO LONG":
        # Calculate baseline SL from structural threat + 0.5 ATR
        proposed_stop_loss = threat_level - (config.THREAT_BUFFER_ATR * atr_14)

        # Calculate a pure volatility SL (MIN_STOP_ATR full ATRs below entry)
        min_volatility_stop = current_price - config.MIN_STOP_ATR * atr_14

        # The Stop Loss must be the LOWER of the two (safest distance)
        stop_loss = min(proposed_stop_loss, min_volatility_stop)

        take_profit = magnet_target

        if stop_loss >= current_price or take_profit <= current_price:
            return OrderCalc(valid=False, stop_loss=stop_loss, take_profit=take_profit)

        risk_reward_ratio = (take_profit - current_price) / (current_price - stop_loss)
        position_size_btc = config.RISK_USD / (current_price - stop_loss)
        position_size_usd = position_size_btc * current_price

    elif verdict == "GO SHORT":
        # Calculate baseline SL from structural threat + 0.5 ATR
        proposed_stop_loss = threat_level + (config.THREAT_BUFFER_ATR * atr_14)

        # Calculate a pure volatility SL (MIN_STOP_ATR full ATRs above entry)
        min_volatility_stop = current_price + config.MIN_STOP_ATR * atr_14

        # The Stop Loss must be the HIGHER of the two (safest distance)
        stop_loss = max(proposed_stop_loss, min_volatility_stop)

        take_profit = magnet_target

        if stop_loss <= current_price or take_profit >= current_price:
            return OrderCalc(valid=False, stop_loss=stop_loss, take_profit=take_profit)

        risk_reward_ratio = (current_price - take_profit) / (stop_loss - current_price)
        position_size_btc = config.RISK_USD / (stop_loss - current_price)
        position_size_usd = position_size_btc * current_price
    else:
        return None

    return OrderCalc(True, stop_loss, take_profit, risk_reward_ratio, position_size_usd)


def build_operator_report(entry_price, order: OrderCalc) -> dict:
    """The rounded ticket that is shown, saved in the footprint and copied into the ledger."""
    return {
        "order_type": "MARKET",
        "entry_price": round(entry_price, 2),
        "stop_loss": round(order.stop_loss, 2),
        "take_profit": round(order.take_profit, 2),
        "risk_reward_ratio": round(order.risk_reward_ratio, 2),
        "position_size_usd": round(order.position_size_usd, 2)
    }
