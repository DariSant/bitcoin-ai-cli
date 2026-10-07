"""Regenerate the saved candle fixtures used by the characterization tests.

The candles are a seeded random walk, not real market data: the tests only need
inputs that never change. Rerunning this script must reproduce the committed
files byte for byte; if it doesn't, the snapshots in tests/snapshots/ are stale.

    uv run python tests/fixtures/make_candles.py
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent

# Last candle opens exactly at the frozen test clock (see tests/conftest.py),
# so it is the still-forming candle, as it would be on the exchange.
LAST_OPEN = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
COUNT = 200


def make_candles(seed: int, interval_s: int, start_price: float, step_sigma: float, base_volume: float) -> list[list[float]]:
    """Return COUNT [timestamp_ms, open, high, low, close, volume] rows ending at LAST_OPEN."""
    rng = random.Random(seed)
    last_ms = int(LAST_OPEN.timestamp() * 1000)
    rows = []
    close = start_price
    for i in range(COUNT):
        ts = last_ms - (COUNT - 1 - i) * interval_s * 1000
        open_ = close
        close = open_ * (1 + rng.gauss(0.0002, step_sigma))
        high = max(open_, close) * (1 + abs(rng.gauss(0, step_sigma / 2)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, step_sigma / 2)))
        volume = base_volume * rng.uniform(0.4, 1.8)
        rows.append([ts, round(open_, 2), round(high, 2), round(low, 2), round(close, 2), round(volume, 3)])
    return rows


def main() -> None:
    fixtures = {
        "candles_4h.json": make_candles(seed=4, interval_s=4 * 3600, start_price=60000.0, step_sigma=0.008, base_volume=4000.0),
        "candles_15m.json": make_candles(seed=15, interval_s=15 * 60, start_price=64000.0, step_sigma=0.0015, base_volume=250.0),
    }
    for name, rows in fixtures.items():
        with open(FIXTURES_DIR / name, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rows, f)
            f.write("\n")


if __name__ == "__main__":
    main()
