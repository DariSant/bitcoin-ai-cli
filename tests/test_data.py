"""Market data access: one exchange object, a timeout, and retries for temporary errors only (AGENTS.md §5)."""

import ccxt
import pytest

from btc_cli import config, data
from tests.characterization.support import read_json
from tests.characterization.test_resolution import LONG, NEXT_CANDLE, candle, write_ledger


class Flaky:
    """Fails with the given errors in turn, then returns the value."""

    def __init__(self, *errors: Exception, value: str = "ok") -> None:
        self.errors = list(errors)
        self.value = value
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.value


@pytest.mark.parametrize("error", [ccxt.RequestTimeout("t"), ccxt.DDoSProtection("rate"), ccxt.ExchangeNotAvailable("down"), ccxt.OnMaintenance("m")])
def test_temporary_network_errors_are_retried(error, exchange):
    call = Flaky(error, error)

    assert data.with_retries(call, "candles") == "ok"
    assert call.calls == 3
    assert exchange.sleeps == [2.0, 4.0]


def test_it_gives_up_after_the_configured_attempts(exchange):
    call = Flaky(*[ccxt.NetworkError("down")] * 5)

    with pytest.raises(data.MarketDataError, match="after 3 attempts"):
        data.with_retries(call, "candles")
    assert call.calls == 3


@pytest.mark.parametrize("error", [ccxt.BadSymbol("no such market"), ccxt.BadRequest("bad"), ccxt.ExchangeError("other")])
def test_permanent_exchange_errors_are_not_retried(error, exchange):
    call = Flaky(error)

    with pytest.raises(data.MarketDataError, match="Exchange error"):
        data.with_retries(call, "candles")
    assert call.calls == 1 and exchange.sleeps == []


def test_one_exchange_object_per_run_with_timeout_and_rate_limit(run_cli, exchange):
    assert run_cli("status").exit_code == 0

    assert exchange.instances == 1
    assert exchange.options == [{"timeout": config.EXCHANGE_TIMEOUT_SECONDS * 1000, "enableRateLimit": True}]


def test_a_short_outage_while_checking_a_trade_is_ridden_out(run_cli, exchange, tmp_path):
    write_ledger(tmp_path, "defensive", LONG)
    exchange.candles["15m"] = [candle(NEXT_CANDLE, 67600.0, 66950.0)]
    original = exchange.fetch_ohlcv
    failures = [ccxt.RequestTimeout("slow")]

    def flaky_fetch(*args, **kwargs):
        if failures:
            raise failures.pop()
        return original(*args, **kwargs)

    exchange.fetch_ohlcv = flaky_fetch

    out = run_cli("operate", "--def")

    assert out.exit_code == 0, out.output
    assert read_json(tmp_path / "output_alpha" / "defensive" / "BTC_USDT_trade_history.json")[0]["result"] == "WIN"
    assert exchange.sleeps == [2.0]


def test_an_outage_on_one_strategy_does_not_stop_the_other(run_cli, exchange, tmp_path):
    """Defensive has an open trade whose check fails; Greedy has none and still reaches its analysis lookup."""
    write_ledger(tmp_path, "defensive", LONG)
    exchange.error = ccxt.NetworkError("outage")

    out = run_cli("operate")

    assert out.exit_code == 1
    assert "Could not fetch market data to check the open DEFENSIVE trade" in out.output
    assert "No recent analysis found for GREEDY strategy." in out.output
