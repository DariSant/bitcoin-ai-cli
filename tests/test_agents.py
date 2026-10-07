"""Gemini call policy (AGENTS.md §5): retries, fallback, no-fallback errors and timeouts."""

import httpx
import pytest
from google.genai import errors

from btc_cli import agents, config
from tests.characterization.support import files_under
from tests.conftest import FALLBACK_MODEL, PRIMARY_MODEL


def server_error(code: int = 503) -> errors.ServerError:
    return errors.ServerError(code, {"error": {"code": code, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})


def client_error(code: int, retry_delay: str | None = None) -> errors.ClientError:
    error = {"code": code, "message": "client error", "status": "RESOURCE_EXHAUSTED" if code == 429 else "INVALID"}
    if retry_delay:
        error["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    return errors.ClientError(code, {"error": error})


def ask(gemini, router=None):
    return agents.query_llm_with_fallback(gemini, "prompt", None, "agent_test", "BTC/USDT", router)


def test_a_temporary_server_error_is_retried_on_the_same_model(gemini):
    gemini.script = [server_error(503), "answer"]

    assert ask(gemini) == ("answer", PRIMARY_MODEL)
    assert gemini.models_used() == [PRIMARY_MODEL, PRIMARY_MODEL]
    assert gemini.sleeps == [2.0]


def test_a_timeout_is_retried(gemini):
    gemini.script = [httpx.ReadTimeout("timed out"), "answer"]

    assert ask(gemini) == ("answer", PRIMARY_MODEL)
    assert gemini.sleeps == [2.0]


def test_three_server_errors_move_the_call_to_the_fallback(gemini):
    gemini.script = [server_error(), server_error(), server_error(), "answer"]

    assert ask(gemini) == ("answer", FALLBACK_MODEL)
    assert gemini.models_used() == [PRIMARY_MODEL] * 3 + [FALLBACK_MODEL]
    assert gemini.sleeps == [2.0, 4.0]


def test_a_429_waits_as_long_as_gemini_asks(gemini):
    gemini.script = [client_error(429, retry_delay="7s"), "answer"]

    assert ask(gemini) == ("answer", PRIMARY_MODEL)
    assert gemini.sleeps == [7.0]


def test_a_429_asking_for_a_long_wait_falls_back_at_once(gemini):
    """E.g. the daily quota is used up: waiting an hour would block the scheduled run."""
    gemini.script = [client_error(429, retry_delay="3600s"), "answer"]

    assert ask(gemini) == ("answer", FALLBACK_MODEL)
    assert gemini.sleeps == []


@pytest.mark.parametrize("code", [400, 401, 403])
def test_request_key_and_permission_errors_stop_without_fallback(code, gemini):
    gemini.script = [client_error(code)]

    with pytest.raises(errors.ClientError):
        ask(gemini)
    assert gemini.models_used() == [PRIMARY_MODEL]
    assert gemini.sleeps == []


def test_a_bad_key_stops_analyze_with_exit_1_and_writes_no_record(run_cli, gemini, tmp_path):
    gemini.script = [client_error(401)]

    out = run_cli("analyze")

    assert out.exit_code == 1
    assert "AI processing failed" in out.output
    assert gemini.models_used() == [PRIMARY_MODEL]
    assert not [f for f in files_under(tmp_path) if f.startswith("output_")]


def test_a_missing_model_falls_back_without_retrying(gemini):
    gemini.script = [client_error(404), "answer"]

    assert ask(gemini) == ("answer", FALLBACK_MODEL)
    assert gemini.sleeps == []


def test_both_models_failing_returns_no_answer_and_logs_the_fallback_error(gemini, caplog):
    gemini.script = [RuntimeError("primary down"), RuntimeError("fallback down")]

    assert ask(gemini) == (None, None)
    assert "fallback model gemini-2.5-flash failed too" in caplog.text
    assert "fallback down" in caplog.text


def test_once_the_primary_failed_the_run_goes_straight_to_the_fallback(gemini):
    router = agents.ModelRouter()
    gemini.script = [RuntimeError("primary down"), "first", "second"]

    assert ask(gemini, router) == ("first", FALLBACK_MODEL)
    assert ask(gemini, router) == ("second", FALLBACK_MODEL)
    assert gemini.models_used() == [PRIMARY_MODEL, FALLBACK_MODEL, FALLBACK_MODEL]


def test_without_a_router_each_call_tries_the_primary_again(gemini):
    gemini.script = [RuntimeError("primary down"), "first", "second"]

    ask(gemini)
    ask(gemini)
    assert gemini.models_used() == [PRIMARY_MODEL, FALLBACK_MODEL, PRIMARY_MODEL]


def test_max_attempts_1_means_no_retries(gemini, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MAX_ATTEMPTS", 1)
    gemini.script = [server_error(), "answer"]

    assert ask(gemini) == ("answer", FALLBACK_MODEL)
    assert gemini.sleeps == []


def test_every_client_has_the_configured_timeout(run_cli, gemini):
    gemini.script = ["Canned answer."]

    assert run_cli("ask", "What is ATR?").exit_code == 0

    (kwargs,) = gemini.client_kwargs
    assert kwargs["http_options"].timeout == config.GEMINI_TIMEOUT_SECONDS * 1000
