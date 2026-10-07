"""Characterization: offline commands (`mock`, `commands`) and `ask`."""

import shutil

import pytest

from tests.conftest import PRIMARY_MODEL, REPO_ROOT


@pytest.fixture
def mock_inputs(tmp_path):
    """The harness points config.MOCK_DIR at tmp_path/mock_json, so copy the real inputs there."""
    shutil.copytree(REPO_ROOT / "mock_json", tmp_path / "mock_json")


@pytest.mark.parametrize("name", ["mock_long.json", "mock_short.json"])
def test_mock_prints_the_operator_ticket(name, mock_inputs, run_cli, gemini, exchange, tmp_path, snapshot):
    out = run_cli("mock", name)

    assert out.exit_code == 0, out.output
    assert gemini.calls == [] and exchange.calls == []
    # mock_json/ is the only thing in tmp_path: the command writes no files.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mock_json"]
    snapshot(f"{name.removesuffix('.json')}_console.txt", out.output)


def test_mock_missing_file_exits_1(mock_inputs, run_cli):
    out = run_cli("mock", "does_not_exist.json")

    assert out.exit_code == 1
    assert "File not found" in out.output


def test_mock_rejects_ticket_against_the_trade_direction(mock_inputs, run_cli, tmp_path):
    (tmp_path / "mock_json" / "bad.json").write_text(
        '{"verdict": "GO LONG", "account_balance_usdt": 10000.0, "risk_per_trade_percent": 1.0, '
        '"current_price": 70000.0, "atr_14": 1000.0, "agent_1_threat_level": 68500.0, "agent_2_magnet_target": 69000.0}',
        encoding="utf-8",
    )

    out = run_cli("mock", "bad.json")

    assert out.exit_code == 0
    assert "INVALID TICKET" in out.output


def test_commands_table(run_cli, snapshot):
    out = run_cli("commands")

    assert out.exit_code == 0, out.output
    snapshot("commands_console.txt", out.output)


def test_ask_sends_the_raw_question_without_a_schema(run_cli, gemini, tmp_path, snapshot):
    gemini.script = ["Canned answer."]

    out = run_cli("ask", "What is ATR?")

    assert out.exit_code == 0, out.output
    assert gemini.calls == [{"model": PRIMARY_MODEL, "contents": "What is ATR?", "response_mime_type": None, "response_schema": None}]
    assert list(tmp_path.iterdir()) == []
    snapshot("ask_console.txt", out.output)


def test_ask_with_both_models_down_exits_0(run_cli, gemini):
    """Known issue: ask's broad `except Exception` also catches its own typer.Exit (a RuntimeError)."""
    gemini.script = [RuntimeError("primary down"), RuntimeError("fallback down")]

    out = run_cli("ask", "What is ATR?")

    assert out.exit_code == 0
    assert "Both models unreachable" in out.output
    # str(typer.Exit) is empty, so the message ends right after the colon.
    assert out.output.endswith("An unexpected error occurred: \n")
