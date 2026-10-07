"""CLI-level behaviour: the run lock around the commands that write data (AGENTS.md §5), and mock."""

import json
import shutil

import pytest

from btc_cli import storage, trade_operator
from tests.characterization.support import files_under
from tests.conftest import REPO_ROOT


@pytest.mark.parametrize("command", ["status", "analyze", "operate", "auto"])
def test_a_command_exits_3_and_does_nothing_while_another_run_holds_the_lock(command, run_cli, gemini, exchange, tmp_path):
    with storage.run_lock("held by the test"):
        out = run_cli(command)

    assert out.exit_code == 3, out.output
    assert "Another run is in progress (pid" in out.output
    assert "command 'held by the test'" in out.output
    assert "Nothing was done." in out.output
    assert exchange.calls == [] and gemini.calls == []
    assert files_under(tmp_path) == []


def test_the_lock_is_released_after_each_command(run_cli, tmp_path):
    assert run_cli("status").exit_code == 0
    assert run_cli("status").exit_code == 0
    with storage.run_lock("next run"):
        pass


@pytest.mark.parametrize("args", [("commands",), ("mock", "missing.json")])
def test_commands_that_write_no_data_ignore_the_lock(args, run_cli):
    with storage.run_lock("held by the test"):
        out = run_cli(*args)

    assert out.exit_code != 3, out.output
    assert "Another run is in progress" not in out.output


@pytest.mark.parametrize("name", ["mock_long.json", "mock_short.json", "mock_floor.json"])
def test_mock_shows_the_same_ticket_operate_would_compute(name, run_cli, tmp_path):
    """TODO.md done-when: mock and operate use the same math (mock_floor.json is where the 1-ATR floor decides)."""
    shutil.copytree(REPO_ROOT / "mock_json", tmp_path / "mock_json")
    payload = json.loads((tmp_path / "mock_json" / name).read_text(encoding="utf-8"))
    order = trade_operator.compute_order(
        payload["verdict"], payload["current_price"], payload["atr_14"], payload["agent_1_threat_level"], payload["agent_2_magnet_target"]
    )
    report = trade_operator.build_operator_report(payload["current_price"], order)

    out = run_cli("mock", name)

    assert out.exit_code == 0, out.output
    assert f"Stop Loss: ${report['stop_loss']:,.2f}" in out.output
    assert f"Take Profit: ${report['take_profit']:,.2f}" in out.output
    assert f"Position Size USD: ${report['position_size_usd']:,.2f}" in out.output
