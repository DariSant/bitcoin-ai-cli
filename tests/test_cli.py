"""CLI-level behaviour: the run lock around the commands that write data (AGENTS.md §5)."""

import pytest

from btc_cli import storage
from tests.characterization.support import files_under


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
