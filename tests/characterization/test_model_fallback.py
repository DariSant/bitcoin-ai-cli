"""Characterization: primary/fallback model routing and the output_alpha/output_beta split."""

import json

from tests.conftest import FALLBACK_MODEL, PRIMARY_MODEL
from tests.characterization.support import LONG_MAGNET, analysis_replies, files_under, normalize_local_time, read_json, scenario_replies


def test_primary_failure_uses_fallback_and_routes_the_run_to_output_beta(run_cli, gemini, clock, tmp_path, snapshot):
    replies = scenario_replies("long")
    gemini.script = [RuntimeError("primary unavailable"), *replies]

    out = run_cli("analyze")

    assert out.exit_code == 0, out.output
    # Only Agent 1 fell back; later agents try the primary again and succeed.
    assert gemini.models_used() == [PRIMARY_MODEL, FALLBACK_MODEL, PRIMARY_MODEL, PRIMARY_MODEL, PRIMARY_MODEL]
    # The primary and the fallback receive the identical prompt.
    assert gemini.calls[0]["contents"] == gemini.calls[1]["contents"]
    # One fallback anywhere moves the whole run, both strategies, to output_beta.
    written = [f for f in files_under(tmp_path) if f.startswith("output_")]
    assert [f.split("/")[0:3] for f in written] == [["output_beta", "analyze", "defensive"], ["output_beta", "analyze", "greedy"]]
    # The record does not say which model produced it.
    record = read_json(tmp_path / written[0])
    assert "model" not in json.dumps(record["metadata"])
    health = (tmp_path / "logs" / "system_health.log").read_text(encoding="utf-8")
    snapshot("fallback_system_health.log", health)
    snapshot("fallback_console.txt", normalize_local_time(out.output, clock.epoch))


def test_fallback_in_greedy_manager_splits_one_run_across_both_folders(run_cli, gemini, tmp_path):
    a1, a2, defensive, greedy = analysis_replies("BULLISH", LONG_MAGNET, "GO LONG", "GO LONG")
    gemini.script = [a1, a2, defensive, RuntimeError("primary unavailable"), greedy]

    out = run_cli("analyze")

    assert out.exit_code == 0, out.output
    written = [f.split("/")[0:3] for f in files_under(tmp_path) if f.startswith("output_")]
    assert written == [["output_alpha", "analyze", "defensive"], ["output_beta", "analyze", "greedy"]]


def test_both_models_failing_on_agent_1_skips_the_cycle_with_exit_0(run_cli, gemini, tmp_path):
    """Known P1 bug: a total AI failure exits with success."""
    gemini.script = [RuntimeError("primary down"), RuntimeError("fallback down")]

    out = run_cli("analyze")

    assert out.exit_code == 0
    assert "[CRITICAL] Both models unreachable. Skipping cycle." in out.output
    assert gemini.models_used() == [PRIMARY_MODEL, FALLBACK_MODEL]
    assert files_under(tmp_path) == ["logs/system_health.log"]


def test_both_models_failing_on_defensive_manager_still_runs_greedy(run_cli, gemini, tmp_path):
    a1, a2, _, greedy = analysis_replies("BULLISH", LONG_MAGNET, "GO LONG", "GO LONG")
    gemini.script = [a1, a2, RuntimeError("primary down"), RuntimeError("fallback down"), greedy]

    out = run_cli("analyze")

    assert out.exit_code == 0, out.output
    written = [f.split("/")[0:3] for f in files_under(tmp_path) if f.startswith("output_")]
    assert written == [["output_alpha", "analyze", "greedy"]]


def test_invalid_json_reply_exits_1_without_writing(run_cli, gemini, tmp_path):
    gemini.script = ["this is not json"]

    out = run_cli("analyze")

    assert out.exit_code == 1
    assert "AI processing failed" in out.output
    assert files_under(tmp_path) == []


def test_auto_after_a_fallback_operates_on_output_beta(run_cli, gemini, exchange, clock, tmp_path, snapshot):
    """`auto` runs status, analyze and operate in one process, so the output_beta switch carries into operate."""
    gemini.script = [RuntimeError("primary unavailable"), *scenario_replies("long")]

    out = run_cli("auto")

    assert out.exit_code == 0, out.output
    files = files_under(tmp_path)
    assert any(f.startswith("output_alpha/status/") for f in files)
    assert "output_beta/defensive/BTC_USDT_paper_ledger.json" in files
    assert "output_beta/greedy/BTC_USDT_paper_ledger.json" in files
    # Market data is downloaded twice: once by status, once by analyze.
    assert [c["timeframe"] for c in exchange.calls] == ["4h", "15m", "4h", "15m"]
    snapshot("auto_fallback_files.json", normalize_local_time(json.dumps(files, indent=2), clock.epoch))
