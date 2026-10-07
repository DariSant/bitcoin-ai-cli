"""Canned Gemini replies and file helpers for the characterization tests."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.conftest import FROZEN_EPOCH, FakeGemini

# Fixture 15m values these replies were written against (see tests/fixtures/):
# price 66976.62, support 66802.39, resistance 67282.57, ATR 140.55, POC 64475.44.
LONG_MAGNET = "TARGET: 67450.00 | DISTANCE: 0.71%"
SHORT_MAGNET = "TARGET: 66500.00 | DISTANCE: 0.71%"
POC_MAGNET = "TARGET: 64475.44 | DISTANCE: 3.88%"


def tech_report(bias: str) -> dict[str, str]:
    """A canned Agent 1 (technical) reply."""
    direction = {"BULLISH": "BULLISH", "BEARISH": "BEARISH"}.get(bias, "RANGE")
    return {
        "general_analysis": f"4H and 15m EMA stacks CONFIRMED {direction}. RSI_13 below RSI_47 on both timeframes.",
        "trend_state": f"MACRO: {direction} | MICRO: {direction} | STATUS: ALIGNMENT",
        "momentum_divergence": "RSI_FAST: 52.13 | RSI_SLOW: 60.94 | DELTA: -8.81 | STATE: DECAY",
        "key_level_interaction": "THREAT: SUPPORT | DISTANCE: 0.26% | ACTION: TESTING",
        "bias": bias,
    }


def vol_report(bias: str, magnet_target: str) -> dict[str, str]:
    """A canned Agent 2 (volume) reply."""
    return {
        "general_analysis": "Volume EXPANSION above VMA_20 with price INSIDE_VALUE on both timeframes.",
        "liquidity_state": "STATUS: INSIDE_VALUE | ACTION: RANGE_ROTATION",
        "volume_momentum": "VOL_VS_VMA: 1.12 | STATE: EXPANSION",
        "magnet_target": magnet_target,
        "bias": bias,
    }


def manager_report(verdict: str, strategy: str) -> dict[str, str]:
    """A canned Agent 3 (defensive or greedy manager) reply."""
    return {
        "executive_summary": f"{strategy.capitalize()} desk verdict {verdict} on the canned test reports.",
        "confluence_matrix": "STRUCTURE: ALIGN | VOLUME: SUPPORTIVE | ACTION: VALID",
        "risk_vector": "THREAT_PROXIMITY: LOW | MAGNET_PULL: STRONG | OVERALL_RISK: ASYMMETRIC",
        "final_verdict": verdict,
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    "long": {"bias": "BULLISH", "magnet": LONG_MAGNET, "defensive": "GO LONG", "greedy": "GO LONG"},
    "short": {"bias": "BEARISH", "magnet": SHORT_MAGNET, "defensive": "GO SHORT", "greedy": "GO SHORT"},
    "sit": {"bias": "NEUTRAL", "magnet": POC_MAGNET, "defensive": "SIT ON HANDS", "greedy": "SIT ON HANDS"},
}


def analysis_replies(bias: str, magnet: str, defensive: str | None, greedy: str | None) -> list[str]:
    """Reply texts in call order: Agent 1, Agent 2, then each requested manager."""
    replies = [json.dumps(tech_report(bias)), json.dumps(vol_report(bias, magnet))]
    if defensive is not None:
        replies.append(json.dumps(manager_report(defensive, "defensive")))
    if greedy is not None:
        replies.append(json.dumps(manager_report(greedy, "greedy")))
    return replies


def scenario_replies(name: str) -> list[str]:
    s = SCENARIOS[name]
    return analysis_replies(s["bias"], s["magnet"], s["defensive"], s["greedy"])


def prompts_text(gemini: FakeGemini) -> str:
    """Every Gemini call as readable text: model, schema, then the exact prompt."""
    parts = []
    for i, call in enumerate(gemini.calls, start=1):
        parts.append(
            f"===== call {i} | model={call['model']} | schema={call['response_schema']} | mime={call['response_mime_type']} =====\n"
            f"{call['contents']}"
        )
    return "\n".join(parts)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def files_under(root: Path) -> list[str]:
    """All files below root as sorted POSIX paths relative to root."""
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def only_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    assert len(matches) == 1, f"expected one {pattern} under {root}, found {[str(m) for m in matches]}"
    return matches[0]


def local_stamp(epoch: float) -> str:
    """The CLI's file-name stamp: naive local wall time (machine dependent, a known issue)."""
    return datetime.fromtimestamp(epoch).strftime("%Y%m%d_%H%M%S")


def normalize_local_time(text: str, *epochs: float) -> str:
    """Replace machine-dependent local-time strings with stable placeholders.

    The CLI uses naive local time for file names and `metadata.timestamp`, so the same
    instant renders differently in Madrid and on a UTC server. UTC strings are left as is.
    """
    for epoch in epochs:
        local = datetime.fromtimestamp(epoch)
        label = f"T{int(epoch - FROZEN_EPOCH):+d}s"
        text = text.replace(local.isoformat(), f"<LOCAL_ISO {label}>")
        text = text.replace(local.strftime("%Y%m%d_%H%M%S"), f"<LOCAL_STAMP {label}>")
    return text
