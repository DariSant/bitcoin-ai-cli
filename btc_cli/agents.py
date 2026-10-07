"""Gemini agents: response schemas, prompts and the primary/fallback model call."""

import json
import logging
import re
import time
import typing
from dataclasses import dataclass
from datetime import datetime, timezone

# httpx is google-genai's HTTP library (a dependency of it, see uv.lock). The SDK lets its
# timeouts and connection errors through unwrapped, so they are caught by httpx's types.
import httpx
from google import genai
from google.genai import errors, types
from pydantic import TypeAdapter, ValidationError
from rich.panel import Panel

from btc_cli import config, storage
from btc_cli.console import console

# --- Response schemas ---

class Agent1TechSchema(typing.TypedDict):
    general_analysis: str
    trend_state: str
    momentum_divergence: str
    key_level_interaction: str
    bias: typing.Literal["STRONGLY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONGLY_BEARISH"]

class Agent2VolumeSchema(typing.TypedDict):
    general_analysis: str
    liquidity_state: str
    volume_momentum: str
    magnet_target: str
    bias: typing.Literal["STRONGLY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONGLY_BEARISH"]

class Agent3ManagerSchema(typing.TypedDict):
    executive_summary: str
    confluence_matrix: str
    risk_vector: str
    final_verdict: typing.Literal["GO LONG", "GO SHORT", "SIT ON HANDS"]


# --- Reply validation ---

class InvalidReplyError(Exception):
    """An AI reply that is not valid JSON or does not match the schema the model was given."""

    def __init__(self, agent: str, reason: str, raw: str) -> None:
        super().__init__(f"{agent}: {reason}")
        self.agent = agent
        self.reason = reason
        self.raw = raw


def parse_reply(text: str, schema: type, agent: str) -> dict:
    """The reply as a dict, checked against the same schema Gemini was given; raises InvalidReplyError.

    The dict is the reply exactly as parsed, so the saved record is unchanged; validation only checks it.
    """
    try:
        TypeAdapter(schema).validate_json(text)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'reply'}: {err['msg']}" for err in e.errors()[:3])
        raise InvalidReplyError(agent, problems, text) from e
    return json.loads(text)


# --- Model call with retries and fallback (AGENTS.md §5) ---

# Wrong request, key or permission: the fallback would fail the same way, so the run stops (exit 1).
NO_FALLBACK_CODES = (400, 401, 403)

# Patched by the tests so retries don't really wait.
_sleep = time.sleep


def make_client(api_key: str) -> genai.Client:
    """A Gemini client whose requests time out, so a hung call can't block a scheduled run."""
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT_SECONDS * 1000))


@dataclass
class ModelRouter:
    """Per-run state: once the primary model has failed, the rest of the run goes straight to the fallback."""

    primary_failed: bool = False


def _retry_delay(error: errors.APIError) -> float | None:
    """The wait a 429 asks for (RetryInfo.retryDelay or a Retry-After header), in seconds; None if it gives none."""
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", str(error.details))
    if match:
        return float(match.group(1))
    headers = getattr(error.response, "headers", None) or {}
    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(retry_after) if retry_after is not None else None
    except ValueError:
        return None


def _generate(client: genai.Client, model: str, prompt: str, config_dict: dict | None, agent_name: str) -> str:
    """One model's answer, retrying temporary failures; raises the last error when it gives up.

    Temporary: HTTP 5xx, timeouts and connection errors (waits 2 s, 4 s), and 429 when the wait it
    asks for is short. Every attempt counts against the daily quota.
    """
    attempts = config.GEMINI_MAX_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config_dict)
            return response.text
        except errors.APIError as e:
            if e.code == 429:
                wait = _retry_delay(e)
                wait = 2.0 ** attempt if wait is None else wait
                if wait > config.GEMINI_MAX_RETRY_WAIT_SECONDS:
                    raise  # e.g. the daily quota is used up: waiting won't help this run
            elif e.code is not None and e.code >= 500:
                wait = 2.0 ** attempt
            else:
                raise
            reason = f"HTTP {e.code}"
        except httpx.TransportError as e:
            wait = 2.0 ** attempt
            reason = type(e).__name__
        if attempt == attempts:
            raise
        logging.warning(f"{agent_name}: {model} failed ({reason}); retry {attempt + 1} of {attempts} in {wait:.0f}s")
        console.print(f"[yellow]{model} is busy ({reason}). Retrying in {wait:.0f}s (attempt {attempt + 1} of {attempts})...[/yellow]")
        _sleep(wait)
    raise AssertionError("unreachable")


def query_llm_with_fallback(client: genai.Client, prompt: str, schema_class: type | None, agent_name: str, ticker: str = "BTC/USDT", router: ModelRouter | None = None) -> tuple[str | None, str | None]:
    """
    The answer and the model that gave it, or (None, None) if neither model answered.

    The primary model is tried first (with retries); on failure the call moves to the fallback model,
    and `router` remembers that for the rest of the run. A 400/401/403 is raised instead (no fallback).
    """
    primary_model = config.PRIMARY_MODEL
    fallback_model = config.FALLBACK_MODEL
    router = router if router is not None else ModelRouter()

    config_dict = None
    if schema_class:
        config_dict = {"response_mime_type": "application/json", "response_schema": schema_class}

    if not router.primary_failed:
        try:
            return _generate(client, primary_model, prompt, config_dict, agent_name), primary_model
        except errors.APIError as e:
            if e.code in NO_FALLBACK_CODES:
                raise
            primary_error: Exception = e
        except Exception as e:  # timeouts, 404 (model gone) and unexpected SDK errors: fall back, as before
            primary_error = e
        router.primary_failed = True
        logging.error(f"{agent_name}: primary model {primary_model} failed, switching to {fallback_model}", exc_info=primary_error)
        error_type = type(primary_error).__name__
        error_message = str(primary_error)

        failover_msg = (
            f"[bold yellow]⚠️ LLM ENDPOINT REROUTE TRIGGERED[/bold yellow]\n\n"
            f"Primary Model ({primary_model}) is currently unreachable.\n"
            f"Issue: {error_type} - {error_message}\n\n"
            f"Status: Instant failover to {fallback_model} initiated.\n"
            f"Routing: Diverting cycle data to [cyan]output_beta/[/cyan] to protect lake."
        )
        console.print(Panel(failover_msg, border_style="yellow", expand=False))

        # Log to logs/system_health.log
        health_payload = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ticker": ticker,
            "agent_failing": agent_name,
            "error_code": error_type,
            "error_message": error_message,
            "primary_model": primary_model,
            "fallback_model": fallback_model,
            "data_routed_to": "output_beta"
        }

        storage.append_system_health(health_payload)

    try:
        return _generate(client, fallback_model, prompt, config_dict, agent_name), fallback_model
    except errors.APIError as e:
        if e.code in NO_FALLBACK_CODES:
            raise
        logging.error(f"{agent_name}: fallback model {fallback_model} failed too", exc_info=True)
        return None, None
    except Exception:  # timeouts and unexpected SDK errors: report "no answer", logged in full
        logging.error(f"{agent_name}: fallback model {fallback_model} failed too", exc_info=True)
        return None, None


# --- Prompt payloads (the "data diet": each analyst sees only its own fields) ---

def extract_tech_data(d: dict) -> dict:
    return {k: d.get(k) for k in ['price', 'ema_34', 'ema_89', 'ema_144', 'rsi_13', 'rsi_47', 'rsi_delta', 'calculated_support', 'calculated_resistance', 'dist_144_percent', 'atr_14']}


def extract_vol_data(d: dict) -> dict:
    return {k: d.get(k) for k in ['price', 'volume', 'vma_20', 'poc_price', 'vah', 'val', 'va_status', 'dist_poc_percent']}


def build_tech_payload(data_4h: dict, data_15m: dict) -> str:
    """JSON market data for Agent 1."""
    return json.dumps({
        "4H_Timeframe": extract_tech_data(data_4h),
        "15m_Timeframe": extract_tech_data(data_15m)
    }, indent=2)


def build_vol_payload(data_4h: dict, data_15m: dict) -> str:
    """JSON market data for Agent 2."""
    return json.dumps({
        "4H_Timeframe": extract_vol_data(data_4h),
        "15m_Timeframe": extract_vol_data(data_15m)
    }, indent=2)


# --- Prompts (strategy-frozen text: any edit needs approval and a strategy_version bump) ---

def build_technical_prompt(tech_payload: str) -> str:
    """Agent 1: Technical Analyst."""
    return f"""SYSTEM PROMPT:
You are the Lead Technical Analyst for a quantitative trading firm. Your sole function is to evaluate calculated market structure, trend momentum, and key level interactions using exact data variables.

YOU ARE STRICTLY FORBIDDEN from using conversational filler, predictions, or retail trading slang.
BANNED WORDS: "shows", "suggests", "might", "appears", "contradiction", "maybe", "I think".
REQUIRED VOCABULARY: "CONFIRMED", "INVALIDATED", "CONVERGENCE", "DIVERGENCE", "TESTING", "REJECTING", "ALIGNMENT".

DATA INPUTS PROVIDED:
`price`, `ema_34` (fast), `ema_89` (baseline), `ema_144` (macro S/R), `rsi_13`, `rsi_47`, `rsi_delta`, `calculated_support`, `calculated_resistance`. (Data includes both 4H Macro and 15m Micro timeframes).

OUTPUT INSTRUCTIONS:
You must return a strictly formatted response adhering to the following schema. Use pipe operators (|) for multi-variable states.

1. general_analysis: A highly dense, cold, quantitative summary of the structural state. No fluff. Evaluate the synergy or conflict between the 4H and 15m timeframes.
2. trend_state: Format as `MACRO: [STATE] | MICRO: [STATE] | STATUS: [ALIGNMENT/CONFLICT]`. (Use EMA hierarchy to define state).
3. momentum_divergence: Format as `RSI_FAST: [VALUE] | RSI_SLOW: [VALUE] | DELTA: [VALUE] | STATE: [ACCELERATION/DECAY/OVERSOLD/OVERBOUGHT]`.
4. key_level_interaction: Note the exact distance of `price` to `calculated_support` and `calculated_resistance`. Format as `THREAT: [SUPPORT/RESISTANCE] | DISTANCE: [X]% | ACTION: [TESTING/REJECTING/CLEAR]`.
5. bias: Must be exactly one of the following based strictly on the data: STRONGLY_BULLISH, BULLISH, NEUTRAL, BEARISH, STRONGLY_BEARISH.

EXECUTION:
Synthesize the provided JSON payload into the schema above. Prioritize mathematical exactness over narrative.

--- MARKET DATA ---
{tech_payload}
"""


def build_volume_prompt(vol_payload: str) -> str:
    """Agent 2: Liquidity/Volume Analyst."""
    return f"""SYSTEM PROMPT:
You are the Lead Volume & Liquidity Analyst for a quantitative prop firm. Your objective is to map institutional capital flows, volume node acceptance/rejection, and Value Area (VA) transitions.

YOU ARE STRICTLY FORBIDDEN from using conversational filler or narrative forecasting.
BANNED WORDS: "shows", "suggests", "might", "appears", "chopping", "fair value".
REQUIRED VOCABULARY: "ACCEPTANCE", "REJECTION", "EXPANSION", "CONTRACTION", "ROTATION", "LIQUIDITY_VOID".

DATA INPUTS PROVIDED:
`price`, `volume`, `vma_20`, `poc_price`, `vah`, `val`, `price_to_va_status`, `distance_to_poc_percent`.

OUTPUT INSTRUCTIONS:
You must return a strictly formatted response adhering to the following schema. Use pipe operators (|) for multi-variable states.

1. general_analysis: A highly dense, quantitative summary of institutional flow. Assess if volume supports the current price action and identify where price is relative to the Value Area.
2. liquidity_state: Format as `STATUS: [price_to_va_status] | ACTION: [MEAN_REVERSION / BREAKOUT_DISCOVERY / RANGE_ROTATION]`.
3. volume_momentum: Format as `VOL_VS_VMA: [Ratio/Difference] | STATE: [EXPANSION / CONTRACTION / ANOMALY]`.
4. magnet_target: State the exact price of the primary liquidity draw (typically the `poc_price` if inside VA, or next high-volume node if outside). Format as `TARGET: [PRICE] | DISTANCE: [X]%`.
5. bias: Must be exactly one of the following based strictly on the data: STRONGLY_BULLISH, BULLISH, NEUTRAL, BEARISH, STRONGLY_BEARISH.

EXECUTION:
Synthesize the provided JSON payload into the schema above. Track the math, map the liquidity.

--- MARKET DATA ---
{vol_payload}
"""


def build_defensive_prompt(tech_report: dict, vol_report: dict, data_15m: dict) -> str:
    """Agent 3: Lead Market Strategist, Defensive."""
    return f"""SYSTEM PROMPT:
You are the Lead Portfolio Manager for a deterministic Quantitative AI Execution Engine.
You do NOT look at raw market data. Your function is to synthesize the structured reports from the Technical Analyst (Agent 1) and the Volume & Liquidity Analyst (Agent 2).

YOUR PRIORITIES:
1. Capital Preservation: Any conflict between Macro/Micro timeframes or Technicals/Volume equals an immediate stand-down.
2. Confluence Validation: For an execution command, Agent 1 and Agent 2 must align in their Biases.
3. Risk-to-Reward (R:R) Asymmetry: Price must be optimally positioned against structural invalidation levels with a clear path to liquidity targets.

OUTPUT INSTRUCTIONS:
Return a strictly formatted response adhering to the following schema:

1. executive_summary: Write a highly professional, deep, 3-4 sentence human-readable analysis. Synthesize the core structural thesis, liquidity flow, and the institutional logic behind your final decision. Read like a prop-firm portfolio manager addressing the trading desk.
2. confluence_matrix: Format as `STRUCTURE: [ALIGN/CONFLICT] | VOLUME: [SUPPORTIVE/DECAY] | ACTION: [VALID/INVALID]`.
3. risk_vector: Format as `THREAT_PROXIMITY: [HIGH/LOW] | MAGNET_PULL: [STRONG/WEAK] | OVERALL_RISK: [ASYMMETRIC/TOXIC]`.
4. final_verdict: Must be EXACTLY ONE of the following literals: "GO LONG", "GO SHORT", "SIT ON HANDS".

EXECUTION LOGIC:
If Timeframes contradict -> SIT ON HANDS.
If Volume is contracting while Price tests Resistance -> SIT ON HANDS.
If Agent 1 Bias = NEUTRAL or Agent 2 Bias = NEUTRAL -> SIT ON HANDS.
If Technicals and Flow align perfectly with Asymmetric Risk -> GO LONG / GO SHORT.

--- SUB-AGENT REPORTS ---
Technical Agent Report:
{json.dumps(tech_report, indent=2)}

Volume Agent Report:
{json.dumps(vol_report, indent=2)}

--- CURRENT 15m MARKET DATA (For Context) ---
Price: ${data_15m.get('price', 0)}
"""


def build_greedy_prompt(tech_report: dict, vol_report: dict, data_15m: dict) -> str:
    """Agent 3: Lead Market Strategist, Greedy."""
    return f"""SYSTEM PROMPT:
You are the Lead Portfolio Manager and Chief Risk Officer for a deterministic Quantitative AI Execution Engine.
You do NOT look at raw market data. Your function is to synthesize the structured reports from the Technical Analyst (Agent 1) and the Volume & Liquidity Analyst (Agent 2) to identify asymmetric trading opportunities.

YOUR PRIORITIES:
1. Contextual Dominance over Consensus: You do not require democratic agreement between agents. A "NEUTRAL" volume reading is not a veto; it often signals balance before expansion. Contradictory timeframes frequently present highly profitable mean-reversion or pullback reload opportunities.
2. Mathematical Expectancy (Asymmetry): You must rigorously evaluate the distance between the current price, Agent 1's structural threat (Invalidation Level), and Agent 2's Magnet/Target.
3. Intelligent Aggression: Do not default to "SIT ON HANDS" at the first sign of conflict. If an asset is resting on heavy support with exhausted selling volume and a high upside magnet, that is a prime asymmetric LONG setup, even if the strict LTF trend state is "bearish".

OUTPUT INSTRUCTIONS:
Return a strictly formatted response adhering to the following schema:

1. executive_summary: Write a highly professional, deep, 3-4 sentence human-readable analysis. Synthesize the core structural thesis, liquidity flow, and explicitly weigh the mathematical asymmetry. Read like a prop-firm portfolio manager instructing the execution desk.
2. confluence_matrix: Format as `STRUCTURE: [STATE] | VOLUME: [STATE] | EDGE: [MEAN-REVERSION / TREND-CONTINUATION / CHOP]`.
3. risk_vector: Format as `THREAT_PROXIMITY: [HIGH/LOW] | MAGNET_PULL: [STRONG/WEAK] | ASYMMETRY: [FAVORABLE / UNFAVORABLE]`.
4. final_verdict: Must be EXACTLY ONE of the following literals: "GO LONG", "GO SHORT", "SIT ON HANDS".

EXECUTION LOGIC:
- Rule of Exhaustion: If Agent 1's momentum shows oversold/overbought conditions at a structural Key Level AND Agent 2 reports volume contraction, this confirms exhaustion. -> GO LONG / GO SHORT (Mean Reversion).
- Rule of Initiative: If price is breaking a Key Level and Agent 2 reports expanding volume outside the Value Area, this confirms acceptance. -> GO LONG / GO SHORT (Trend Continuation).
- Rule of Asymmetry: Mentally calculate $R:R = \\frac{{|Magnet Target - Current Price|}}{{|Current Price - Nearest Structural Threat|}}$. If Asymmetry is highly favorable and volume is not actively opposing the setup, execute the trade regardless of isolated timeframe conflicts.
- Veto Condition: ONLY output "SIT ON HANDS" if price is trapped in the middle of a range with no clear magnet, OR if high-momentum volume is aggressively opposing the structural level (e.g., heavy expanding volume smashing through support).

--- SUB-AGENT REPORTS ---
Technical Agent Report:
{json.dumps(tech_report, indent=2)}

Volume Agent Report:
{json.dumps(vol_report, indent=2)}

--- CURRENT 15m MARKET DATA (For Context) ---
Price: ${data_15m.get('price', 0)}
"""
