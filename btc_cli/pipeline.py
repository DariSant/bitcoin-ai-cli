"""The command flows: status, analyze, operate, mock, and the open-trade check they share.

Moved from app.py without behaviour changes; known bugs are listed in TODO.md and
pinned by tests/characterization/.
"""

import json
import logging
import os
import pathlib
from datetime import datetime, timezone

import typer
from google import genai
from rich import box
from rich.panel import Panel

from btc_cli import agents, config, data, ledger, storage, trade_operator
from btc_cli.console import (
    console,
    render_execution_ticket,
    render_invalid_ticket,
    render_manager_report,
    render_technical_report,
    render_volume_report,
)
from btc_cli.indicators import calculate_indicators


def check_open_positions(symbol: str, strategy: str) -> bool:
    """
    Pre-flight check for open positions. Reads BASE_DIR/{strategy}/{symbol}_paper_ledger.json.
    If OPEN, fetches recent 15m candles to see if TP/SL was hit.
    If hit, logs to history and returns False (proceed). If not hit, returns True (skip).
    If no open position, returns False (proceed).
    """
    # Ensure strategy directory exists
    storage.ensure_strategy_dir(strategy)

    ledger_path = storage.ledger_path(strategy, symbol)
    history_path = storage.history_path(strategy, symbol)

    trade = storage.read_ledger(ledger_path)
    if trade is None:
        return False

    if trade.get("status") != "OPEN":
        return False

    if not trade.get("entry_timestamp"):
        return False

    try:
        ohlcv = data.fetch_resolution_candles(symbol)
    except Exception as e:
        typer.secho(f"\n❌ Error fetching data to verify open positions: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    exit_ = ledger.find_exit(trade, ohlcv)

    if exit_ is not None:
        pnl_usd = ledger.calculate_pnl(trade, exit_.result)
        # The trade keeps the strategy_version it was opened with; this records the rules that resolved it.
        closed_trade = {
            **ledger.close_trade(trade, exit_, pnl_usd),
            "resolved_at_utc": storage.utc_iso(datetime.now(timezone.utc)),
            "resolved_by_strategy_version": config.STRATEGY_VERSION,
        }
        storage.move_to_history(ledger_path, history_path, closed_trade)

        console.print(Panel(
            f"[bold]Trade Closed ({strategy.upper()}):[/bold] {exit_.result}\n[bold]PnL:[/bold] ${pnl_usd:,.2f}",
            title="[Position Update]", border_style="cyan", box=box.ROUNDED, expand=False
        ))
        # Allow pipeline to proceed
        return False

    render_execution_ticket(trade, strategy)
    console.print(f"[yellow]Status: Active position detected for {strategy.upper()}. Execution halted until TP/SL resolution.[/yellow]")
    return True


def _print_timeframe_metrics(title: str, d: dict) -> None:
    typer.secho(title, fg=typer.colors.BLUE, bold=True)
    typer.echo(f"Current Price : ${d['price']:,.2f}")
    typer.echo(f"Volume / VMA (20) : {d['volume']} / {d['vma_20']}")
    typer.echo(f"Volume Profile POC: ${d['poc_price']}")
    typer.echo(f"EMAs (34,89,144) : {d['ema_34']}, {d['ema_89']}, {d['ema_144']}")
    typer.echo(f"RSI (13,47)      : {d['rsi_13']}, {d['rsi_47']}")
    typer.echo(f"RSI Delta        : {d['rsi_delta']}")
    typer.echo(f"Volatility (ATR 14): ${d['atr_14']}")
    typer.echo(f"Value Area (VAL - VAH): ${d['val']} - ${d['vah']} ({d['va_status']})")
    typer.echo(f"Mean Reversion: {d['dist_144_percent']}% from EMA | {d['dist_poc_percent']}% from POC")


def run_status(symbol: str = 'BTC/USDT') -> None:
    """
    Fetch MTF (4h, 15m) data for a given symbol and print the raw metrics.
    No AI analysis is executed.
    """
    try:
        exchange = data.create_exchange()

        # Fetch and analyze data for both timeframes
        df_4h = data.fetch_ohlcv_data(exchange, symbol, '4h')
        data_4h = calculate_indicators(df_4h, '4h')
        df_15m = data.fetch_ohlcv_data(exchange, symbol, '15m')
        data_15m = calculate_indicators(df_15m, '15m')

    except (RuntimeError, ValueError, Exception) as e:
        logging.error("Failed to calculate metrics", exc_info=True)
        typer.secho(f"\n❌ Error: Could not calculate metrics. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

    # Print the raw metrics cleanly to the console (header hardcodes BTC/USDT, see TODO.md cleanup item)
    typer.secho("\n📊 Multi-Timeframe Analysis (Binance: BTC/USDT)", fg=typer.colors.CYAN, bold=True)
    typer.secho("=" * 60, fg=typer.colors.CYAN)
    _print_timeframe_metrics("📈 4H Macro Trend Metrics", data_4h)
    typer.secho("-" * 60, fg=typer.colors.CYAN)
    _print_timeframe_metrics("📉 15m Micro Trend Metrics", data_15m)
    typer.secho("=" * 60 + "\n", fg=typer.colors.CYAN)

    # Status is saved under the "system" strategy
    filepath = storage.log_execution("status", "system", symbol, data_4h, data_15m)
    now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    console.print(f"[dim]💾 [{now_utc_str}] Footprint saved to: {filepath}[/dim]")


def _reject_ticket(final_verdict: str, current_price, threat_level, magnet_target, order: trade_operator.OrderCalc) -> None:
    """Show and record a ticket whose stop or target contradicts its direction."""
    render_invalid_ticket(final_verdict, current_price, threat_level, magnet_target, order.stop_loss, order.take_profit)

    timestamp_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    log_entry = (
        f"[{timestamp_str}] ERROR: INVALID TICKET LOGIC | "
        f"Verdict: {final_verdict} | "
        f"Price: {current_price} | "
        f"Threat: {threat_level} | "
        f"Magnet: {magnet_target} | "
        f"SL: {order.stop_loss} | "
        f"TP: {order.take_profit}\n"
    )
    storage.append_operator_error(log_entry)


def run_operate(symbol: str = 'BTC/USDT', run_def: bool = True, run_greed: bool = True) -> None:
    """
    Execute trading operations based on recent analysis.
    """
    strategies_to_run = []
    if run_def and not check_open_positions(symbol, "defensive"):
        strategies_to_run.append("defensive")
    if run_greed and not check_open_positions(symbol, "greedy"):
        strategies_to_run.append("greedy")

    # Known P2 issue: operate never calls Gemini but still requires the key and a client.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        typer.secho("Error: GEMINI_API_KEY environment variable is missing. Please set it in your .env file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    client = genai.Client(api_key=api_key)

    for strategy in strategies_to_run:
        json_files = storage.analysis_files(strategy)
        if not json_files:
            console.print(f"[yellow]No recent analysis found for {strategy.upper()} strategy.[/yellow]")
            continue

        most_recent_file = storage.latest_analysis_for_symbol(json_files, symbol)
        if most_recent_file is None:
            console.print(f"[yellow]No recent analysis found for {symbol} on {strategy.upper()} strategy.[/yellow]")
            continue

        try:
            analysis = storage.read_analysis(most_recent_file)

            timestamp_str = analysis.get("metadata", {}).get("timestamp")
            if not timestamp_str:
                raise ValueError("No timestamp found in metadata.")

            file_time = datetime.fromisoformat(timestamp_str)
            now_dt = datetime.now(timezone.utc) if file_time.tzinfo else datetime.now()

            diff = now_dt - file_time
            if diff.total_seconds() > 600:
                console.print(f"[yellow]No recent analysis has been run in the last 10 minutes for {strategy.upper()}.[/yellow]")
                continue

            synthesis = analysis.get("agent_3_synthesis", {})
            final_verdict = synthesis.get("final_verdict", "SIT ON HANDS")

            if final_verdict == "SIT ON HANDS":
                console.print(f"\n[yellow]{strategy.capitalize()} Verdict: SIT ON HANDS. Bypassing execution.[/yellow]")
                continue

            console.print(f"\n[green]Proceeding to Agent 4 Execution for {strategy.upper()} Strategy...[/green]")

            # --- Python Short-Circuit Routing & Operator Payload Construction ---
            raw_15m = analysis.get("raw_market_data", {}).get("15m", {})

            current_price = raw_15m.get("price")
            atr_14 = raw_15m.get("atr_14")
            poc_price = raw_15m.get("poc_price")
            calculated_support = raw_15m.get("calculated_support")
            calculated_resistance = raw_15m.get("calculated_resistance")

            if final_verdict == "GO LONG":
                agent_1_threat_level = calculated_support
            else: # "GO SHORT"
                agent_1_threat_level = calculated_resistance

            raw_magnet_string = analysis.get("agent_2_volume", {}).get("magnet_target", "")

            agent_2_magnet_target = trade_operator.parse_magnet_target(raw_magnet_string)
            if agent_2_magnet_target is None:
                agent_2_magnet_target = poc_price
                console.print(f"[yellow]Warning: Could not parse Agent 2 Magnet '{raw_magnet_string}'. Falling back to POC.[/yellow]")

            operator_payload = {
                "verdict": final_verdict,
                "account_balance_usdt": 10000.0,
                "risk_per_trade_percent": 1.0,
                "current_price": current_price,
                "atr_14": atr_14,
                "agent_1_threat_level": agent_1_threat_level,
                "agent_2_magnet_target": agent_2_magnet_target
            }

            # --- The Python Operator ---
            entry_price = current_price
            order = trade_operator.compute_order(final_verdict, current_price, atr_14, agent_1_threat_level, agent_2_magnet_target)
            if order is None:
                continue
            if not order.valid:
                _reject_ticket(final_verdict, current_price, agent_1_threat_level, agent_2_magnet_target, order)
                continue

            operator_report = trade_operator.build_operator_report(entry_price, order)

            # Render Operator Panel using centralized ticket
            ticket_data = {
                "verdict": final_verdict,
                "order_type": operator_report.get("order_type"),
                "entry_price": operator_report.get("entry_price"),
                "stop_loss": operator_report.get("stop_loss"),
                "take_profit": operator_report.get("take_profit"),
                "risk_reward_ratio": operator_report.get("risk_reward_ratio"),
                "position_size_usd": operator_report.get("position_size_usd"),
                "entry_timestamp": datetime.now().timestamp()
            }
            render_execution_ticket(ticket_data, strategy)

            # --- Save Execution Footprint ---
            now_utc = datetime.now(timezone.utc)
            now = storage.local_now(now_utc)
            source = storage.analysis_link(most_recent_file, analysis)
            filepath = storage.write_execution_footprint(now, now_utc, strategy, symbol, operator_payload, operator_report, source)

            now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            console.print(f"[dim]💾 [{now_utc_str}] Execution Footprint saved to: {filepath}[/dim]")

            # --- Save Active Ledger (paper_ledger.json) ---
            ledger_path = storage.ledger_path(strategy, symbol)

            ledger_entry = {
                "symbol": symbol,
                "status": "OPEN",
                "entry_timestamp": now.timestamp(),
                "verdict": operator_payload["verdict"],
                "order_type": operator_report.get("order_type"),
                "entry_price": operator_report.get("entry_price"),
                "stop_loss": operator_report.get("stop_loss"),
                "take_profit": operator_report.get("take_profit"),
                "risk_reward_ratio": operator_report.get("risk_reward_ratio"),
                "position_size_usd": operator_report.get("position_size_usd"),
                **storage.record_header(),
                "strategy": strategy,
                "trade_id": storage.record_id(now_utc, strategy, symbol),
                "entry_time_utc": storage.utc_iso(now_utc),
                **source,
            }

            storage.write_ledger(ledger_path, ledger_entry)

            now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            console.print(f"[dim]💾 [{now_utc_str}] Active Ledger updated: {ledger_path}[/dim]")

        except typer.Exit:
            raise
        except Exception as e:
            logging.error(f"Failed to read or parse analysis file {most_recent_file} for {strategy}", exc_info=True)
            typer.secho(f"\n❌ Error: Could not read analysis file for {strategy}. Check error.log.\n", fg=typer.colors.RED, bold=True)
            continue


def run_mock(filename: str) -> None:
    """
    Feed a mock JSON payload directly to Agent 4 and display the Execution Ticket.
    """
    # Read and validate the payload
    filepath = pathlib.Path(f"mock_json/{filename}")
    if not filepath.exists() or not filepath.is_file():
        raise ValueError(f"File not found: {filepath}")

    try:
        with open(filepath, "r") as f:
            operator_payload = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in payload file: {e}")

    # Basic schema validation
    required_keys = ["verdict", "account_balance_usdt", "risk_per_trade_percent", "current_price", "atr_14", "agent_1_threat_level", "agent_2_magnet_target"]
    for key in required_keys:
        if key not in operator_payload:
            raise ValueError(f"Missing required key in payload: {key}")

    verdict = operator_payload.get("verdict", "")
    if verdict not in ("GO LONG", "GO SHORT"):
        raise ValueError(f"Invalid verdict '{verdict}'. Must be 'GO LONG' or 'GO SHORT'.")

    current_price = operator_payload["current_price"]
    order = trade_operator.compute_mock_order(
        verdict,
        current_price,
        operator_payload["atr_14"],
        operator_payload["agent_1_threat_level"],
        operator_payload["agent_2_magnet_target"],
    )

    if not order.valid:
        console.print(Panel("[bold red]⛔ INVALID TICKET: Mathematical risk logic or Magnet target contradicts the trade direction. Execution halted.[/bold red]", border_style="red", box=box.ROUNDED, expand=False))
        return

    operator_report = trade_operator.build_operator_report(current_price, order)

    # Render Operator Panel using centralized ticket
    ticket_data = {
        "verdict": verdict,
        "order_type": operator_report.get("order_type"),
        "entry_price": operator_report.get("entry_price"),
        "stop_loss": operator_report.get("stop_loss"),
        "take_profit": operator_report.get("take_profit"),
        "risk_reward_ratio": operator_report.get("risk_reward_ratio"),
        "position_size_usd": operator_report.get("position_size_usd"),
        "entry_timestamp": datetime.now().timestamp()
    }
    render_execution_ticket(ticket_data, "MOCK")


def _ai_failed(message: str) -> typer.Exit:
    logging.error(message, exc_info=True)
    typer.secho("\n❌ Error: AI processing failed. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
    return typer.Exit(code=1)


def _route_to_beta_if_fallback(active_model: str | None) -> None:
    """Known P0 bug: one fallback answer moves the rest of this process to output_beta."""
    if active_model == config.FALLBACK_MODEL:
        config.BASE_DIR = "output_beta"


def _run_manager(client: genai.Client, symbol: str, strategy: str, prompt: str, tech_report: dict, vol_report: dict, data_4h: dict, data_15m: dict, models_used: dict[str, str]) -> None:
    """Agent 3 for one strategy: ask, show the synthesis and save the analysis footprint.

    `models_used` holds the models that answered Agents 1 and 2; this strategy's Agent 3 is added to a copy.
    """
    label = strategy.capitalize()
    with console.status(f"[bold cyan]Agent 3 ({label} Manager) Thinking... (Model: {config.PRIMARY_MODEL})[/bold cyan]", spinner="dots"):
        manager_text, active_model = agents.query_llm_with_fallback(client, prompt, agents.Agent3ManagerSchema, f"agent_3_{strategy}", symbol)

    if manager_text is None:
        # Skip only this strategy; the other one may still run.
        console.print("[bold red][CRITICAL] Both models unreachable. Skipping cycle.[/bold red]")
        return

    _route_to_beta_if_fallback(active_model)

    try:
        report = json.loads(manager_text)
    except json.JSONDecodeError:
        raise _ai_failed(f"Failed to parse Agent 3 ({label}) JSON output. Raw text: {manager_text}")

    title, border = ("[Defensive Strategy Synthesis]", "magenta") if strategy == "defensive" else ("[Greedy Strategy Synthesis]", "yellow")
    render_manager_report(report, title, border)

    models_used = {**models_used, f"agent_3_{strategy}": active_model}
    filepath = storage.log_execution("analyze", strategy, symbol, data_4h, data_15m, tech_report, vol_report, report, models_used=models_used)
    now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    console.print(f"[dim]💾 [{now_utc_str}] {label} Footprint saved to: {filepath}[/dim]")


def run_analyze(symbol: str = 'BTC/USDT', run_def: bool = True, run_greed: bool = True) -> None:
    """
    Fetch MTF (4h, 15m) data for a given symbol and execute trading analysis via AI agents.
    Outputs the Lead Market Strategist thesis for active strategies.
    """
    skip_def = check_open_positions(symbol, "defensive") if run_def else True
    skip_greed = check_open_positions(symbol, "greedy") if run_greed else True

    # If every requested strategy already has an open trade, there is nothing to analyze.
    # Stop here so we don't waste AI calls on Agents 1 and 2 whose reports would be thrown away.
    if skip_def and skip_greed:
        console.print("[yellow]All requested strategies have open positions. Skipping AI analysis to save API calls.[/yellow]")
        return

    # Ensure the Gemini API key is loaded securely
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        typer.secho("Error: GEMINI_API_KEY environment variable is missing. Please set it in your .env file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        exchange = data.create_exchange()

        # Fetch and analyze data for both timeframes (silently)
        df_4h = data.fetch_ohlcv_data(exchange, symbol, '4h')
        data_4h = calculate_indicators(df_4h, '4h')
        df_15m = data.fetch_ohlcv_data(exchange, symbol, '15m')
        data_15m = calculate_indicators(df_15m, '15m')

    except (RuntimeError, ValueError, Exception) as e:
        logging.error("Failed to calculate metrics", exc_info=True)
        typer.secho(f"\n❌ Error: Could not calculate metrics. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

    try:
        client = genai.Client(api_key=api_key)

        tech_payload = agents.build_tech_payload(data_4h, data_15m)
        vol_payload = agents.build_vol_payload(data_4h, data_15m)

        # --- Agent 1: Technical Analyst ---
        with console.status(f"[bold cyan]Agent 1 (Technical Analyst) Thinking... (Model: {config.PRIMARY_MODEL})[/bold cyan]", spinner="dots"):
            agent1_text, active_model = agents.query_llm_with_fallback(client, agents.build_technical_prompt(tech_payload), agents.Agent1TechSchema, "agent_1_technical", symbol)

        if agent1_text is None:
            # Known P1 bug: the run still exits 0.
            console.print("[bold red][CRITICAL] Both models unreachable. Skipping cycle.[/bold red]")
            return

        _route_to_beta_if_fallback(active_model)
        models_used = {"agent_1_technical": active_model}

        try:
            tech_report = json.loads(agent1_text)
        except json.JSONDecodeError:
            raise _ai_failed(f"Failed to parse Agent 1 (Technical) JSON output. Raw text: {agent1_text}")

        render_technical_report(tech_report)

        # --- Agent 2: Liquidity/Volume Analyst ---
        with console.status(f"[bold cyan]Agent 2 (Liquidity/Volume Analyst) Thinking... (Model: {config.PRIMARY_MODEL})[/bold cyan]", spinner="dots"):
            agent2_text, active_model = agents.query_llm_with_fallback(client, agents.build_volume_prompt(vol_payload), agents.Agent2VolumeSchema, "agent_2_volume", symbol)

        if agent2_text is None:
            console.print("[bold red][CRITICAL] Both models unreachable. Skipping cycle.[/bold red]")
            return

        _route_to_beta_if_fallback(active_model)
        models_used["agent_2_volume"] = active_model

        try:
            vol_report = json.loads(agent2_text)
        except json.JSONDecodeError:
            raise _ai_failed(f"Failed to parse Agent 2 (Volume) JSON output. Raw text: {agent2_text}")

        render_volume_report(vol_report)

        # --- Agent 3: Lead Market Strategist, one call per strategy ---
        if not skip_def:
            _run_manager(client, symbol, "defensive", agents.build_defensive_prompt(tech_report, vol_report, data_15m), tech_report, vol_report, data_4h, data_15m, models_used)

        if not skip_greed:
            _run_manager(client, symbol, "greedy", agents.build_greedy_prompt(tech_report, vol_report, data_15m), tech_report, vol_report, data_4h, data_15m, models_used)

    except typer.Exit:
        # Re-raise Typer's Exit exception so the CLI can exit gracefully
        raise
    except genai.errors.APIError as e:
        logging.error("Gemini API Error", exc_info=True)
        typer.secho("\n❌ Error: AI processing failed. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)
    except Exception as e:
        logging.error("Unexpected error during AI analysis", exc_info=True)
        typer.secho("\n❌ Error: AI processing failed. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)
