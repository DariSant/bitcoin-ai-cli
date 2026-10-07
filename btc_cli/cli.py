"""Typer commands: argument parsing and terminal output only; the work happens in pipeline.py."""

import logging
import os

import typer
from google import genai
from rich import box
from rich.panel import Panel
from rich.table import Table

from btc_cli import agents, pipeline
from btc_cli.console import console

# Create the Typer app instance
app = typer.Typer(help="Bitcoin AI CLI Tool")


def _strategy_flags(def_flag: bool, greed_flag: bool) -> tuple[bool, bool]:
    """(run_def, run_greed) from --def/--greed; both flags at once is an error."""
    if def_flag and greed_flag:
        console.print("[red]ERROR: Cannot pass both flags. To run both, omit flags entirely.[/red]")
        raise typer.Exit(1)

    run_def = True if not greed_flag else False
    run_greed = True if not def_flag else False
    return run_def, run_greed


@app.command("status")
def status_command(symbol: str = typer.Argument("BTC/USDT")):
    """
    Fetch MTF (4h, 15m) data for the given symbol and print the raw metrics.
    No AI analysis is executed.
    """
    pipeline.run_status(symbol)

@app.command("analyze")
def analyze_command(
    symbol: str = typer.Argument("BTC/USDT"),
    def_flag: bool = typer.Option(False, "--def", help="Run ONLY the Defensive Strategy"),
    greed_flag: bool = typer.Option(False, "--greed", help="Run ONLY the Greedy Strategy")
):
    """
    Fetch MTF (4h, 15m) data and execute trading analysis via AI agents.
    Outputs the Lead Market Strategist thesis for selected strategies.
    """
    run_def, run_greed = _strategy_flags(def_flag, greed_flag)
    if pipeline.run_analyze(symbol, run_def=run_def, run_greed=run_greed):
        raise typer.Exit(code=1)  # a damaged ledger or history blocked a strategy

@app.command("operate")
def operate_command(
    symbol: str = typer.Argument("BTC/USDT"),
    def_flag: bool = typer.Option(False, "--def", help="Run ONLY the Defensive Strategy"),
    greed_flag: bool = typer.Option(False, "--greed", help="Run ONLY the Greedy Strategy")
):
    """
    Execute trading operations based on recent analysis.
    """
    run_def, run_greed = _strategy_flags(def_flag, greed_flag)
    if pipeline.run_operate(symbol, run_def=run_def, run_greed=run_greed):
        raise typer.Exit(code=1)  # a damaged ledger or history blocked a strategy

@app.command("mock")
def mock_command(filename: str = typer.Argument(..., help="The mock payload file name (e.g. mock_long.json)")):
    """
    Feed a mock JSON payload directly to Agent 4 (The Operator) for stress testing.
    """
    try:
        pipeline.run_mock(filename)
    except (RuntimeError, ValueError) as e:
        logging.error(f"Mock command failed: {e}", exc_info=True)
        console.print(Panel(f"[red]ERROR: {e}[/red]", title="[red]System Failure[/red]", border_style="red", box=box.ROUNDED, expand=False))
        raise typer.Exit(code=1)
    except genai.errors.APIError as e:
        logging.error(f"Gemini API Error in mock command: {e}", exc_info=True)
        console.print(Panel("[red]ERROR: AI processing failed. Check error.log for details.[/red]", title="[red]System Failure[/red]", border_style="red", box=box.ROUNDED, expand=False))
        raise typer.Exit(code=1)
    except Exception as e:
        logging.error(f"Unexpected error in mock command: {e}", exc_info=True)
        console.print(Panel("[red]ERROR: An unexpected failure occurred. Check error.log.[/red]", title="[red]System Failure[/red]", border_style="red", box=box.ROUNDED, expand=False))
        raise typer.Exit(code=1)


@app.command("auto")
def auto_command(
    symbol: str = typer.Argument("BTC/USDT"),
    def_flag: bool = typer.Option(False, "--def", help="Run ONLY the Defensive Strategy"),
    greed_flag: bool = typer.Option(False, "--greed", help="Run ONLY the Greedy Strategy")
):
    """
    Execute the entire sequential pipeline: Status -> Analyze -> Operate.
    """
    run_def, run_greed = _strategy_flags(def_flag, greed_flag)

    pipeline.run_status(symbol)
    # Both steps run even if a damaged file blocks one strategy, so the healthy one keeps trading.
    analyze_damaged = pipeline.run_analyze(symbol, run_def=run_def, run_greed=run_greed)
    operate_damaged = pipeline.run_operate(symbol, run_def=run_def, run_greed=run_greed)
    if analyze_damaged or operate_damaged:
        raise typer.Exit(code=1)

@app.command()
def ask(question: str):
    """
    Ask the AI model a question and print the response.
    """
    # Ensure the Gemini API key is loaded securely
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        typer.secho("Error: GEMINI_API_KEY environment variable is missing. Please set it in your .env file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Initialize the Google GenAI client
        client = genai.Client(api_key=api_key)

        typer.secho(f"Thinking...", fg=typer.colors.YELLOW)

        # Use the global fallback wrapper
        # schema_class is None because this is a raw prompt, no strict json schema required
        response_text, active_model = agents.query_llm_with_fallback(client, question, None, "ask_agent", "N/A")

        if response_text is None:
            typer.secho("\n❌ Error: Both models unreachable. Please try again later.\n", fg=typer.colors.RED, bold=True)
            raise typer.Exit(code=1)

        # Print the response cleanly to the console
        typer.secho(f"\n🤖 AI Response (Model: {active_model}):", fg=typer.colors.MAGENTA, bold=True)
        typer.secho("-" * 40, fg=typer.colors.MAGENTA)
        typer.echo(response_text)
        typer.secho("-" * 40 + "\n", fg=typer.colors.MAGENTA)

    except genai.errors.APIError as e:
        # Handle errors directly from the Gemini API
        typer.secho(f"API Error: Failed to generate content. Please check your API key and connection. Details: {e}", fg=typer.colors.RED)
    except Exception as e:
        # Catch any other unexpected errors (known issue: this also catches the Exit above, so it exits 0)
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)


@app.command(name="commands", help="Displays a matrix of all executable commands and their variables.")
def commands_command():
    """
    Displays a matrix of all executable commands and their variables.
    """
    table = Table(box=box.ROUNDED)
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Variables / Strategies", style="yellow")

    table.add_row(
        "status",
        "Fetch MTF (4h, 15m) data for the given symbol and print the raw metrics.",
        r"\[SYMBOL]"
    )
    table.add_row(
        "analyze",
        "Fetch MTF (4h, 15m) data and execute trading analysis via AI agents.",
        r"\[SYMBOL] | --def (Defensive) | --greed (Greedy)"
    )
    table.add_row(
        "operate",
        "Execute trading operations based on recent analysis.",
        r"\[SYMBOL] | --def (Defensive) | --greed (Greedy)"
    )
    table.add_row(
        "mock",
        "Feed a mock JSON payload directly to Agent 4 (The Operator) for stress testing.",
        r"\[filename]"
    )
    table.add_row(
        "auto",
        "Execute the entire sequential pipeline: Status -> Analyze -> Operate.",
        r"\[SYMBOL] | --def (Defensive) | --greed (Greedy)"
    )
    table.add_row(
        "ask",
        "Ask the AI model a question and print the response.",
        r"\[question]"
    )
    table.add_row(
        "commands",
        "Displays a matrix of all executable commands and their variables.",
        "None"
    )

    console.print(table)
