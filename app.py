import os
import json
import typer
import ccxt
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from google import genai
import typing
from rich.console import Console
from datetime import datetime, timezone
from rich.panel import Panel
from rich import box
import logging
import pathlib

# Load environment variables from the .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Instantiate the Rich console
console = Console()

# Create the Typer app instance
app = typer.Typer(help="Bitcoin AI CLI Tool")

# --- Global AI Schemas & Helpers ---

class Agent1TechSchema(typing.TypedDict):
    bias: typing.Literal["BULLISH", "BEARISH", "NEUTRAL"]
    trend_state: str
    momentum_divergence: str
    key_level_interaction: str

class Agent2VolumeSchema(typing.TypedDict):
    bias: typing.Literal["BULLISH", "BEARISH", "NEUTRAL"]
    liquidity_state: str
    volume_momentum: str
    magnet_target: str

class Agent3ManagerSchema(typing.TypedDict):
    technical_synthesis: str
    risk_profile: str
    confidence_score: int # Must be an integer between 0 and 100
    final_verdict: typing.Literal["GO LONG", "GO SHORT", "SIT ON HANDS"]

def get_bias_color(bias: str) -> str:
    if bias in ("STRONGLY_BULLISH", "BULLISH"):
        return "green"
    elif bias in ("STRONGLY_BEARISH", "BEARISH"):
        return "red"
    return "yellow"

def log_execution(command_name: str, symbol: str, data_4h: dict, data_15m: dict, agent1_report: dict = None, agent2_report: dict = None, agent3_report: dict = None) -> str:
    """
    Universally log execution state to a JSON footprint in output_alpha/.
    """
    now = datetime.now()
    directory_path = f"output_alpha/{command_name}/{now.strftime('%Y-%m')}/"
    os.makedirs(directory_path, exist_ok=True)

    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{symbol.replace('/', '')}.json"
    filepath = os.path.join(directory_path, filename)

    payload = {
        "metadata": {
            "timestamp": now.isoformat(),
            "symbol": symbol,
            "command_run": command_name
        },
        "raw_market_data": {
            "4h": data_4h,
            "15m": data_15m
        }
    }

    # Only add AI reports if they were passed to the function
    if agent1_report: payload["agent_1_technical"] = agent1_report
    if agent2_report: payload["agent_2_volume"] = agent2_report
    if agent3_report: payload["agent_3_synthesis"] = agent3_report

    with open(filepath, "w") as file:
        json.dump(payload, file, indent=2)

    return filepath

def fetch_and_analyze(exchange, symbol: str, timeframe: str) -> dict:
    """
    Fetch OHLCV data for a given symbol and timeframe, calculate technical indicators
    (EMAs, RSIs, and RSI Delta) using pandas-ta, and return the latest row's values.
    """
    try:
        # Fetch the last 200 candles
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    except ccxt.NetworkError as e:
        raise RuntimeError(f"Network error fetching data for {timeframe} timeframe: {e}")
    except ccxt.ExchangeError as e:
        raise RuntimeError(f"Exchange error fetching data for {timeframe} timeframe: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching data for {timeframe} timeframe: {e}")

    if not ohlcv:
        raise RuntimeError(f"No data returned for {timeframe} timeframe.")

    # Convert to Pandas DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Calculate EMAs
    df['EMA_34'] = ta.ema(df['close'], length=34)
    df['EMA_89'] = ta.ema(df['close'], length=89)
    df['EMA_144'] = ta.ema(df['close'], length=144)

    # Calculate RSIs
    df['RSI_13'] = ta.rsi(df['close'], length=13)
    df['RSI_47'] = ta.rsi(df['close'], length=47)

    # Calculate RSI Delta
    df['RSI_Delta'] = df['RSI_13'] - df['RSI_47']

    # Calculate Volume Moving Average (VMA)
    df['VMA_20'] = ta.sma(df['volume'], length=20)

    # Calculate ATR (Volatility)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Calculate Structural Levels (Swing High / Swing Low)
    df['swing_high'] = df['high'].rolling(20).max()
    df['swing_low'] = df['low'].rolling(20).min()

    # Calculate Volume Profile Point of Control (POC) and Value Area
    # Create 10 equal price bins based on the close column
    bins = pd.cut(df['close'], bins=10)
    # Group by bins and sum the volume for each bin
    volume_by_bin = df.groupby(bins, observed=False)['volume'].sum()
    # Find the bin with the maximum volume
    max_volume_bin = volume_by_bin.idxmax()
    # Extract the midpoint price of that highest-volume bin
    poc_price = float(max_volume_bin.mid)

    # Calculate Value Area (VAH / VAL) - 70% True Distribution
    total_volume = volume_by_bin.sum()
    target_volume = total_volume * 0.70

    # Sort bins by volume descending
    sorted_bins = volume_by_bin.sort_values(ascending=False)

    accumulated_volume = 0
    selected_bins = []

    # Accumulate volume until we reach >= 70%
    for bin_interval, vol in sorted_bins.items():
        accumulated_volume += vol
        selected_bins.append(bin_interval)
        if accumulated_volume >= target_volume:
            break

    # Calculate VAL and VAH based on selected bins
    val = float(min(b.left for b in selected_bins))
    vah = float(max(b.right for b in selected_bins))

    # Check for NaNs on required indicators in the last row
    last_row = df.iloc[-1]

    if pd.isna(last_row['EMA_144']) or pd.isna(last_row['RSI_13']) or pd.isna(last_row['RSI_47']) or pd.isna(last_row['VMA_20']) or pd.isna(last_row['ATR_14']) or pd.isna(last_row['swing_high']) or pd.isna(last_row['swing_low']):
        raise ValueError(f"Insufficient candle data fetched to calculate required technical and volume metrics for {timeframe} timeframe.")

    current_price = float(last_row['close'])
    ema_144 = float(last_row['EMA_144'])

    # Calculate Distance Percentages
    distance_to_144_ema_percent = ((current_price - ema_144) / ema_144) * 100
    distance_to_poc_percent = ((current_price - poc_price) / poc_price) * 100

    # Logic for Value Area Status
    if current_price > vah:
        price_to_va_status = "BREAKING_ABOVE_VAH"
    elif current_price < val:
        price_to_va_status = "BREAKING_BELOW_VAL"
    else:
        price_to_va_status = "INSIDE_VALUE"

    # Return rounded values for the latest row
    return {
        'price': round(current_price, 2),
        'calculated_resistance': round(float(last_row['swing_high']), 2),
        'calculated_support': round(float(last_row['swing_low']), 2),
        'volume': round(float(last_row['volume']), 2),
        'vma_20': round(float(last_row['VMA_20']), 2),
        'poc_price': round(poc_price, 2),
        'ema_34': round(float(last_row['EMA_34']), 2),
        'ema_89': round(float(last_row['EMA_89']), 2),
        'ema_144': round(ema_144, 2),
        'rsi_13': round(float(last_row['RSI_13']), 2),
        'rsi_47': round(float(last_row['RSI_47']), 2),
        'rsi_delta': round(float(last_row['RSI_Delta']), 2),
        'atr_14': round(float(last_row['ATR_14']), 2),
        'vah': round(vah, 2),
        'val': round(val, 2),
        'dist_144_percent': round(distance_to_144_ema_percent, 2),
        'dist_poc_percent': round(distance_to_poc_percent, 2),
        'va_status': price_to_va_status,
    }

def _run_status(symbol: str = 'BTC/USDT'):
    """
    Internal helper to fetch MTF (4h, 15m) data for a given symbol and print the raw metrics.
    No AI analysis is executed.
    """
    try:
        # Initialize a CCXT exchange instance for Binance
        exchange = ccxt.binance()

        # Fetch and analyze data for both timeframes
        data_4h = fetch_and_analyze(exchange, symbol, '4h')
        data_15m = fetch_and_analyze(exchange, symbol, '15m')

    except (RuntimeError, ValueError, Exception) as e:
        logging.error("Failed to calculate metrics", exc_info=True)
        typer.secho(f"\n❌ Error: Could not calculate metrics. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

    # Print the raw metrics cleanly to the console
    typer.secho("\n📊 Multi-Timeframe Analysis (Binance: BTC/USDT)", fg=typer.colors.CYAN, bold=True)
    typer.secho("=" * 60, fg=typer.colors.CYAN)

    # 4H Print
    typer.secho("📈 4H Macro Trend Metrics", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"Current Price : ${data_4h['price']:,.2f}")
    typer.echo(f"Volume / VMA (20) : {data_4h['volume']} / {data_4h['vma_20']}")
    typer.echo(f"Volume Profile POC: ${data_4h['poc_price']}")
    typer.echo(f"EMAs (34,89,144) : {data_4h['ema_34']}, {data_4h['ema_89']}, {data_4h['ema_144']}")
    typer.echo(f"RSI (13,47)      : {data_4h['rsi_13']}, {data_4h['rsi_47']}")
    typer.echo(f"RSI Delta        : {data_4h['rsi_delta']}")
    typer.echo(f"Volatility (ATR 14): ${data_4h['atr_14']}")
    typer.echo(f"Value Area (VAL - VAH): ${data_4h['val']} - ${data_4h['vah']} ({data_4h['va_status']})")
    typer.echo(f"Mean Reversion: {data_4h['dist_144_percent']}% from EMA | {data_4h['dist_poc_percent']}% from POC")
    typer.secho("-" * 60, fg=typer.colors.CYAN)

    # 15M Print
    typer.secho("📉 15m Micro Trend Metrics", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"Current Price : ${data_15m['price']:,.2f}")
    typer.echo(f"Volume / VMA (20) : {data_15m['volume']} / {data_15m['vma_20']}")
    typer.echo(f"Volume Profile POC: ${data_15m['poc_price']}")
    typer.echo(f"EMAs (34,89,144) : {data_15m['ema_34']}, {data_15m['ema_89']}, {data_15m['ema_144']}")
    typer.echo(f"RSI (13,47)      : {data_15m['rsi_13']}, {data_15m['rsi_47']}")
    typer.echo(f"RSI Delta        : {data_15m['rsi_delta']}")
    typer.echo(f"Volatility (ATR 14): ${data_15m['atr_14']}")
    typer.echo(f"Value Area (VAL - VAH): ${data_15m['val']} - ${data_15m['vah']} ({data_15m['va_status']})")
    typer.echo(f"Mean Reversion: {data_15m['dist_144_percent']}% from EMA | {data_15m['dist_poc_percent']}% from POC")
    typer.secho("=" * 60 + "\n", fg=typer.colors.CYAN)

    filepath = log_execution("status", symbol, data_4h, data_15m)
    console.print(f"[dim]💾 Footprint saved to: {filepath}[/dim]")

def _run_operate(symbol: str = 'BTC/USDT'):
    """
    Internal helper to execute trading operations based on recent analysis.
    """
    analyze_dir = pathlib.Path("output_alpha/analyze")

    if not analyze_dir.exists():
        typer.secho("[yellow]No recent analysis has been run in the last 10 minutes.[/yellow]", fg=typer.colors.YELLOW)
        raise typer.Exit()

    json_files = list(analyze_dir.rglob("*.json"))
    if not json_files:
        typer.secho("[yellow]No recent analysis has been run in the last 10 minutes.[/yellow]", fg=typer.colors.YELLOW)
        raise typer.Exit()

    # Filter files for the requested symbol by reading the payload metadata
    symbol_files = []
    for file in json_files:
        if symbol.replace("/", "") in file.name:
            symbol_files.append(file)
        else:
            # Fallback to reading the file to check the symbol in metadata
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if data.get("metadata", {}).get("symbol") == symbol:
                        symbol_files.append(file)
            except Exception:
                pass

    if not symbol_files:
        typer.secho("[yellow]No recent analysis has been run in the last 10 minutes.[/yellow]", fg=typer.colors.YELLOW)
        raise typer.Exit()

    # Sort files by modification time descending to get the most recent one
    symbol_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    most_recent_file = symbol_files[0]

    try:
        with open(most_recent_file, "r") as f:
            data = json.load(f)

        timestamp_str = data.get("metadata", {}).get("timestamp")
        if not timestamp_str:
            raise ValueError("No timestamp found in metadata.")

        file_time = datetime.fromisoformat(timestamp_str)
        now = datetime.now(timezone.utc) if file_time.tzinfo else datetime.now()

        diff = now - file_time
        if diff.total_seconds() > 600:
            typer.secho("[yellow]No recent analysis has been run in the last 10 minutes.[/yellow]", fg=typer.colors.YELLOW)
            raise typer.Exit()

        synthesis = data.get("agent_3_synthesis", {})
        final_verdict = synthesis.get("final_verdict", "SIT ON HANDS")

        if final_verdict == "SIT ON HANDS":
            typer.secho("[yellow]Last analysis says that it is better not to operate given the actual market conditions.[/yellow]", fg=typer.colors.YELLOW)
            raise typer.Exit()

        # Display the strategist's analysis panel
        strategist_summary = (
            f"[bold]Final Verdict:[/bold] {final_verdict}\n"
            f"[bold]Confidence Score:[/bold] {synthesis.get('confidence_score', 0)}\n\n"
            f"[bold]Technical Synthesis:[/bold]\n{synthesis.get('technical_synthesis', '')}\n\n"
            f"[bold]Risk Profile:[/bold]\n{synthesis.get('risk_profile', '')}"
        )

        console.print(Panel(strategist_summary, title="[Lead Market Strategist - Synthesis]", border_style="magenta", box=box.ROUNDED, expand=False))
        console.print("[green]Proceeding to Agent 4 Execution...[/green]")
        return

    except typer.Exit:
        raise
    except Exception as e:
        logging.error(f"Failed to read or parse analysis file {most_recent_file}", exc_info=True)
        typer.secho(f"\n❌ Error: Could not read analysis file. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)


def _run_analyze(symbol: str = 'BTC/USDT'):
    """
    Internal helper to fetch MTF (4h, 15m) data for a given symbol and execute trading analysis via AI agents.
    Outputs the Lead Market Strategist thesis.
    """
    # Ensure the Gemini API key is loaded securely
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        typer.secho("Error: GEMINI_API_KEY environment variable is missing. Please set it in your .env file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Initialize a CCXT exchange instance for Binance
        exchange = ccxt.binance()

        # Fetch and analyze data for both timeframes (silently)
        data_4h = fetch_and_analyze(exchange, symbol, '4h')
        data_15m = fetch_and_analyze(exchange, symbol, '15m')

    except (RuntimeError, ValueError, Exception) as e:
        logging.error("Failed to calculate metrics", exc_info=True)
        typer.secho(f"\n❌ Error: Could not calculate metrics. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

    try:
        # Initialize the Google GenAI client
        client = genai.Client(api_key=api_key)

        # --- Agent 1: Technical Analyst ---
        agent1_prompt = (
            "You are the Lead Technical Analyst. Your objective is precise, data-driven analysis. "
            "Trend State: Analyze the alignment of the fast EMA (34) against the slow EMAs (89, 144) across 15m and 4H timeframes. State if they are in confluence or contradiction. "
            "Momentum: Analyze RSI (13, 47) and rsi_delta. State if overbought, oversold, or diverging. "
            "Key Levels: Look strictly at the calculated_support and calculated_resistance in the 4H and 15m JSON data. You MUST select the nearest structural threat from these lists and explain why. Do not invent prices.\n\n"
            "Return a strict JSON object with EXACTLY these keys: \n"
            "- bias (string: BULLISH, BEARISH, or NEUTRAL)\n"
            "- trend_state (string)\n"
            "- momentum_divergence (string)\n"
            "- key_level_interaction (string)\n\n"
            f"--- MARKET DATA ---\n"
            f"4H Data: Price: ${data_4h.get('price', 0)}, "
            f"Calculated Support: ${data_4h.get('calculated_support', 0)}, Calculated Resistance: ${data_4h.get('calculated_resistance', 0)}, "
            f"EMA(34,89,144): [{data_4h.get('ema_34', 0)}, {data_4h.get('ema_89', 0)}, {data_4h.get('ema_144', 0)}], "
            f"RSI(13,47): [{data_4h.get('rsi_13', 0)}, {data_4h.get('rsi_47', 0)}], RSI Delta: {data_4h.get('rsi_delta', 0)}\n"
            f"15m Data: Price: ${data_15m.get('price', 0)}, "
            f"Calculated Support: ${data_15m.get('calculated_support', 0)}, Calculated Resistance: ${data_15m.get('calculated_resistance', 0)}, "
            f"EMA(34,89,144): [{data_15m.get('ema_34', 0)}, {data_15m.get('ema_89', 0)}, {data_15m.get('ema_144', 0)}], "
            f"RSI(13,47): [{data_15m.get('rsi_13', 0)}, {data_15m.get('rsi_47', 0)}], RSI Delta: {data_15m.get('rsi_delta', 0)}"
        )

        with console.status("[bold cyan]Agent 1 (Technical Analyst) Thinking... (Model: gemini-2.5-flash)[/bold cyan]", spinner="dots"):
            agent1_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=agent1_prompt,
                config={"response_mime_type": "application/json", "response_schema": Agent1TechSchema}
            )

        try:
            tech_report = json.loads(agent1_response.text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Agent 1 (Technical) JSON output. Raw text: {agent1_response.text}", exc_info=True)
            typer.secho("\n❌ Error: AI processing failed. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
            raise typer.Exit(code=1)

        # Print Agent 1 Panel
        tech_bias = tech_report.get('bias', 'NEUTRAL')
        tech_color = get_bias_color(tech_bias)

        tech_summary = (
            f"Bias: [{tech_color} bold]{tech_bias}[/{tech_color} bold]\n\n"
            f"[bold]Trend State:[/bold]\n{tech_report.get('trend_state', '')}\n\n"
            f"[bold]Momentum Divergence:[/bold]\n{tech_report.get('momentum_divergence', '')}\n\n"
            f"[bold]Key Level Interaction:[/bold]\n{tech_report.get('key_level_interaction', '')}"
        )

        console.print(Panel(tech_summary, title="[Technical Analysis Agent]", border_style=tech_color, box=box.ROUNDED, expand=False))


        # --- Agent 2: Liquidity/Volume Analyst ---
        agent2_prompt = (
            "You are the Lead Volume Analyst. "
            "Liquidity State: Look strictly at price_to_va_status and the vah/val prices. Is the asset in price discovery or chopping inside fair value? "
            "Volume Momentum: Compare the latest volume against VMA_20. Is volume expanding or contracting? "
            "Magnet Target: Look strictly at the poc_price, vah, and val in the JSON. You MUST select the single most likely liquidity node price will be drawn to next. Do not invent prices.\n\n"
            "Return a strict JSON object with EXACTLY these keys: \n"
            "- bias (string: BULLISH, BEARISH, or NEUTRAL)\n"
            "- liquidity_state (string)\n"
            "- volume_momentum (string)\n"
            "- magnet_target (string)\n\n"
            f"--- MARKET DATA ---\n"
            f"4H Data: Price: ${data_4h.get('price', 0)}, Volume: {data_4h.get('volume', 0)} / VMA(20): {data_4h.get('vma_20', 0)}, "
            f"POC: ${data_4h.get('poc_price', 0)}, VAL: ${data_4h.get('val', 0)}, VAH: ${data_4h.get('vah', 0)}, "
            f"VA Status: {data_4h.get('va_status', 'UNKNOWN')}\n"
            f"15m Data: Price: ${data_15m.get('price', 0)}, Volume: {data_15m.get('volume', 0)} / VMA(20): {data_15m.get('vma_20', 0)}, "
            f"POC: ${data_15m.get('poc_price', 0)}, VAL: ${data_15m.get('val', 0)}, VAH: ${data_15m.get('vah', 0)}, "
            f"VA Status: {data_15m.get('va_status', 'UNKNOWN')}"
        )

        with console.status("[bold cyan]Agent 2 (Liquidity/Volume Analyst) Thinking... (Model: gemini-2.5-flash)[/bold cyan]", spinner="dots"):
            agent2_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=agent2_prompt,
                config={"response_mime_type": "application/json", "response_schema": Agent2VolumeSchema}
            )

        try:
            vol_report = json.loads(agent2_response.text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Agent 2 (Volume) JSON output. Raw text: {agent2_response.text}", exc_info=True)
            typer.secho("\n❌ Error: AI processing failed. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
            raise typer.Exit(code=1)

        # Print Agent 2 Panel
        vol_bias = vol_report.get('bias', 'NEUTRAL')
        vol_color = get_bias_color(vol_bias)

        vol_summary = (
            f"Bias: [{vol_color} bold]{vol_bias}[/{vol_color} bold]\n\n"
            f"[bold]Liquidity State:[/bold]\n{vol_report.get('liquidity_state', '')}\n\n"
            f"[bold]Volume Momentum:[/bold]\n{vol_report.get('volume_momentum', '')}\n\n"
            f"[bold]Magnet Target:[/bold]\n{vol_report.get('magnet_target', '')}"
        )

        console.print(Panel(vol_summary, title="[Volume & Liquidity Agent]", border_style=vol_color, box=box.ROUNDED, expand=False))

        # --- Agent 3: Lead Market Strategist ---
        agent3_prompt = (
            "You are the Lead Portfolio Manager. Synthesize Agent 1 and 2's reports. "
            "Technical Synthesis: State explicitly if the TA and Volume reports are in confluence or contradiction. "
            "Risk Profile: Evaluate Risk-to-Reward by comparing the distance to Agent 1's nearest structural threat vs Agent 2's magnet target. "
            "Confidence Score: Assign an integer from 0 to 100 representing the probability of a successful trade. High confluence and excellent R:R should score above 80. Contradictions must score below 50. "
            "Final Verdict: Select exactly one: GO LONG, GO SHORT, or SIT ON HANDS. If there is contradiction, poor R:R, or the confidence score is below 70, you MUST select SIT ON HANDS.\n\n"
            "Return a strict JSON object with EXACTLY these keys: \n"
            "- technical_synthesis (string)\n"
            "- risk_profile (string)\n"
            "- confidence_score (integer 0-100)\n"
            "- final_verdict (string: GO LONG, GO SHORT, or SIT ON HANDS)\n\n"
            f"--- SUB-AGENT REPORTS ---\n"
            f"Technical Agent Report:\n{json.dumps(tech_report, indent=2)}\n\n"
            f"Volume Agent Report:\n{json.dumps(vol_report, indent=2)}\n\n"
            f"--- CURRENT 15m MARKET DATA (For Context) ---\n"
            f"Price: ${data_15m.get('price', 0)}"
        )

        with console.status("[bold cyan]Agent 3 (Lead Market Strategist) Thinking... (Model: gemini-3.1-flash-lite-preview)[/bold cyan]", spinner="dots"):
            agent3_response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=agent3_prompt,
                config={"response_mime_type": "application/json", "response_schema": Agent3ManagerSchema}
            )

        try:
            strategist_report = json.loads(agent3_response.text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Agent 3 (Strategist) JSON output. Raw text: {agent3_response.text}", exc_info=True)
            typer.secho("\n❌ Error: AI processing failed. Check error.log for details.\n", fg=typer.colors.RED, bold=True)
            raise typer.Exit(code=1)

        # Print the Lead Market Strategist Panel cleanly
        final_verdict = strategist_report.get('final_verdict', 'SIT ON HANDS')
        strategist_summary = (
            f"[bold]Final Verdict:[/bold] {final_verdict}\n"
            f"[bold]Confidence Score:[/bold] {strategist_report.get('confidence_score', 0)}\n\n"
            f"[bold]Technical Synthesis:[/bold]\n{strategist_report.get('technical_synthesis', '')}\n\n"
            f"[bold]Risk Profile:[/bold]\n{strategist_report.get('risk_profile', '')}"
        )

        console.print(Panel(strategist_summary, title="[Lead Market Strategist - Synthesis]", border_style="magenta", box=box.ROUNDED, expand=False))

        filepath = log_execution("analyze", symbol, data_4h, data_15m, tech_report, vol_report, strategist_report)
        console.print(f"[dim]💾 Footprint saved to: {filepath}[/dim]")

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


@app.command("status")
def status_command(symbol: str = typer.Argument("BTC/USDT")):
    """
    Fetch MTF (4h, 15m) data for the given symbol and print the raw metrics.
    No AI analysis is executed.
    """
    _run_status(symbol)

@app.command("analyze")
def analyze_command(symbol: str = typer.Argument("BTC/USDT")):
    """
    Fetch MTF (4h, 15m) data and execute trading analysis via AI agents.
    Outputs the Lead Market Strategist thesis.
    """
    _run_analyze(symbol)

@app.command("operate")
def operate_command(symbol: str = typer.Argument("BTC/USDT")):
    """
    Execute trading operations based on recent analysis.
    """
    _run_operate(symbol)

@app.command("auto")
def auto_command(symbol: str = typer.Argument("BTC/USDT")):
    """
    Execute the entire sequential pipeline: Status -> Analyze -> Operate.
    """
    _run_status(symbol)
    _run_analyze(symbol)
    _run_operate(symbol)

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

        # Define the specific Gemma model we are using
        model_id = "gemma-3-12b-it"

        typer.secho(f"Thinking... (Model: {model_id})", fg=typer.colors.YELLOW)

        # Call the Google GenAI model to generate content
        response = client.models.generate_content(
            model=model_id,
            contents=question
        )

        # Print the response cleanly to the console
        typer.secho("\n🤖 AI Response:", fg=typer.colors.MAGENTA, bold=True)
        typer.secho("-" * 40, fg=typer.colors.MAGENTA)
        typer.echo(response.text)
        typer.secho("-" * 40 + "\n", fg=typer.colors.MAGENTA)

    except genai.errors.APIError as e:
        # Handle errors directly from the Gemini API
        typer.secho(f"API Error: Failed to generate content. Please check your API key and connection. Details: {e}", fg=typer.colors.RED)
    except Exception as e:
        # Catch any other unexpected errors
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)


if __name__ == "__main__":
    # Run the Typer application
    app()
