import os
import json
import typer
import ccxt
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from google import genai
import typing

# Load environment variables from the .env file
load_dotenv()

# Create the Typer app instance
app = typer.Typer(help="Bitcoin AI CLI Tool")

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

    if pd.isna(last_row['EMA_144']) or pd.isna(last_row['RSI_13']) or pd.isna(last_row['RSI_47']) or pd.isna(last_row['VMA_20']) or pd.isna(last_row['ATR_14']):
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

@app.command()
def analyze():
    """
    Fetch MTF (4h, 15m) data for BTC/USDT, generate indicators, and get an AI analysis.
    """
    # Ensure the Gemini API key is loaded securely
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        typer.secho("Error: GEMINI_API_KEY environment variable is missing. Please set it in your .env file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Initialize a CCXT exchange instance for Binance
        exchange = ccxt.binance()
        symbol = 'BTC/USDT'

        # Fetch and analyze data for both timeframes
        data_4h = fetch_and_analyze(exchange, symbol, '4h')
        data_15m = fetch_and_analyze(exchange, symbol, '15m')

    except (RuntimeError, ValueError, Exception) as e:
        typer.secho(f"\n❌ Error: Could not calculate metrics - {e}\n", fg=typer.colors.RED, bold=True)
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

    try:
        # Initialize the Google GenAI client
        client = genai.Client(api_key=api_key)

        # Define schemas for structured JSON output
        class Agent1Schema(typing.TypedDict):
            bias: str
            analysis: str
            confidence_score: int
            timeframe_alignment: bool
            nearest_support: float
            nearest_resistance: float
            distance_to_144_ema_percent: float
            current_atr_14: float

        class Agent2Schema(typing.TypedDict):
            bias: str
            analysis: str
            confidence_score: int
            timeframe_alignment: bool
            price_to_value_area_status: str
            distance_to_poc_percent: float
            volume_vs_20ema: str

        class Agent3Schema(typing.TypedDict):
            symbol: str
            reasoning: str
            operative: str
            order_type: str
            system_confidence_score: int
            risk_allocation_percent: float
            entry_price: float
            atr_multiplier: float
            risk_reward_ratio: float
            ttl_hours: int


        # --- Agent 1: Technical Analyst ---
        typer.secho(f"Agent 1 (Technical Analyst) Thinking... (Model: gemini-2.5-flash)", fg=typer.colors.YELLOW)
        agent1_prompt = (
            "You are the Technical Analysis Agent for an algorithmic trading system. "
            "Your objective is to analyze the 15m and 4H timeframes using EMAs, RSI momentum, and Volatility (ATR). Do not consider volume. "
            "Your specific duties: \n"
            "1. Determine the trend bias strictly based on price action and EMA alignment.\n"
            "2. Identify the nearest major support and resistance levels based on the EMAs.\n"
            "3. Evaluate timeframe alignment (Does the 15m trend agree with the 4H?).\n"
            "4. Assess mean reversion risk using the distance to the 144 EMA.\n\n"
            "Return a strict JSON object with EXACTLY these keys: \n"
            "- bias (string: STRONGLY_BULLISH, BULLISH, NEUTRAL, BEARISH, or STRONGLY_BEARISH)\n"
            "- analysis (string: max 3 sentences summarizing momentum, EMAs, and mean reversion risk)\n"
            "- confidence_score (integer: 0-100)\n"
            "- timeframe_alignment (boolean: true if 15m and 4H align, false otherwise)\n"
            "- nearest_support (float: price level)\n"
            "- nearest_resistance (float: price level)\n"
            "- distance_to_144_ema_percent (float)\n"
            "- current_atr_14 (float)\n\n"
            f"--- MARKET DATA ---\n"
            f"4H Data: Price: ${data_4h.get('price', 0)}, EMA(34,89,144): [{data_4h.get('ema_34', 0)}, {data_4h.get('ema_89', 0)}, {data_4h.get('ema_144', 0)}], "
            f"RSI(13,47): [{data_4h.get('rsi_13', 0)}, {data_4h.get('rsi_47', 0)}], RSI Delta: {data_4h.get('rsi_delta', 0)}, "
            f"ATR(14): {data_4h.get('atr_14', 0)}, Dist to 144 EMA: {data_4h.get('dist_144_percent', 0)}%\n"
            f"15m Data: Price: ${data_15m.get('price', 0)}, EMA(34,89,144): [{data_15m.get('ema_34', 0)}, {data_15m.get('ema_89', 0)}, {data_15m.get('ema_144', 0)}], "
            f"RSI(13,47): [{data_15m.get('rsi_13', 0)}, {data_15m.get('rsi_47', 0)}], RSI Delta: {data_15m.get('rsi_delta', 0)}, "
            f"ATR(14): {data_15m.get('atr_14', 0)}, Dist to 144 EMA: {data_15m.get('dist_144_percent', 0)}%"
        )

        agent1_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=agent1_prompt,
            config={"response_mime_type": "application/json", "response_schema": Agent1Schema}
        )

        try:
            tech_report = json.loads(agent1_response.text)
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse Agent 1 (Technical) JSON output.")


        # --- Agent 2: Liquidity/Volume Analyst ---
        typer.secho(f"Agent 2 (Liquidity/Volume Analyst) Thinking... (Model: gemini-2.5-flash)", fg=typer.colors.YELLOW)
        agent2_prompt = (
            "You are the Volume Analysis Agent for an algorithmic trading system. "
            "Your objective is to analyze the 15m and 4H timeframes using Total Volume, Volume Moving Average (VMA), and Volume Profile (POC, VAH, VAL). "
            "Your specific duties: \n"
            "1. Determine the volume bias based on momentum and Value Area positioning.\n"
            "2. Analyze price interaction with the Value Area (e.g., inside value, rejecting boundaries, breaking out).\n"
            "3. Evaluate timeframe alignment (Does the 15m volume profile agree with the 4H?).\n"
            "4. Assess mean reversion risk using the percentage distance to the Point of Control (POC).\n\n"
            "Return a strict JSON object with EXACTLY these keys: \n"
            "- bias (string: STRONGLY_BULLISH, BULLISH, NEUTRAL, BEARISH, or STRONGLY_BEARISH)\n"
            "- analysis (string: max 3 sentences summarizing volume strength, POC interaction, and breakout/mean-reversion conditions)\n"
            "- confidence_score (integer: 0-100)\n"
            "- timeframe_alignment (boolean: true if 15m and 4H align, false otherwise)\n"
            "- price_to_value_area_status (string: INSIDE_VALUE, BREAKING_ABOVE_VAH, BREAKING_BELOW_VAL, REJECTING_VAH, or REJECTING_VAL)\n"
            "- distance_to_poc_percent (float)\n"
            "- volume_vs_20ema (string: ABOVE_AVERAGE or BELOW_AVERAGE)\n\n"
            f"--- MARKET DATA ---\n"
            f"4H Data: Price: ${data_4h.get('price', 0)}, Volume: {data_4h.get('volume', 0)} / VMA(20): {data_4h.get('vma_20', 0)}, "
            f"POC: ${data_4h.get('poc_price', 0)}, VAL-VAH: ${data_4h.get('val', 0)} - ${data_4h.get('vah', 0)}, "
            f"VA Status: {data_4h.get('va_status', 'UNKNOWN')}, Dist to POC: {data_4h.get('dist_poc_percent', 0)}%\n"
            f"15m Data: Price: ${data_15m.get('price', 0)}, Volume: {data_15m.get('volume', 0)} / VMA(20): {data_15m.get('vma_20', 0)}, "
            f"POC: ${data_15m.get('poc_price', 0)}, VAL-VAH: ${data_15m.get('val', 0)} - ${data_15m.get('vah', 0)}, "
            f"VA Status: {data_15m.get('va_status', 'UNKNOWN')}, Dist to POC: {data_15m.get('dist_poc_percent', 0)}%"
        )

        agent2_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=agent2_prompt,
            config={"response_mime_type": "application/json", "response_schema": Agent2Schema}
        )

        try:
            vol_report = json.loads(agent2_response.text)
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse Agent 2 (Volume) JSON output.")


        # --- Agent 3: Portfolio Manager (The Synthesizer) ---
        typer.secho(f"Agent 3 (Portfolio Manager) Thinking... (Model: gemini-3.1-flash-lite-preview)", fg=typer.colors.YELLOW)

        agent3_prompt = (
            "You are the Lead Portfolio Manager for an algorithmic trading system. "
            f"Your objective is to synthesize the structured JSON reports from the Technical Agent and Volume Agent for {symbol}. "
            "Your specific duties and strict constraints: \n"
            "1. Synthesize the bias, timeframe alignment, and confidence scores from both sub-agents.\n"
            "2. Determine the 'operative' (ENTER LONG, ENTER SHORT, or SIT ON HANDS). If the agents heavily conflict, timeframe alignment is false across the board, or volume is dead, you MUST output 'SIT ON HANDS'.\n"
            "3. Determine the 'order_type' (MARKET, LIMIT, or NONE). Use MARKET for breakouts, LIMIT for mean-reversion pullbacks.\n"
            "4. CRITICAL MATH RULE: Do NOT attempt to calculate exact price levels for Stop Loss or Take Profit. You must only output an 'atr_multiplier' (e.g., 1.5, 2.0) for the Stop Loss distance, and a 'risk_reward_ratio' (e.g., 2.0, 3.0) for the Take Profit distance.\n"
            "5. SIT ON HANDS RULE: If your operative is 'SIT ON HANDS', you must set 'entry_price', 'risk_allocation_percent', 'atr_multiplier', 'risk_reward_ratio', and 'ttl_hours' to exactly 0. Set 'order_type' to 'NONE'.\n\n"
            "Return a strict JSON object with EXACTLY these keys: \n"
            "- symbol (string)\n"
            "- reasoning (string: max 3 sentences explaining confluence/conflict and your decision)\n"
            "- operative (string: ENTER LONG, ENTER SHORT, SIT ON HANDS)\n"
            "- order_type (string: MARKET, LIMIT, NONE)\n"
            "- system_confidence_score (integer: 0-100)\n"
            "- risk_allocation_percent (float: 0.0 to 2.0)\n"
            "- entry_price (float: current price for market, pullback target for limit. 0 if SIT ON HANDS)\n"
            "- atr_multiplier (float: e.g., 1.5. 0 if SIT ON HANDS)\n"
            "- risk_reward_ratio (float: e.g., 2.0. 0 if SIT ON HANDS)\n"
            "- ttl_hours (integer: e.g., 12, 24, 48. 0 if SIT ON HANDS)\n\n"
            f"--- SUB-AGENT REPORTS ---\n"
            f"Technical Agent Report:\n{json.dumps(tech_report, indent=2)}\n\n"
            f"Volume Agent Report:\n{json.dumps(vol_report, indent=2)}\n\n"
            f"--- CURRENT 15m MARKET DATA (For Entry Context) ---\n"
            f"Price: ${data_15m.get('price', 0)}, ATR(14): ${data_15m.get('atr_14', 0)}, POC: ${data_15m.get('poc_price', 0)}"
        )

        agent3_response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=agent3_prompt,
            config={"response_mime_type": "application/json", "response_schema": Agent3Schema}
        )

        try:
            final_directive = json.loads(agent3_response.text)
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse Agent 3 (Portfolio Manager) JSON output.")

        # --- Handle "SIT ON HANDS" Edge Case ---
        operative = final_directive.get("operative", "SIT ON HANDS")
        if operative == "SIT ON HANDS":
            typer.secho("\n⚠️ ACTION: SIT ON HANDS", fg=typer.colors.YELLOW, bold=True)
            typer.echo(f"📝 REASONING: {final_directive.get('reasoning', 'No reasoning provided.')}")
            typer.echo("🛑 Skipping execution math. Awaiting next candle.\n")
            raise typer.Exit()

        # --- Stop Loss and Take Profit Math ---
        take_profit = 0.0
        stop_loss = 0.0

        entry = final_directive['entry_price']
        atr = data_15m.get('atr_14', 0)
        stop_distance = atr * final_directive['atr_multiplier']
        profit_distance = stop_distance * final_directive['risk_reward_ratio']

        if final_directive['operative'] == "ENTER LONG":
            stop_loss = entry - stop_distance
            take_profit = entry + profit_distance
        elif final_directive['operative'] == "ENTER SHORT":
            stop_loss = entry + stop_distance
            take_profit = entry - profit_distance

        # Append calculated values to the payload
        final_directive['calculated_tp'] = round(take_profit, 2)
        final_directive['calculated_sl'] = round(stop_loss, 2)

        # Print the Portfolio Manager's final calculated payload cleanly to the console
        typer.secho("\n🤖 Portfolio Manager Directive:", fg=typer.colors.MAGENTA, bold=True)
        typer.secho("-" * 60, fg=typer.colors.MAGENTA)
        typer.echo(f"Reasoning: {final_directive['reasoning']}")
        typer.echo(f"Operative: {final_directive['operative']} ({final_directive['order_type']}) @ ${final_directive['entry_price']}")
        typer.echo(f"Risk Allocation: {final_directive['risk_allocation_percent']}% | System Confidence: {final_directive['system_confidence_score']}%")
        typer.echo(f"Target (TP): ${final_directive['calculated_tp']} | Invalidation (SL): ${final_directive['calculated_sl']}")
        typer.echo(f"TTL: {final_directive['ttl_hours']} Hours")
        typer.secho("-" * 60 + "\n", fg=typer.colors.MAGENTA)

    except typer.Exit:
        # Re-raise Typer's Exit exception so the CLI can exit gracefully (e.g., during "SIT ON HANDS")
        raise
    except genai.errors.APIError as e:
        typer.secho(f"API Error: Failed to generate content. Please check your API key and connection. Details: {e}", fg=typer.colors.RED)
    except Exception as e:
        typer.secho(f"An unexpected error occurred during AI analysis: {e}", fg=typer.colors.RED)


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
