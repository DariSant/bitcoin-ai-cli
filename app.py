import os
import json
import typer
import ccxt
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from google import genai

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

    # Calculate Volume Profile Point of Control (POC)
    # Create 10 equal price bins based on the close column
    bins = pd.cut(df['close'], bins=10)
    # Group by bins and sum the volume for each bin
    volume_by_bin = df.groupby(bins, observed=False)['volume'].sum()
    # Find the bin with the maximum volume
    max_volume_bin = volume_by_bin.idxmax()
    # Extract the midpoint price of that highest-volume bin
    poc_price = float(max_volume_bin.mid)

    # Check for NaNs on required indicators in the last row
    last_row = df.iloc[-1]

    if pd.isna(last_row['EMA_144']) or pd.isna(last_row['RSI_13']) or pd.isna(last_row['RSI_47']) or pd.isna(last_row['VMA_20']):
        raise ValueError(f"Insufficient candle data fetched to calculate required technical and volume metrics for {timeframe} timeframe.")

    # Return rounded values for the latest row
    return {
        'price': round(float(last_row['close']), 2),
        'volume': round(float(last_row['volume']), 2),
        'vma_20': round(float(last_row['VMA_20']), 2),
        'poc_price': round(poc_price, 2),
        'ema_34': round(float(last_row['EMA_34']), 2),
        'ema_89': round(float(last_row['EMA_89']), 2),
        'ema_144': round(float(last_row['EMA_144']), 2),
        'rsi_13': round(float(last_row['RSI_13']), 2),
        'rsi_47': round(float(last_row['RSI_47']), 2),
        'rsi_delta': round(float(last_row['RSI_Delta']), 2),
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
    typer.secho("-" * 60, fg=typer.colors.CYAN)

    # 15M Print
    typer.secho("📉 15m Micro Trend Metrics", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"Current Price : ${data_15m['price']:,.2f}")
    typer.echo(f"Volume / VMA (20) : {data_15m['volume']} / {data_15m['vma_20']}")
    typer.echo(f"Volume Profile POC: ${data_15m['poc_price']}")
    typer.echo(f"EMAs (34,89,144) : {data_15m['ema_34']}, {data_15m['ema_89']}, {data_15m['ema_144']}")
    typer.echo(f"RSI (13,47)      : {data_15m['rsi_13']}, {data_15m['rsi_47']}")
    typer.echo(f"RSI Delta        : {data_15m['rsi_delta']}")
    typer.secho("=" * 60 + "\n", fg=typer.colors.CYAN)

    try:
        # Initialize the Google GenAI client
        client = genai.Client(api_key=api_key)
        model_id = "gemini-2.5-flash"

        # Set up the configuration for the first two agents to force structured output
        json_config = {"response_mime_type": "application/json"}



        # --- Agent 1: Technical Analyst ---
        typer.secho(f"Agent 1 (Technical Analyst) Thinking... (Model: {model_id})", fg=typer.colors.YELLOW)
        agent1_prompt = (
            "You are a pure Technical Analyst. Analyze these EMAs and RSI momentum metrics. Do not consider volume. "
            "Return a strict JSON object with three keys: bias (string: Bullish, Bearish, Neutral), "
            "analysis (string: a highly dense, 2-sentence summary of the momentum/trend), "
            "confidence_score (integer: 1-100)\n\n"
            f"4H Data: Price: ${data_4h['price']}, EMA(34,89,144): [{data_4h['ema_34']}, {data_4h['ema_89']}, {data_4h['ema_144']}], "
            f"RSI(13,47): [{data_4h['rsi_13']}, {data_4h['rsi_47']}], RSI Delta: {data_4h['rsi_delta']}\n"
            f"15m Data: Price: ${data_15m['price']}, EMA(34,89,144): [{data_15m['ema_34']}, {data_15m['ema_89']}, {data_15m['ema_144']}], "
            f"RSI(13,47): [{data_15m['rsi_13']}, {data_15m['rsi_47']}], RSI Delta: {data_15m['rsi_delta']}"
        )

        agent1_response = client.models.generate_content(
            model=model_id,
            contents=agent1_prompt,
            config=json_config
        )

        try:
            agent1_report = json.loads(agent1_response.text)
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse Agent 1 (Technical) JSON output.")


        # --- Agent 2: Liquidity/Volume Analyst ---
        typer.secho(f"Agent 2 (Liquidity/Volume Analyst) Thinking... (Model: {model_id})", fg=typer.colors.YELLOW)
        agent2_prompt = (
            "You are a Liquidity and Volume Analyst. Analyze Volume vs VMA, and where Price sits relative to the Volume Profile POC. "
            "Return a strict JSON object with three keys: bias (string: Bullish, Bearish, Neutral), "
            "analysis (string: a highly dense, 2-sentence summary of the liquidity/support/resistance), "
            "confidence_score (integer 1-100)\n\n"
            f"4H Data: Price: ${data_4h['price']}, Volume: {data_4h['volume']}, VMA(20): {data_4h['vma_20']}, POC Price: ${data_4h['poc_price']}\n"
            f"15m Data: Price: ${data_15m['price']}, Volume: {data_15m['volume']}, VMA(20): {data_15m['vma_20']}, POC Price: ${data_15m['poc_price']}"
        )

        agent2_response = client.models.generate_content(
            model=model_id,
            contents=agent2_prompt,
            config=json_config
        )

        try:
            agent2_report = json.loads(agent2_response.text)
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse Agent 2 (Volume) JSON output.")


        # --- Agent 3: Portfolio Manager (The Synthesizer) ---
        typer.secho(f"Agent 3 (Portfolio Manager) Thinking... (Model: {model_id})", fg=typer.colors.YELLOW)

        # Combine the parsed JSON dictionaries into a single payload
        payload = json.dumps({
            "technical_agent": agent1_report,
            "volume_agent": agent2_report
        }, indent=2)

        agent3_prompt = (
            "You are the Lead Portfolio Manager. Review the JSON reports from the Technical and Liquidity agents. "
            "You must use step-by-step reasoning to synthesize their findings, looking for confluence or divergence. "
            "Format your final response EXACTLY as follows:\n\n"
            "Reasoning: [2-3 sentences explaining your logic based on the agent reports. Look for confluence or conflicts.]\n"
            "Operative: [ENTER LONG, ENTER SHORT, or SIT ON HANDS]\n"
            "Confidence: [0-100%]\n"
            "Invalidation (Stop Loss): [Specific price level where this thesis is proven wrong, usually based on the POC or a key EMA.]\n\n"
            f"Agent Reports:\n{payload}"
        )

        agent3_response = client.models.generate_content(
            model=model_id,
            contents=agent3_prompt
        )

        # Print the Portfolio Manager's final response cleanly to the console
        typer.secho("\n🤖 Portfolio Manager Directive:", fg=typer.colors.MAGENTA, bold=True)
        typer.secho("-" * 60, fg=typer.colors.MAGENTA)
        typer.echo(agent3_response.text.strip())
        typer.secho("-" * 60 + "\n", fg=typer.colors.MAGENTA)

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
