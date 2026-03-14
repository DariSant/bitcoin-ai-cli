import os
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

    # Check for NaNs on required indicators in the last row
    last_row = df.iloc[-1]

    if pd.isna(last_row['EMA_144']) or pd.isna(last_row['RSI_13']) or pd.isna(last_row['RSI_47']):
        raise ValueError(f"Insufficient candle data fetched to calculate the 144 EMA or RSIs for {timeframe} timeframe.")

    # Return rounded values for the latest row
    return {
        'price': round(float(last_row['close']), 2),
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
    typer.echo(f"EMAs (34,89,144) : {data_4h['ema_34']}, {data_4h['ema_89']}, {data_4h['ema_144']}")
    typer.echo(f"RSI (13,47)      : {data_4h['rsi_13']}, {data_4h['rsi_47']}")
    typer.echo(f"RSI Delta        : {data_4h['rsi_delta']}")
    typer.secho("-" * 60, fg=typer.colors.CYAN)

    # 15M Print
    typer.secho("📉 15m Micro Trend Metrics", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"Current Price : ${data_15m['price']:,.2f}")
    typer.echo(f"EMAs (34,89,144) : {data_15m['ema_34']}, {data_15m['ema_89']}, {data_15m['ema_144']}")
    typer.echo(f"RSI (13,47)      : {data_15m['rsi_13']}, {data_15m['rsi_47']}")
    typer.echo(f"RSI Delta        : {data_15m['rsi_delta']}")
    typer.secho("=" * 60 + "\n", fg=typer.colors.CYAN)

    try:
        # Initialize the Google GenAI client
        client = genai.Client(api_key=api_key)
        model_id = "gemini-2.5-flash"

        prompt = (
            f"You are an expert crypto market analyst. Analyze this multi-timeframe data for {symbol}.\n"
            f"4H MACRO TREND: Price: ${data_4h['price']}, EMA(34,89,144): [{data_4h['ema_34']}, {data_4h['ema_89']}, {data_4h['ema_144']}], "
            f"RSI(13,47): [{data_4h['rsi_13']}, {data_4h['rsi_47']}], RSI Delta: {data_4h['rsi_delta']}.\n"
            f"15M MICRO TREND: Price: ${data_15m['price']}, EMA(34,89,144): [{data_15m['ema_34']}, {data_15m['ema_89']}, {data_15m['ema_144']}], "
            f"RSI(13,47): [{data_15m['rsi_13']}, {data_15m['rsi_47']}], RSI Delta: {data_15m['rsi_delta']}.\n"
            "Based strictly on these Fibonacci EMAs and Dual-RSI momentum metrics, provide a professional, highly concise 3-sentence analysis: "
            "1 sentence on the macro 4H trend, 1 sentence on the micro 15m momentum, and 1 concluding sentence with an actionable bias "
            "(e.g., bullish continuation, bearish divergence, no clear trend). At the end of the inference include a trading set up (long/short/no action) "
            "with a confidence score (1-100%)."
        )

        typer.secho(f"Thinking... (Model: {model_id})", fg=typer.colors.YELLOW)

        # Call the Google GenAI model to generate content
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )

        # Print the response cleanly to the console
        typer.secho("\n🤖 AI Market Analysis:", fg=typer.colors.MAGENTA, bold=True)
        typer.secho("-" * 60, fg=typer.colors.MAGENTA)
        typer.echo(response.text)
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
