import os
import typer
import ccxt
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()

# Create the Typer app instance
app = typer.Typer(help="Bitcoin AI CLI Tool")

@app.command()
def analyze():
    """
    Fetch and display the current ticker data for BTC/USDT from Binance.
    """
    try:
        # Initialize a CCXT exchange instance for Binance
        exchange = ccxt.binance()

        # Fetch the current ticker data for BTC/USDT
        ticker = exchange.fetch_ticker('BTC/USDT')

        # Extract the relevant data: price, 24h volume, high, low
        price = ticker.get('last')
        volume = ticker.get('baseVolume')
        high = ticker.get('high')
        low = ticker.get('low')

        # Print the raw data cleanly to the console
        typer.secho("\n📊 BTC/USDT Market Data (Binance)", fg=typer.colors.CYAN, bold=True)
        typer.secho("=" * 40, fg=typer.colors.CYAN)
        typer.echo(f"Current Price : ${price:,.2f}" if price else "Current Price : N/A")
        typer.echo(f"24h High      : ${high:,.2f}" if high else "24h High      : N/A")
        typer.echo(f"24h Low       : ${low:,.2f}" if low else "24h Low       : N/A")
        typer.echo(f"24h Volume    : {volume:,.2f} BTC" if volume else "24h Volume    : N/A")
        typer.secho("=" * 40 + "\n", fg=typer.colors.CYAN)

    except ccxt.NetworkError as e:
        # Handle connection and network timeouts
        typer.secho("Network Error: Could not connect to Binance. Please check your internet connection.", fg=typer.colors.RED)
    except ccxt.ExchangeError as e:
        # Handle API-specific errors
        typer.secho(f"Exchange Error: Could not fetch data from Binance ({e}).", fg=typer.colors.RED)
    except Exception as e:
        # Catch any other unexpected errors
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)


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
