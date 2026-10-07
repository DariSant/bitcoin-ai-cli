"""Entry point: `uv run app.py <command>`. The code lives in the btc_cli/ package."""

from dotenv import load_dotenv

from btc_cli.logging_setup import configure_logging

# Load environment variables from the .env file, then set up logging, before any command runs.
load_dotenv()
configure_logging()

from btc_cli.cli import app  # noqa: E402

if __name__ == "__main__":
    # Run the Typer application
    app()
