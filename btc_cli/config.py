"""Settings. These constants move to config.toml with the Phase 3 config item."""

# Output root for recorded data. The analyze pipeline rebinds this to "output_beta"
# for the rest of the process once any agent uses the fallback model (known P0 bug,
# TODO.md "Two ledgers per strategy"). Always read it as `config.BASE_DIR` at call
# time, never `from btc_cli.config import BASE_DIR`, or that rebinding is missed.
BASE_DIR = "output_alpha"

# AI models used by every agent.
# PRIMARY_MODEL is tried first. If it fails, we instantly switch to FALLBACK_MODEL.
# To change models later, edit only these two lines.
PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash"
