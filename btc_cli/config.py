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

# Market data source. `data.create_exchange()` builds the exchange from EXCHANGE_ID,
# so the source written into every record is the one actually used. MARKET_TYPE must
# match that exchange class (ccxt "binance" = spot). Changing either is a §2.4 change.
EXCHANGE_ID = "binance"
MARKET_TYPE = "spot"

# Record versions (AGENTS.md §6), written into every record.
# SCHEMA_VERSION: bump when a record's structure changes. Records without it are schema 0.
# STRATEGY_VERSION: bump for any §2.4 change (owner decision 2026-10-07). "0.x" versions
# are warm-up data; "1.0" is the first official version, once Phase 1 is complete.
# Records without it are legacy "0.0".
SCHEMA_VERSION = 1
STRATEGY_VERSION = "0.1"
