# Bitcoin AI CLI Tool

A free command-line tool that pulls live Bitcoin market data from Binance, runs technical analysis on it, and asks a team of AI agents (Google Gemini) whether to **GO LONG**, **GO SHORT**, or **SIT ON HANDS**. If the AI says to trade, plain Python math (not the AI) works out the entry, stop loss, take profit and position size, and records a **paper trade**. No real money is ever used.

> ⚠️ **Not financial advice.** This is a learning and research project. It does not place real orders on any exchange.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Commands](#commands)
6. [The Two Strategies](#the-two-strategies)
7. [Where Your Data Is Saved](#where-your-data-is-saved)
8. [Testing](#testing)
9. [Project Files](#project-files)
10. [Roadmap](#roadmap)

---

## How It Works

Each run moves through four stages:

```
 Binance (ccxt)            Gemini AI agents                         Pure Python
┌──────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────────┐
│ 4H + 15m     │──▶│ Agent 1: Technical Analyst    │──▶│ Agent 4: The Operator         │
│ candles      │   │ Agent 2: Volume/Liquidity     │   │ entry, stop loss, take profit,│
│ + indicators │   │ Agent 3: Lead Manager         │   │ risk:reward, position size    │
└──────────────┘   │  (Defensive and/or Greedy)    │   └───────────────────────────────┘
                   └───────────────────────────────┘                  │
                                                                      ▼
                                                     Paper trade ledger + trade history
```

1. **Data Engine** – Downloads the last 200 candles for the 4-hour (big picture) and 15-minute (short term) timeframes and calculates indicators.
2. **Sub-Agents** – Agent 1 reads the trend and momentum. Agent 2 reads volume and liquidity, and picks a "magnet" price target.
3. **Lead Manager** – Agent 3 combines both reports and gives a final verdict: `GO LONG`, `GO SHORT`, or `SIT ON HANDS`.
4. **The Operator** – If the verdict is a trade, Python calculates the order. The AI never does the arithmetic, so it can't get the numbers wrong.

---

## Features

### Data Engine
- Live **BTC/USDT** data from Binance via `ccxt` (any Binance symbol can be passed in).
- Two timeframes: **4H** (macro trend) and **15m** (micro trend).
- **EMAs** 34, 89, 144 – trend direction.
- **RSI** 13 and 47, plus **RSI Delta** (13 minus 47) – momentum shifts.
- **Volume MA (20)** – is volume above or below normal?
- **ATR (14)** – how much the price typically moves (volatility).
- **Volume Profile** – Point of Control (POC, the price with the most volume) and the **Value Area** (VAL–VAH, where 70% of volume traded). Shows whether price is inside value, breaking above VAH, or breaking below VAL.
- **Mean reversion** – how far (in %) price is from the 144 EMA and from the POC.
- **Swing support / resistance** – lowest low and highest high of the last 20 candles.

### AI Agents
- All agents must reply in **strict JSON** (enforced with Python `TypedDict` schemas through Gemini's `response_schema`), so the AI can't wander off into chat.
- Each agent gives a bias: `STRONGLY_BULLISH`, `BULLISH`, `NEUTRAL`, `BEARISH`, or `STRONGLY_BEARISH`.
- **Model fallback:** every request goes to `gemini-3.5-flash-lite` first. If that model is down, it switches instantly to `gemini-2.5-flash`, logs the incident to `logs/system_health.log`, and saves that run's data to `output_beta/` instead of `output_alpha/` so backup-model results don't mix with the main results.

### The Operator (Python risk math)
- Order type: **MARKET**, entering at the current 15m price.
- **Stop loss:** swing support/resistance ± 0.5 × ATR, with a **volatility floor**. The stop is always at least 1 full ATR away from entry.
- **Take profit:** Agent 2's magnet target. If it can't be read, the POC is used.
- **Fixed risk:** $100 per trade (1% of a $10,000 paper account). Position size is calculated from that.
- **Safety check:** if the stop loss or take profit is on the wrong side of the price, the ticket is rejected and written to `output_alpha/operator_errors.log`.
- Analysis older than **10 minutes** is ignored, so trades are never based on stale data.

### Paper Trading
- Each strategy can have **one open position per symbol** at a time.
- Before every new run, the tool checks recent 15m candles to see if the open trade hit its take profit (**WIN**) or stop loss (**LOSS**), calculates the profit or loss, and moves it to the trade history.
- New analysis is paused while a trade is still open.
- Records are written crash-safely: a file is either fully replaced or left as it was, and a closed trade is never added to the history twice.
- If a ledger or trade history file can't be read, it is **left exactly as it is** (never overwritten or deleted), the error goes to `error.log`, and that strategy stops opening trades until the file is repaired by hand. The other strategy keeps running, and the command exits with code 1 so a scheduler notices.

### Terminal Experience
- Colour-coded panels and loading spinners (via `rich`).
- Friendly error messages. Full technical details go to `error.log` instead of the screen.

---

## Requirements

- **Python 3.13** or newer
- **[uv](https://docs.astral.sh/uv/)** – manages Python and the project's libraries
- A free **Google Gemini API key** – get one at [Google AI Studio](https://aistudio.google.com/app/apikey)
- An internet connection (Binance public market data needs no account or key)

### Libraries Used

| Library | Why it's used |
|---|---|
| `ccxt` | Connects to Binance (and 100+ other exchanges) with one simple interface |
| `pandas` | Stores candle data in tables that are easy to calculate on |
| `pandas-ta` | Ready-made indicators (EMA, RSI, ATR, SMA), so we don't write the formulas by hand |
| `google-genai` | Official Google library for talking to Gemini |
| `python-dotenv` | Reads your API key from a `.env` file so it never lives in the code |
| `typer` | Turns Python functions into terminal commands with `--help` built in |
| `rich` | Coloured panels, tables, and spinners in the terminal |
| `requests` | Simple web requests |

---

## Installation

**1. Install uv** (skip if you already have it). In PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal after it finishes.

**2. Go to the project folder:**

```powershell
cd C:\Users\dsant\Desktop\Bitcoin_CLI_Repo
```

**3. Install all libraries** (uv reads them from `pyproject.toml`):

```powershell
uv sync
```

**4. Add your Gemini API key.** Create a file named `.env` in the project folder containing this one line (swap in your real key):

```
GEMINI_API_KEY=paste_your_key_here
```

`.env` is listed in `.gitignore`, so your key will never be uploaded to GitHub.

**5. Check it works:**

```powershell
uv run app.py commands
```

To add a new library later, always use `uv add <library>` (for example `uv add numpy`).

---

## Commands

Every command starts with `uv run app.py`. `SYMBOL` is optional and defaults to `BTC/USDT`.

| Command | What it does | Options |
|---|---|---|
| `status` | Fetches data and prints all indicators. **No AI is used.** | `[SYMBOL]` |
| `analyze` | Fetches data and runs Agents 1, 2 and 3. Saves the result. | `[SYMBOL]` `--def` `--greed` |
| `operate` | Reads the latest analysis (under 10 min old) and creates a paper trade if the verdict is LONG/SHORT. | `[SYMBOL]` `--def` `--greed` |
| `auto` | Runs the full pipeline: `status` → `analyze` → `operate`. | `[SYMBOL]` `--def` `--greed` |
| `mock` | Sends a test file from `mock_json/` straight to the Operator to check the math. **No AI or internet needed.** | `FILENAME` |
| `ask` | Asks Gemini any free-form question. | `"QUESTION"` |
| `commands` | Shows a table of all commands. | none |

`--def` runs only the Defensive strategy and `--greed` runs only the Greedy one. Leave both off to run both. Passing both at once gives an error.

### Examples

```powershell
# See today's raw market metrics (no AI)
uv run app.py status

# Run the full pipeline for both strategies
uv run app.py auto

# Run the full pipeline for the Defensive strategy only
uv run app.py auto --def

# Analyze Ethereum instead of Bitcoin
uv run app.py analyze ETH/USDT

# Test the Operator math with a sample LONG trade
uv run app.py mock mock_long.json

# Ask the AI a question
uv run app.py ask "What does RSI measure?"

# Built-in help for any command
uv run app.py analyze --help
```

---

## The Two Strategies

Agent 3 runs as two different "managers" so their results can be compared side by side (A/B testing):

| | 🛡️ Defensive (`--def`) | 💰 Greedy (`--greed`) |
|---|---|---|
| Main goal | Protect capital | Find favorable risk/reward |
| Timeframes disagree | Sit on hands | May trade a pullback or mean reversion |
| Any agent is NEUTRAL | Sit on hands | Not a veto by itself |
| Trades when | Agents 1 & 2 fully agree and risk is favorable | Exhaustion at a key level, a confirmed breakout, or a strong reward-to-risk ratio |
| Sits out when | Any conflict at all | Price is stuck mid-range with no target, or heavy volume is pushing against the setup |

Each strategy keeps its own separate ledger and trade history.

---

## Where Your Data Is Saved

These folders are created automatically the first time you run a command:

```
output_alpha/                                 ← main results (primary AI model)
├── analyze/<strategy>/<YYYY-MM>/             ← one JSON file per analysis run
├── operate/<strategy>/<YYYY-MM>/             ← one JSON file per trade ticket
├── defensive/BTC_USDT_paper_ledger.json      ← the currently OPEN paper trade
├── defensive/BTC_USDT_trade_history.json     ← all CLOSED trades with WIN/LOSS and PnL
├── greedy/...                                ← same files for the Greedy strategy
└── operator_errors.log                       ← rejected (invalid) trade tickets
output_beta/                                  ← same layout, used when the backup AI model ran
logs/system_health.log                        ← record of every switch to the backup model
error.log                                     ← technical error details
```

Analysis files are named like `20261002_143000_BTCUSDT_DEF_analysis.json` (date, time, symbol, strategy).

### Versions and traceability in each record

Every analysis, status, ticket, ledger and history record carries these fields (in `metadata` for analyses and tickets, at the top level for ledgers and history):

| Field | Meaning |
|---|---|
| `schema_version` | Structure of the record. Currently `1`. Records without it are legacy (`0`). |
| `strategy_version` | The trading rules in force when the record was written. `0.x` is warm-up data (before the Phase 1 fixes), `1.0` will be the first official version. Records without it are legacy (`0.0`). Never mix versions in one performance figure. |
| `exchange`, `market_type` | Where the market data came from (today `binance`, `spot`). |
| `timestamp_utc` / `entry_time_utc` | When the record was written / the trade was opened, in UTC. The older `timestamp` field is local time and kept as it was. |
| `run_id` (analysis) | Unique id such as `20261002T123000Z-DEF-BTCUSDT`. |
| `models_used` | The model that answered **each** agent call, e.g. `{"agent_1_technical": "...", "agent_2_volume": "...", "agent_3_defensive": "..."}`. Compare with `models_configured` to see which calls used the backup model. |
| `trade_id`, `analysis_file`, `analysis_run_id` (tickets, ledger, history) | Which trade this is and the analysis that opened it. |
| `resolved_at_utc`, `resolved_by_strategy_version` (history) | When the trade was closed and under which rules. The trade still counts under the version it was **opened** with. |

Files written before these fields existed are never edited to add them; the program reads them as legacy.

---

## Testing

Tests use `pytest` (a dev dependency). They run fully offline: a fake exchange serves saved candles, a fake Gemini client returns canned replies, and every test works in its own temporary folder, so `output_alpha/`, `logs/` and your Gemini quota are never touched.

```powershell
uv run pytest
```

`tests/characterization/` records exactly what the tool does today: indicator values, the prompt sent to each agent, the analysis, ticket, ledger and history files, and the console output. The expected outputs live in `tests/characterization/snapshots/`. Some of them pin known bugs from `TODO.md` on purpose. A snapshot difference means behaviour changed. Only after that change is intended and approved, regenerate the snapshots and review the diff:

```powershell
$env:UPDATE_SNAPSHOTS = "1"; uv run pytest; Remove-Item Env:UPDATE_SNAPSHOTS
git diff tests/characterization/snapshots
```

To check the Operator's trade math without using the AI or internet:

```powershell
uv run app.py mock mock_long.json
uv run app.py mock mock_short.json
```

A mock file must contain: `verdict`, `account_balance_usdt`, `risk_per_trade_percent`, `current_price`, `atr_14`, `agent_1_threat_level`, `agent_2_magnet_target`.

---

## Project Files

| File / Folder | Purpose |
|---|---|
| `app.py` | Entry point: loads `.env`, sets up logging and starts the CLI, so `uv run app.py <command>` works |
| `btc_cli/` | The application package. `cli.py`: commands. `pipeline.py`: the status / analyze / operate / mock flows. `data.py`: exchange data. `indicators.py`: EMAs, RSI, ATR, volume profile. `agents.py`: Gemini prompts and model fallback. `trade_operator.py`: entry, stop, target and size math. `ledger.py`: WIN/LOSS and PnL. `storage.py`: every file the tool writes. `console.py`: terminal panels. `config.py`: settings. `logging_setup.py`: error log |
| `tests/` | Offline tests (`uv run pytest`); `tests/characterization/` holds the behaviour snapshots, `tests/fixtures/` the saved candles |
| `mock_json/` | Sample trade payloads for the `mock` command |
| `list_models.py` | Helper that prints every Gemini model your API key can use |
| `pyproject.toml` / `uv.lock` | Library list and exact pinned versions, managed by uv |
| `AGENTS.md` | Instructions for AI coding assistants working on this repo |
| `CHANGELOG.md` | **Record of every change and decision**, newest first. Read this to see what changed and why. |
| `TODO.md` | Audit findings and the phased roadmap of work still to do |
| `synthetic_data/` | **Planned.** Generated Bitcoin-like data for testing the trading rules offline. See its `README.md` and `PLAN.md` |
| `research/` | Study scripts and their dated reports. Run `uv run research/activity_gate_study.py` monthly to re-check the activity gate thresholds |

---

## Roadmap

**Completed**
- ✅ Phase 1–2: Live data engine and core indicators
- ✅ Phase 3: Multi-agent AI pipeline
- ✅ Phase 4: Advanced metrics (ATR, Value Area, mean reversion) and strict JSON output
- ✅ Python Operator with a volatility-floored stop loss
- ✅ Paper-trading ledger with automatic WIN/LOSS resolution
- ✅ Defensive vs. Greedy A/B testing
- ✅ Separate `status` / `analyze` / `operate` / `auto` commands, and a `mock` test harness
- ✅ AI model fallback with separate (`output_beta`) data storage
- ✅ Automatic JSON logging of every run

**Possible improvements**
- [ ] Smarter take-profit logic (e.g. partial exits, trailing stops)
- [ ] A learning feedback loop that uses past trade results to improve the agents
- [ ] A proper database instead of JSON files
- [ ] Printable / exportable performance reports
