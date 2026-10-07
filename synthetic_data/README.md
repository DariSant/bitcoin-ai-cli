# Synthetic market data

> **Status: planned, not implemented.** This folder holds only the plan so far. The owner answered the plan's open questions on 2026-10-07, and will review [PLAN.md](PLAN.md) as a whole before any code is written. Roadmap entry: `TODO.md`, Phase 1, "Synthetic market data for testing the rules".

## What this is for

Generated, Bitcoin-like price data, made first by reshuffling 3 years of real BTC USDT-M perpetual history, used to test the **mechanics** of the strategy offline, with known correct answers:

- **Trade resolution:** stop and target hit in the same minute, a hit inside the entry minute, missing minutes, trades held past 24 h.
- **The Operator and cost model:** stop floor, minimum R:R, maximum target distance, fees, slippage, funding.
- **The activity gate:** quiet, low-volume periods that should block trading.
- **A no-edge check (null test).** On data with no edge, every rule set should average about **−costs per trade, in R**. A profit there means a bug or look-ahead, not skill.

## What this is not for

- **It cannot show whether the AI has an edge.** Synthetic data has no real edge to find.
- **It never replaces the paper-trading run.** That run, on real data from Oracle Cloud, is the only evidence that decides about real money.
- **The real Gemini agents are never run on this data.** It would spend the shared free-tier quota (`AGENTS.md` §2.7) and prove nothing. A simple rule-based stand-in makes the verdicts instead.

## Rules

- Output never goes into `output_alpha/` or `output_beta/`, and synthetic results are never mixed into performance figures.
- Every generated dataset is reproducible: the same seed and parameters give byte-identical files.
- Only the generator, its parameters and small hand-made scenarios are committed. Large generated files are rebuilt on demand and ignored by Git.
- The calibration step reads public market data only. No API keys, no orders (`AGENTS.md` §2.1).
- Changing this folder never changes `strategy_version`, because it doesn't touch how real trades are taken or scored.

## Planned layout

```
synthetic_data/
├── README.md          # this file
├── PLAN.md            # implementation plan (for review)
├── calibrate.py       # (planned) measures real BTC statistics from public 1m data
├── generate.py        # (planned) builds a dataset from a seed and parameters
├── validate.py        # (planned) compares a dataset's statistics with real BTC
├── params/            # (planned) committed parameter files, one per dataset
├── scenarios/         # (planned) small hand-made cases with expected outcomes, used by tests/
└── output/            # (planned, Git-ignored) generated multi-year datasets
```

## Data format (planned)

Each dataset or scenario is one folder:

| File | Content |
|---|---|
| `meta.json` | Seed, generator name and version, parameters, regime labels, symbol, market type, UTC start and end |
| `candles_1m.csv.gz` | `ts_ms, open, high, low, close, volume`: the single source of truth |
| `trades.csv.gz` | `ts_ms, price, qty, side`: ticks, only for the minutes that need them |
| `funding.csv` | `ts_ms, rate`: every 8 h (perpetual futures) |
| `events.json` | Cases injected on purpose, with timestamps: gaps, flash wicks, both-levels minutes, volume droughts |
| `expected.json` | Hand-made scenarios only: the correct outcome of each trade |

15m and 4h candles are never stored. They are built from the 1m candles, so the timeframes always agree.

## Size

Three years is about 1.58 million 1m candles, roughly 20–30 MB gzipped. It generates in seconds and uses about 75 MB of memory.

## Usage

None yet. Commands will be added here once the plan is approved and implemented.
