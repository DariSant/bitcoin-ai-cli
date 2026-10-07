# Implementation plan: synthetic market data

**Status:** draft, written 2026-10-07, for owner review. Nothing here is approved or implemented yet.
**Roadmap:** `TODO.md`, Phase 1, "Synthetic market data for testing the rules".

## 1. Goal

Produce reproducible, Bitcoin-like market data (1m candles, ticks where needed, funding) so the strategy's rules can be tested offline with known answers:

1. **Exact tests:** small hand-made scenarios with an `expected.json`, run by `uv run pytest`. They give the Phase 1 resolution work its test cases (entry minute, same-minute stop and target, data gaps, 24 h limit).
2. **No-edge check:** 2–3 years of edgeless data run through the real Operator, cost model and resolution, with a rule-based stand-in for the AI verdict. The expected result is about −costs per trade in R. Anything better means a bug or look-ahead.
3. **Stress runs:** flash crashes, volatility spikes, low-volume droughts and exchange gaps, to check the gate, the stop floor and the data-gap handling.

It does **not** measure the AI's edge (see [README.md](README.md)).

## 2. Files touched

| File | Change |
|---|---|
| `synthetic_data/calibrate.py` | New. Fetches public 1m history and writes the measured statistics to `params/` |
| `synthetic_data/generate.py` | New. Seed + parameter file → dataset folder |
| `synthetic_data/validate.py` | New. Statistics report: synthetic vs real |
| `synthetic_data/params/*.json` | New. Committed parameters (a few kB each) |
| `synthetic_data/scenarios/<name>/` | New. Hand-made cases with `expected.json` (small, committed) |
| `synthetic_data/output/` | New, Git-ignored. Generated multi-year datasets |
| `.gitignore` | Add `synthetic_data/output/` and the calibration cache |
| `tests/test_synthetic_data.py` | New. Generator tests (see §6) |
| `tests/conftest.py` | Small extension: let the fake exchange serve a scenario folder (1m, ticks, funding) |
| `README.md`, `TODO.md`, `CHANGELOG.md`, `LESSONS.md` | Docs |

No change to `btc_cli/` and no change to recorded data. No new dependency: `numpy` and `pandas` are already installed. Parquet would need `pyarrow`, which needs approval; gzipped CSV is enough for now.

## 3. Approach

### 3.1 Calibration (public data only)
- Download 2–3 years of BTC 1m candles from the market chosen in Phase 1 (USDT-M perpetual, `binanceusdm`, with OKX or Bybit as backup), plus the funding history.
- Use `ccxt` public endpoints, paginated, with timeouts, retries and rate-limit pauses (`AGENTS.md` §5).
- Cache the download in a Git-ignored folder, because it is about 1.5 M rows.
- Measure and save:
  - volatility per timeframe, and the tail heaviness of returns
  - how strongly calm and wild periods cluster (autocorrelation of absolute returns)
  - how volume tracks the size of price moves, and the daily volume rhythm
  - typical candle wick sizes, and the funding rate distribution
- Record the source, date range and download date in the parameter file.

### 3.2 Generators (two, same output format)
1. **Block bootstrap (do first).** Cut real 1m returns, with their volume, into blocks of a few days and reshuffle them with a seeded random generator.
   - Keeps real texture: tails, clustering, wick shapes, volume.
   - Destroys the real order of history, so there is no edge to find, and the AI couldn't recognise the prices even in theory.
   - Prices are rebuilt from the reshuffled returns, starting at a chosen price.
2. **Calibrated model (second).** A regime-switching random walk:
   - trend, range, and high/low-volatility regimes, with switching probabilities from calibration
   - heavy-tailed shocks (Student-t), volatility that clusters (GARCH-style)
   - volume tied to the size of moves, with a daily rhythm
   - Fully synthetic, and parameters can be dialled up for stress tests.

### 3.3 Building candles and ticks
- 1m candles are the source. 15m and 4h are built from them on UTC-aligned boundaries, matching how the exchange forms candles.
- Ticks are generated only for minutes that need them:
  - They must fit the 1m candle exactly: first trade = open, last = close, the highest and lowest trades = high and low, quantities sum to the volume.
  - The order in which high and low are reached is seeded, so same-minute stop-and-target cases have a known answer.
- Funding is an 8 h series drawn from the calibrated distribution.

### 3.4 Injected events
`events.json` lists everything added on purpose, with timestamps, so a test can tell an injected gap from a bug:
- missing minutes (data gap)
- a flash wick through both levels
- a stop and target touched in the same minute
- a volume drought (should trip the activity gate)
- a run of identical prices (exchange stall)

### 3.5 How it plugs in
- **Tests:** the fake exchange in `tests/conftest.py` serves a scenario folder, through the same calls the real code will use after Phase 1 (1m candles, ticks, funding). Tests compare the closed trade with `expected.json`.
- **No-edge check:** a `research/` script walks forward through a dataset minute by minute, with no look-ahead:
  - The stand-in verdict is one of: always long, always short, random, or a simple EMA-cross rule.
  - The real pure modules do the rest: `btc_cli.trade_operator`, `btc_cli.ledger`, and the Phase 1 cost model and activity gate.
  - Output: a report in `research/results/` with trade count, average R ± standard error, win rate, and cost per trade. Nothing is written to `output_alpha/`.
- **No Gemini calls**, ever.

### 3.6 Reproducibility
- `meta.json` stores the seed, generator name, generator version and a hash of the parameter file.
- The same inputs must give byte-identical files, and a test checks this.
- Large datasets are rebuilt from `params/` + seed instead of being committed.

## 4. Order of work

| Step | Content | Depends on | Effort |
|---|---|---|---|
| 1 | Output format, the 1m → 15m/4h builder, tick builder, `meta.json`; generator tests | nothing | S–M |
| 2 | Hand-made `scenarios/` for the Phase 1 resolution cases, with `expected.json`; fake-exchange support | step 1; written alongside the Phase 1 1m/tick resolution item | M |
| 3 | `calibrate.py` + cached public download | Phase 1 market decision (futures exchange) | M |
| 4 | Block-bootstrap generator + `validate.py` report | step 3 | M |
| 5 | No-edge walk-forward script in `research/` | steps 4 and Phase 1 cost model, gate and resolution | M |
| 6 | Calibrated-model generator + stress presets | step 3 | M |

Steps 1–2 are the most valuable and can start with Phase 1. Steps 3–6 can follow during or after Phase 1.

## 5. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Synthetic results get mistaken for real performance | Bad money decision | Results are always labelled synthetic, never written to `output_alpha/`, never mixed into reports |
| Data unlike real BTC (too smooth, wrong tails) | Gate and stop thresholds tested on unrealistic paths | Block bootstrap first; `validate.py` compares statistics with real data before a dataset is used |
| Bootstrap blocks join with unrealistic jumps | Fake gaps at block edges | Rebuild prices from returns, not raw prices; block length of days, not minutes |
| Ticks that don't fit their 1m candle | Resolution tests that pass for the wrong reason | Generator tests check every tick-filled minute fits exactly |
| Large files committed by mistake | Repo bloat | `output/` and the cache are Git-ignored; a test fails if a committed scenario exceeds a size limit |
| Calibration download is blocked or rate-limited | Step 3 is delayed | Cache, resumable pagination, backup exchange; steps 1–2 don't need it |
| Phase 1 changes the data interface | Rework in step 2 | Step 2 is written together with the Phase 1 resolution item |

**Effect on collected data:** none. Nothing here changes how real trades are taken or scored, so no `strategy_version` bump.

## 6. Testing

- Building 15m/4h from 1m matches a straightforward reference grouping, including UTC day and 4h boundaries.
- Every tick-filled minute satisfies open, close, high, low and volume exactly.
- The same seed and parameters give byte-identical output; a different seed gives different output.
- Injected events appear exactly where `events.json` says.
- For the block bootstrap, the returns are a reshuffle of the source (same set of values), and no candle carries a real timestamp.
- A tiny random walk run through the no-edge script gives an average R consistent with −costs, within its standard error.
- All tests offline: the calibration download is mocked, and generators run on small sizes.

## 7. Cost and size

- Building it: roughly $5–10 of Claude usage at API prices (or part of one session on a subscription). Running it: $0, with no Gemini calls.
- Three years of 1m data: about 1.58 M candles, 20–30 MB gzipped, generated in seconds, about 75 MB of memory.
- Three years at one decision per 30 min gives about 500–1,000 trades per strategy. That's enough to measure the average R to about ±0.05R.

## 8. Open questions for the owner

1. **Generator order:** block bootstrap first, then the calibrated model? (Recommended.)
2. **History depth for calibration:** 3 years? (2 years is enough if the download is slow.)
3. **Calibration market:** the Phase 1 futures market (`binanceusdm`), or spot until Phase 1 lands?
4. **Folder:** keep generated datasets in `synthetic_data/output/` (Git-ignored), as planned here, or under `research/results/`?
5. **AGENTS.md:** add `synthetic_data/` to the §10 layout. That needs your approval, because agents may not edit `AGENTS.md`.
