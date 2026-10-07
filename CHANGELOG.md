# Changelog

This file records **every change made to the project and every decision taken**, in plain language, newest first.
The plan of work still to do lives in [TODO.md](TODO.md); this file records what has actually happened.

**How to add an entry** (for people and AI assistants alike)
- Add your entry at the top of the newest date section (or start a new section `## YYYY-MM-DD`, UTC date).
- Put each line under one of these headings: **Decisions**, **Added**, **Changed**, **Fixed**, **Removed**, **Research**, **Maintenance**.
- Say *what* changed and *why*, in one or two sentences. Name the file (and the commit or PR, if there is one).
- When a TODO item is finished, tick its box in `TODO.md` and mention it here.
- After each monthly threshold re-check, add a row to the [re-check log](#activity-gate-threshold-re-check-log) below.

---

## 2026-10-07

### Fixed
- **`operate` picks the analysis by its recorded time, with bounded work** (branch `fix/operate-picks-analysis-by-time`; `TODO.md` Phase 2 P2 item ticked, M1.7; **M1 is now complete**). Before, every `operate` listed every analysis ever written, so it got slower every day on the small server. It also chose by file modification time, so after a backup restore or copy it could trade an older analysis.
  - **Bounded:** at most two month folders are listed (the ones the 10-minute window touches, so a 23:58 analysis on the last day of a month is still found). Only files named inside the window, plus an hour for clock changes, are opened.
  - **Choice:** the newest by recorded time (`timestamp_utc`, or the local `timestamp` for legacy records) wins.
  - **Nothing recent:** the newest by name is returned, so the "not in the last 10 minutes" message is unchanged.
  - **Messages:** one change. With no analysis in those folders, the message is always "No recent analysis found for <STRATEGY> strategy." The variant naming the symbol is gone.
  - Normal runs pick the same analysis as before. Snapshots unchanged. 6 new tests (190 in total, passing under three time zones). The test that pinned the modification-time behaviour now checks the fix.
- **`mock` now checks the real Operator math** (branch `fix/mock-uses-operator-math`; `TODO.md` Phase 1 item ticked, M1.6). It had its own copy of the stop rule without operate's 1-ATR floor, so it could show a ticket `operate` would never produce.
  - `mock` now calls `trade_operator.compute_order`, and the duplicate `compute_mock_order` is removed.
  - New `mock_json/mock_floor.json`, where the floor decides: SL 69,000, size $7,000. The old mock showed SL 69,400 and $11,667.
  - `mock_long` and `mock_short` print exactly as before.
  - `mock` writes nothing and opens no trade, so no `strategy_version` bump. 3 new tests check that mock and operate give the same ticket (184 in total).
- **Binance calls are retried, and an outage no longer stops the whole run** (branch `fix/binance-retries`; `TODO.md` Phase 2 item ticked, M1.5). Before, every fetch created a new exchange object with no timeout, which also downloaded Binance's market list again each time. One network error while checking an open trade printed the raw exception and stopped the run.
  - **Shared client:** one exchange object per process, with a 15 s timeout and ccxt's rate limiting.
  - **Retries:** temporary network errors (`ccxt.NetworkError`: timeouts, rate limits, maintenance) are retried after 2 s and 4 s, at most 3 attempts. Permanent errors (`ExchangeError`: bad symbol or request) are not retried.
  - **Open-trade check:** if its candles still can't be fetched, the trade **stays open** and is checked again next run (§2.6: never guess). Only that strategy is skipped, the message is short (details in `error.log`), and the command exits 1.
  - **Settings:** new `[exchange_requests]` section (`timeout_seconds`, `max_attempts` 1–3), not frozen.
  - No trade decision changes, so no `strategy_version` bump. Snapshots unchanged. 13 new tests (181 in total), and one characterization test updated for the new outage behaviour.
- **AI replies are now validated** (branch `fix/validate-ai-replies`; `TODO.md` Phase 2 item ticked, M1.4). Before, only "is it JSON?" was checked. A missing field silently became `NEUTRAL` or `SIT ON HANDS`, and a JSON error in one strategy's Agent 3 stopped the whole run, so the other strategy got no analysis that cycle and the A/B arms drifted apart.
  - `agents.parse_reply` checks every reply against the same `TypedDict` schema Gemini is given: required fields, text types, allowed values such as `final_verdict` and `bias`. What Gemini receives is unchanged.
  - **Invalid Agent 3 reply:** only that strategy is skipped and the other still runs.
  - **Invalid Agent 1 or 2 reply:** the cycle is skipped, but `auto` still runs `operate` (before, it stopped the whole run).
  - In both cases the raw reply goes to `error.log` and the command exits 1. Valid replies are saved exactly as before; every snapshot is unchanged.
  - `pydantic` added as a direct dependency (`uv add pydantic`, approved 2026-10-04; same version as before, now declared). `pyproject.toml` and `uv.lock` updated.
  - No trade decision changes: an invalid reply never led to a trade before either. 9 new tests (168 in total).
- **Gemini calls now have a timeout, sensible retries and a stricter fallback** (branch `fix/gemini-error-handling`; `TODO.md` Phase 2 item ticked, M1.3). Before, any error switched straight to the backup model with no retry and no timeout. A bad API key wasted a call on the backup, and the backup's own error was thrown away.
  - **Timeout:** every request times out after 60 s. The SDK never retries by default, so these retries are the only ones.
  - **Temporary errors:** 5xx, timeouts and connection errors are retried after 2 s and 4 s, at most 3 attempts per model (§5).
  - **Rate limits (429):** the call waits the delay Gemini asks for if it is ≤ 60 s. A longer one (e.g. the daily quota is used up) falls back at once.
  - **Bad request, key or permission (400/401/403):** the run stops with exit 1 and no fallback, because the backup would fail the same way. A missing model (404) and unexpected errors fall back at once, as before.
  - **Sticky fallback:** once the primary fails, the rest of the run goes straight to the fallback instead of retrying the dead primary for each agent. As before, the whole run's data goes to `output_beta/`.
  - **Logging:** the fallback's error is now logged. `logs/system_health.log` keeps its format.
  - **Settings:** new `[gemini_requests]` section in `config.toml` (`timeout_seconds`, `max_attempts` 1–3, `max_retry_wait_seconds`), **not frozen**: it changes how calls are retried, not what the models are asked. `tests/test_config.py` now pins only the frozen fields and fails if a new setting isn't classified as frozen or not.
  - No `strategy_version` bump: models, prompts and generation settings are unchanged, and every record names the model actually used (§2.4).
  - Tests: `tests/test_agents.py` (15, each checked to fail on the old code) plus config range checks, 159 in total. Three fallback tests were updated for the sticky fallback.

### Fixed
- **A total AI failure no longer exits with "success"** (branch `fix/ai-failure-exit-code`; `TODO.md` Phase 2 item ticked, M1.2). Before, "Both models unreachable" printed a message but exited 0, so a scheduler or heartbeat would have counted the run as a success.
  - `analyze` and `auto` now exit 1 when neither model answers, whether for Agent 1, Agent 2 or one strategy's Agent 3.
  - The other strategy still runs, and `auto` still runs `operate`, so open trades are still resolved.
  - `ask` now exits 1 when both models are down or on a Gemini API error. Its broad `except Exception` used to catch its own exit, and errors are now logged.
  - No trade decision changes, so no `strategy_version` bump. Three characterization tests that pinned the old exit 0 were updated; two tests are new (139 in total).

### Maintenance
- **`.gitattributes`** (`* text=auto eol=lf`, branch `chore/gitattributes`; `TODO.md` line-ending item ticked, M1.8). Git now keeps LF line endings on Windows too, so the "LF will be replaced by CRLF" warnings stop and the Windows and Linux copies match. Every tracked file was already stored with LF, so no content changed.

### Research
- **Tick data study** (`research/tick_data_study.py`, report `research/results/tick_data_2026-10-07.md`; M1.1 in `TODO.md`). This answers the §2.6 question that had to be settled before the precise trade-resolution design.
  - **Binance USDT-M:** individual trades are available at any age, through REST for the last 48 h and the public daily archive before that. They are complete: a whole day's trades match the candle volume exactly.
  - **Binance candles:** they sometimes count a trade at a minute boundary in the next minute, which moved a high or low by one tick in 4 of 1,440 minutes. So resolution will decide by trade timestamps and check the neighbouring minutes' edges.
  - **Backups:** OKX REST goes back about 3 months, needs the raw endpoint paged by trade id, and every tested minute rebuilt exactly. Bybit has daily archive files only, and its candles open at the previous close. OKX is recommended as the backup exchange.
  - Public data only, no Gemini calls, about 60 MB downloaded.

### Maintenance
- **`TODO.md` reviewed:** new section 0 "Status and next steps", with a progress table per phase, a milestone plan (M1 robustness groundwork → M2 Phase 1 batch, versions 0.2 to 1.0 → M3 ready for unattended running → M4 deploy → M5 evaluate), and the owner decisions needed. Progress notes were added to four partly done items.
- **PR fix:** #35 (safe storage) had merged into its stacked base branch instead of `main`. It is re-opened as #39, and #36–#38 now target `main`. A lesson was recorded.

### Decisions (owner)
- **Phase 1 starts with record versioning** (the "Version every record" item), following the plan proposed in the conversation.
  - **Version numbering:** `strategy_version` is a string. Legacy records = `"0.0"`, today's code = `"0.1"`. Each approved Phase 1 fix bumps `0.x`, and the first official version is `"1.0"`. Anything below `1.0` is warm-up data.
  - **Rejections:** `operator_errors.log` stays free text for now. Rejections get a structured record in the Phase 4 item "Record every decision", together with the new Phase 1 rejection reasons.
  - **Trades spanning a version change** record both versions (`strategy_version` at entry, `resolved_by_strategy_version` at close) and count under the version they were **opened** with.
- **Synthetic market data.** Create a `synthetic_data/` folder with an implementation plan and a README, to be reviewed later, and add the work to the roadmap. Nothing is implemented yet.
  - Answers to the plan's questions:
    - Shuffled real history (block bootstrap) is the first generator.
    - Calibrate on 3 years of 1m data from the futures market (`binanceusdm`, with funding).
    - Generated datasets go in `synthetic_data/output/`.
  - Add `synthetic_data/` to the `AGENTS.md` §10 layout (approved edit, done).
- **Two modules added to the `btc_cli/` layout:**
  - `pipeline.py` holds the status / analyze / operate / mock flows, so `cli.py` keeps only commands and display.
  - `console.py` holds the shared console and panels, so no module has to import the CLI.
- **Delete leftover files** that won't be needed: `main.py` and `bitcoin_ai_cli.egg-info/`.
- **Push to GitHub** once Phase 0 is finished.

### Fixed
- **Two runs can no longer overlap** (branch `fix/run-lock`; `TODO.md` Phase 2 item ticked). A scheduled run and a manual one could both read "no open trade" and both open a trade.
  - `status`, `analyze`, `operate` and `auto` now hold an OS file lock on `run.lock` in the data folder. `auto` holds it across all three steps.
  - A second run prints who holds the lock (PID, command, start time), does nothing, and exits with code **3**. That is separate from errors (1) and usage mistakes (2), so a scheduler can tell "skipped" from "failed".
  - The OS releases the lock when the holder ends, even after a crash or kill, so stale locks can't happen and `run.lock` never needs deleting (`AGENTS.md` §5). The PID-check design in `TODO.md` was dropped, because `os.kill(pid, 0)` sends Ctrl+C on Windows.
  - `mock`, `ask` and `commands` don't take the lock. `run.lock` is Git-ignored.
  - Tests: 11 new (137 in total), including a real second process that holds the lock and one that is killed while holding it. Only the Windows lock code ran here; the Linux `fcntl` branch will first run in the test suite on the server.
  - `README.md` now lists the exit codes.
- **Paths no longer depend on the folder a command is started from** (branch `fix/project-root-paths`; `TODO.md` Phase 3 item ticked). Before this, a scheduler starting the program from another folder would have created new, empty ledgers there, and open trades would have looked "missing".
  - All paths are absolute, built from the project root. The new `[paths] data_dir` setting in `config.toml` (default `"."`, the project folder) holds `output_alpha/`, `output_beta/`, `logs/` and `error.log`, so with the default nothing moves. `mock_json/` is read from the project folder.
  - The environment variable `BTC_CLI_DATA_DIR` overrides the data folder. Development runs can now use a temporary folder and never touch the real dataset (`AGENTS.md` §2.7). `README.md` shows the commands.
  - Console output is unchanged: paths are still shown relative to the data folder.
  - Tests: 7 new (126 in total). The test harness now points every data path into the test's temp folder, and a session-wide guard fails the run if any real data file (`output_alpha/`, `output_beta/`, `logs/`, `error.log`) changes. Only file sizes and times are read.

### Changed
- **Settings moved into `config.toml`** (branch `refactor/config-toml`; `TODO.md` Phase 3 config item ticked). The values are unchanged, so there is no `strategy_version` bump. Tests, both `mock` commands and every snapshot give the same output as before.
  - Sections: `[market]` (exchange, market type, candle counts), `[gemini]` (models), `[indicators]` (swing lookback, volume-profile bins, value-area share) and `[operator]` (risk $, ATR multipliers, 10-minute staleness, payload balance and risk %). All are marked `[frozen]` (§2.4).
  - `btc_cli/config.py` checks types and ranges at start-up and stops with a one-line message, e.g. `Settings error in config.toml: [operator] risk_usd must be greater than 0, found -1.0`.
  - Indicator lengths that appear in field names (`ema_34`, `rsi_13`, `vma_20`, `atr_14`) stay in code, so the names can't drift from the values.
  - `tests/test_config.py` (16 tests) includes a freeze guard that pins the 0.1 values.
  - `tests/test_architecture.py` now accepts `from btc_cli import config` in the pure modules, as `AGENTS.md` §10 allows. It used to flag the package name itself.

### Fixed
- **Recorded files are now crash-safe, and damaged files are never overwritten** (branch `fix/safe-storage`; `TODO.md` Phase 2, three items ticked). These are bug fixes outside the strategy freeze: no trade decision changes, so no `strategy_version` bump.
  - Analyses, tickets, ledgers and history are written atomically (`storage.write_json_atomic`: temp file in the same folder, `fsync`, `os.replace`). The bytes written are identical to before, and every snapshot is unchanged.
  - A damaged **history** file used to be silently replaced, losing all earlier closed trades. Now nothing is written, the file keeps its bytes, and the trade stays OPEN.
  - A damaged **ledger** used to read as "no open trade", so a new trade could overwrite it. Now it is left untouched.
  - In both cases the error goes to `error.log`, that strategy is blocked, the other strategy still runs, and `analyze`, `operate` and `auto` exit with code 1.
  - A trade that is already in history (crash between "append to history" and "delete ledger") is not appended again. Trades are matched by `trade_id`, or by entry details for legacy trades.
  - The `TODO.md` fix wording "rename the damaged file to `…corrupt-<time>.json`" was corrected: `AGENTS.md` §2.3 forbids renaming recorded files.
  - Tests: 13 new (103 in total). The two characterization tests that pinned the old bugs now check the corrected behaviour.

### Changed
- **Every record is now versioned and traceable** (`TODO.md` Phase 1, "Version every record") on branch `feat/record-versioning`. This is a record format change (`schema_version` 1). Trade decisions and resolution are unchanged, so `strategy_version` stays at its first value, `"0.1"` (warm-up).
  - Analysis and status records (`metadata`) now include `schema_version`, `strategy_version`, `exchange`, `market_type`, `timestamp_utc`, `strategy` and `run_id`. Analyses also include `models_used`, the model that answered **each** agent call, and `models_configured`. Before this, a record didn't say which model produced it.
  - Tickets, ledgers and history entries carry the same version and source fields, plus `trade_id`, `analysis_file` and `analysis_run_id`, so every trade links back to the analysis that opened it. Ledgers also gain `entry_time_utc`. History entries gain `resolved_at_utc` and `resolved_by_strategy_version`.
  - Fields are only added, never renamed or removed. The local-time `timestamp` stays, because the 10-minute staleness check still reads it (separate TODO item).
  - Files already in `output_alpha/` are not touched. Readers treat them as legacy (schema 0, strategy `"0.0"`) through `storage.schema_version_of` / `strategy_version_of`.
  - `btc_cli/config.py` gains `SCHEMA_VERSION`, `STRATEGY_VERSION`, `EXCHANGE_ID` and `MARKET_TYPE`. `data.create_exchange()` now builds the exchange from `EXCHANGE_ID` (still `ccxt.binance()`), so a record can never name a different exchange than the one actually used.
  - Tests: new `tests/test_storage.py` and `tests/test_pipeline.py` (90 tests in total, passing under three time zones). The four record snapshots were regenerated on purpose. A script confirmed that every old key and value survives unchanged and that no console or prompt snapshot changed. Three characterization tests were updated for the new fields. One of them used to pin "the record does not say which model produced it" and now checks `models_used`.
  - `README.md`: new section "Versions and traceability in each record".
- **`app.py` split into the `btc_cli/` package (Phase 0 is now complete)** on branch `refactor/btc-cli-package`. There is no behaviour change, so no `strategy_version` bump:
  - Every characterization test and snapshot passed unchanged.
  - Both `mock` commands, `commands` and every `--help` give byte-identical output before and after.
  - Known bugs moved as they were. Prompts and indicator code were copied by line range, not retyped.
  - `app.py` is now a thin entry point (loads `.env`, sets up logging, starts the CLI), so `uv run app.py <command>` works as before.
  - In `TODO.md`, references like `app.py:Lnnn` point to the pre-split file (`git show 98ee1d7:app.py`).
- `pyproject.toml`: a real project description replaces "Add your description here". `.gitignore` now ignores `*.egg-info/`.

### Removed
- `main.py` (unused starter stub) and `bitcoin_ai_cli.egg-info/` (old build leftover). Part of the `TODO.md` cleanup item, which stays open for the `requests` decision and two small code items.

### Added
- `synthetic_data/README.md` and `synthetic_data/PLAN.md`: the purpose, rules, data format and step-by-step plan for synthetic data. It is for testing the rules (resolution, costs, activity gate, a no-edge check), not for measuring the AI's edge.
- `TODO.md`: new Phase 1 item "Synthetic market data for testing the rules". `README.md`: the folder is listed as planned.
- `AGENTS.md` §10: `synthetic_data/` added to the target layout (owner approved 2026-10-07).
- `AGENTS.md` (owner approved 2026-10-07): §10 layout now lists `pipeline.py` and `console.py`, the note about deleting `main.py` and the egg-info is removed (both are gone), the `tests/` line is updated, and §7 drops the `uv run --with pytest` fallback (`pytest` is a dev dependency).
- **Characterization tests (Phase 0, step 1)** on branch `test/characterization-tests`. There are 43 offline tests that record exactly what the tool does today, so the `app.py` split can prove it changed nothing.
  - They drive the real CLI (`status`, `analyze`, `operate`, `auto`, `mock`, `commands`, `ask`) with a fake exchange serving saved candles (`tests/fixtures/`), a fake Gemini client with canned replies, and a frozen clock.
  - They capture the indicator values, the exact prompt sent to each agent, the analysis, ticket, ledger and history files for a long, a short and a `SIT ON HANDS` case, trade resolution, model fallback routing, and the console output. Expected outputs are in `tests/characterization/snapshots/`.
  - Known bugs are pinned as they behave today, not fixed: the entry candle is ignored, a candle touching both levels is a LOSS, only 25 h is checked, a damaged history file is overwritten, a total AI failure exits 0, one fallback moves the whole run to `output_beta`, and `mock` uses different stop math. These tests will change on purpose when each Phase 1/2 fix is approved.
  - Safety: every test runs in a temporary folder with the network blocked, and `.env` is never loaded. `output_alpha/`, `logs/`, `error.log` and the Gemini quota are never touched (checked after the run).
- Unit tests for `trade_operator` and `ledger`, plus `tests/test_architecture.py`, which checks the §10 rules: the pure modules do no I/O, and nothing imports the CLI.
- Characterization test for a new finding: `ask` exits 0 when both models are down, because its `except Exception` also catches its own `typer.Exit`. Added to the `TODO.md` "Total AI failure exits with success" item.
- `pytest` added as a dev dependency (`uv add --dev pytest`, approved 2026-10-04), plus pytest settings in `pyproject.toml`. Tests now run with `uv run pytest`.

### Changed
- `test_app.py` moved to `tests/test_app.py` (part of the approved §10 layout).
- `README.md`: the Testing section and the file table now describe `tests/` and how to update snapshots.
- `TODO.md`: ticked the characterization-tests item and the model-name item (done since PR #28 was merged). Added progress notes to the split and test-suite items. New P2 finding: `operate` scans every saved analysis and picks one by file modification time.

---

## 2026-10-05

### Decisions (owner)
- **New agent instructions.** The revised `NEW_AGENTS.md` replaces the old `AGENTS.md`, which is deleted. The file keeps the name `AGENTS.md`, so AI assistants keep finding it. The owner's Python level is now described as **intermediate** (was "very little coding experience"), so explanations skip the basics.
- **Strategy freeze.** Anything that changes which trades are taken or how they are scored needs approval and a `strategy_version` bump, and each version is analyzed separately. Data recorded before Phase 1 is complete is warm-up data and is left out of results (`AGENTS.md` §2.4).
- **The `app.py` split moves earlier.** It now happens before any other code work (new Phase 0 in `TODO.md`), with no change in behaviour, after tests that record today's behaviour. This replaces the 2026-10-04 plan to split only after Phases 1–2.
- **Trade resolution must be precise, not worst case.** When one candle touches both the stop and the target, the tool must find out which came first from tick-level trade data, instead of assuming the stop. If the data can't be obtained, the trade is marked `UNRESOLVED` and left out of results, never guessed. (`TODO.md`, Phase 1 same-candle item.)
- Commits in this project are signed by the **DariSant** GitHub account, not "Obraisan".
- **The old status report is deleted.** `BTC-CLI PROJECT (start_go-btc).md` described the project as it was in March and made claims that are no longer true. The owner deleted it on purpose. It was never in Git, so there is no copy in the history.

### Removed
- `README.md`: the "Project Files" row for the old status report. `TODO.md`: README discrepancy #15 is marked resolved, and the cleanup item notes that part as done (the rest of that item stays open).

### Changed
- **`AGENTS.md` rewritten** from the owner's draft (`NEW_AGENTS.md`), after reviewing the draft against the project's current state. Main changes from the draft:
  - Fixed contradictions with the code and with earlier decisions (spot vs futures, the warm-up phase, append-only data vs how ledgers work, log rotation, `config.toml`, the order of the `app.py` split).
  - Added safety rules: no development runs against real data or the Gemini quota, no spending or server changes, no self-approval, and limits on retries.
  - Rewrote the no-look-ahead rule to use precise tick-level resolution (decision above).
  - Added a memory and self-improvement section (`LESSONS.md`).
- `TODO.md`:
  - New "Owner decisions (2026-10-05)" table.
  - New **Phase 0 – Foundations**: characterization tests, then the package split (moved from Phase 3).
  - New Phase 1 item "Version every record", which the strategy freeze depends on.
  - The same-candle item now follows the precision decision.
- **PRs opened:** #28 (model update, README, changelog and roadmap) was reopened with a full description. #29 (agent instructions) is stacked on it. Merge #28 first, then change #29's base to `main`.
- **Commit author for this project.**
  - This repository now has its own settings: `user.name = DariSant` and `user.email = 250336211+DariSant@users.noreply.github.com`.
  - That address is GitHub's private "noreply" email for the DariSant account. Commits link to the account without publishing a personal email.
  - Other projects on this PC keep the global "Obraisan" identity.
  - To undo: `git config --local --unset user.name` and `git config --local --unset user.email`.
- The earlier commit on branch `chore/model-update-readme-rewrite` (was `d4a4b59`, now `eed40d8`) was re-signed as DariSant and force-pushed. No PR had been opened yet, so nothing else was affected.
- Committed the documentation from 2026-10-04 to the same branch: `CHANGELOG.md`, `TODO.md`, `research/`, and the `README.md` / `AGENTS.md` updates.

### Maintenance
- **GitHub CLI (`gh`) installed** with `winget install GitHub.cli` (version 2.102.0, owner approved on 2026-10-05), so AI assistants can open PRs.
  - Signed in as **DariSant**; the login is stored in the Windows keyring.
  - Git's own login settings were left unchanged.
  - To undo: `gh auth logout`, then `winget uninstall GitHub.cli`.

---

## 2026-10-04

### Decisions (owner)
- **Market type:** the paper account simulates **USDT-M perpetual futures**, so it can go long and short. Fees are modelled at about 0.05% per side plus funding, with leverage capped at 3x.
- **Position size:** risk a **percentage of current equity** (1% to start), so the risked amount grows or shrinks with the account. Results are compared in R multiples.
- **Holding period:** trades last **at most 24 hours**, then are closed at market. Stop floor = the larger of 1 × 15m ATR and 0.5 × 4H ATR.
- **Trade quality:** minimum reward-to-risk **1.5 after fees**, and the target may be at most **3 × 4H ATR** away.
- **Backup AI model:** one ledger per strategy, whichever model ran. Each trade is tagged with `model_used` instead of being written to a separate `output_beta/` ledger.
- **Gemini quota:** `gemini-3.5-flash-lite` allows **500 requests/day** on this project. The limit will live in `config.toml` and be re-checked regularly, along with the model list.
- **Run frequency (proposed):** every 30 minutes, at most 192 AI calls/day.
- **Market activity gate:** do not trade when the market is too quiet. Accepted thresholds:
  - last-hour volume < 0.5 × the normal volume for that hour
  - 15m ATR < 0.15% of price
  - 4H ATR < 0.60% of price
  - fees + slippage > 0.30R

  These thresholds **must be re-checked monthly** (see the re-check log below).
- **Server region (recommended):** Oracle Cloud **Spain Central (Madrid)**, with a backup exchange in case Binance blocks EU addresses in future.
- **Approvals given:** `pytest` as a dev dependency, `pydantic`, replacing `pandas-ta`, a later split of `app.py` into modules, and a `research/` folder.
- **Timing:** Phase 1 (trading-logic fixes) has **not started yet**. The project foundations come first.

### Added
- `TODO.md`: full audit of the project and a phased roadmap (correctness → robustness → maintainability → evaluation → Oracle Cloud deployment), with the owner's decisions above.
- `research/activity_gate_study.py`: re-runnable study that checks the activity gate thresholds against about a year of public BTC futures data. It is free (no API key, no Gemini calls) and takes about 30 seconds: `uv run research/activity_gate_study.py`.
- `research/results/activity_gate_2026-10-04.md`: the first (baseline) report from that study.
- `CHANGELOG.md`: this file.

### Changed
- `TODO.md` updated with the decisions above (futures, percentage risk, 24 h holding, activity gate, quota handling, region choice). The planned daily health command was renamed from `check-models` to `check-setup`, because it will now also remind you about stale activity gate thresholds.
- `README.md`: the "Project Files" table now lists `TODO.md`, `CHANGELOG.md` and `research/`.
- `AGENTS.md`: new section asking every contributor (people and AI assistants) to update this changelog with each change.
- Commit `eed40d8` on branch `chore/model-update-readme-rewrite`, **pushed to GitHub**. PR still to be opened: <https://github.com/DariSant/bitcoin-ai-cli/pull/new/chore/model-update-readme-rewrite>.
  - Primary AI model switched from `gemini-3.1-flash-lite-preview` (shut down by Google on 2026-05-25) to `gemini-3.5-flash-lite`. Both model names now live in the `PRIMARY_MODEL` / `FALLBACK_MODEL` constants in `app.py`.
  - `analyze` now skips Agents 1 and 2 when every requested strategy already has an open trade, to save AI calls.
  - Fixed the doubled border on the execution ticket's title.
  - `README.md` rewritten to describe the current pipeline, commands and data folders.
  - `.gitignore` now also ignores `output_beta/`, `logs/` and `venv/`; the old `venv/` folder was removed from Git.

### Fixed
- **GitHub login for this project.** Pushes were refused (403) because Git on this PC used the saved login of another GitHub account (`Bahia90Studio`) for all of `github.com`.
  - This repository now has its own setting, `credential.https://github.com.username = DariSant` (stored in `.git/config`, so only this project is affected).
  - The DariSant login was saved through the GitHub sign-in window, and the other account's saved login was left untouched.
  - To undo: `git config --local --unset credential.https://github.com.username`.

### Research
- **Activity gate study** (BTC/USDT perpetual on Binance, 2025-09-29 → 2026-10-03, 17,712 decision points every 30 minutes). Results are in `research/results/activity_gate_2026-10-04.md`.
  - A plain "volume below its 20-candle average" rule **does not work** on 15m candles. BTC volume follows a strong daily rhythm, so that average mostly measures the time of day.
  - Comparing the last hour with **the same hour over the previous 20 days** does work.
  - All gate rules together block 22% of decision points (7% on weekdays, 59% on weekends). The blocked moments moved less over the next 24 h (2.2% vs 3.3%) and paid more in fees (0.25R vs 0.16R).
  - The gate saves fees and Gemini calls, but it does not create profit by itself.

### Audit findings (not fixed yet; see `TODO.md`)
- The most important problems found in the code:
  - trade results can be recorded wrongly (the entry candle is ignored, and only the last 25 hours are checked)
  - the backup-model folder can create duplicate or forgotten trades
  - the AI-chosen take-profit has almost no checks
  - indicators use the unfinished last candle
  - fees are not modelled
  - a damaged history file is silently wiped

---

## Activity gate threshold re-check log

Re-check **monthly** with `uv run research/activity_gate_study.py`, and straight away if fees, the exchange, the symbol, or the stop/target/holding rules change. Change a threshold only after **two `REVIEW` results in a row**. After each check, update `thresholds_checked_on` in `config.toml` (once that file exists).

| Date | RVOL < 0.5 | 15m ATR < 0.15% | 4H ATR < 0.60% | Fees > 0.30R | Whole gate blocks | Action | Report |
|---|---|---|---|---|---|---|---|
| 2026-10-04 | KEEP | KEEP | TOO FEW CASES (normal) | KEEP | 22.0% | Baseline; thresholds accepted | `research/results/activity_gate_2026-10-04.md` |

---

## Earlier history (summary of the Git log, before this changelog existed)

- **2026-03-10 → 03-14: Foundations.** Project created with `uv`; the first `app.py` fetched the Bitcoin price, then became a Typer CLI using `ccxt` (Binance) and `google-genai`. Multi-timeframe indicators (EMA, RSI) were added with `pandas` / `pandas-ta`.
- **2026-03-21 → 03-22: Multi-agent AI.** Volume MA and Point of Control were added. A 3-agent chain (Technical, Volume, Portfolio Manager) was built with strict JSON schemas, plus ATR, Value Area and mean-reversion metrics. `rich` panels were added, the CLI was split into `status` / `analyze` / `operate`, and every run was logged as JSON.
- **2026-03-24 → 03-26: Pipeline hardening.** The AI pipeline was split into sequential steps, an "anti-hallucination" protocol and smaller per-agent payloads ("Data Diet") were added, and Agent 4 (the Operator) was introduced.
- **2026-04-01 → 04-12: Paper trading.** A `mock` test harness was added, then the order management system with a paper ledger, and Defensive vs. Greedy A/B testing. Agent 4 was replaced by pure Python math. The interface was overhauled and timestamps were made timezone-aware.
- **2026-04-18 → 04-21: Operator safety and fallback.** The operator gained diagnostic error logging, take-profit (magnet) parsing and a volatility floor for the stop loss. `fetch_and_analyze` was split up, the backup AI model was added with a separate `output_beta/` folder, and the `commands` table was added.
