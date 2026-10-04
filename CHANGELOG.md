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

## 2026-10-05

### Decisions (owner)
- **New agent instructions.** The revised `NEW_AGENTS.md` replaces the old `AGENTS.md`, which is deleted. The file keeps the name `AGENTS.md`, so AI assistants keep finding it. The owner's Python level is now described as **intermediate** (was "very little coding experience"), so explanations skip the basics.
- **Strategy freeze.** Anything that changes which trades are taken or how they are scored needs approval and a `strategy_version` bump, and each version is analyzed separately. Data recorded before Phase 1 is complete is warm-up data and is left out of results (`AGENTS.md` §2.4).
- **The `app.py` split moves earlier.** It now happens before any other code work (new Phase 0 in `TODO.md`), with no change in behaviour, after tests that record today's behaviour. This replaces the 2026-10-04 plan to split only after Phases 1–2.
- **Trade resolution must be precise, not worst case.** When one candle touches both the stop and the target, the tool must find out which came first from tick-level trade data, instead of assuming the stop. If the data can't be obtained, the trade is marked `UNRESOLVED` and left out of results, never guessed. (`TODO.md`, Phase 1 same-candle item.)
- Commits in this project are signed by the **DariSant** GitHub account, not "Obraisan".

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
- **Commit author for this project.**
  - This repository now has its own settings: `user.name = DariSant` and `user.email = 250336211+DariSant@users.noreply.github.com`.
  - That address is GitHub's private "noreply" email for the DariSant account. Commits link to the account without publishing a personal email.
  - Other projects on this PC keep the global "Obraisan" identity.
  - To undo: `git config --local --unset user.name` and `git config --local --unset user.email`.
- The earlier commit on branch `chore/model-update-readme-rewrite` (was `d4a4b59`, now `eed40d8`) was re-signed as DariSant and force-pushed. No PR had been opened yet, so nothing else was affected.
- Committed the documentation from 2026-10-04 to the same branch: `CHANGELOG.md`, `TODO.md`, `research/`, and the `README.md` / `AGENTS.md` updates.

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
