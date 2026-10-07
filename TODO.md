# TODO: Audit and Roadmap

Audit date: 2026-10-04. Audited the **current working tree** (including the uncommitted edits to `app.py` and `README.md`).
Line numbers refer to `app.py` unless another file is named.
**Since 2026-10-07 the code lives in `btc_cli/`.** `app.py:Lnnn` references below point to the pre-split file: view it with `git show 98ee1d7:app.py`. The function names are unchanged or close: `_check_open_positions` → `pipeline.check_open_positions` + `ledger.find_exit`, the operator math → `trade_operator.compute_order`, and the prompts → `agents.build_*_prompt`.

**How findings are labelled**
- **Confirmed** means I saw it in the code, reproduced it with a script, or saw it in the project's own output files.
- **Suspected** means it is likely but not proven. The item explains what would confirm it.
- **P0** = fix before trusting any paper results. **P1** = fix before running unattended on a server. **P2** = improvement.
- **Effort**: S = under an hour, M = a few hours, L = a day or more.

**What I ran (all offline, no Gemini calls)**
- `uv sync --locked`: OK, 58 packages already in sync.
- `uv run --with pytest pytest`: **1 passed**. The only test is for `format_pipe_string`, a text helper. No trading logic is tested.
- `uv run app.py mock mock_long.json` / `mock_short.json`: both print a ticket with SL 68,000 / 72,000, TP 74,000 / 66,000, R:R 2.00, size $3,500.
- `uv run app.py commands`: prints the command table correctly.
- I wrote a throwaway script outside the repo (now deleted). It called the real functions in `app.py` against a fake Binance to test trade resolution, the ledger, the mock math, indicator warm-up and low-priced symbols. Its results are quoted below as "probe".
- I also read the 4 analysis files and `operator_errors.log` already saved in `output_alpha/` from the runs on 2026-10-02.

---

## 0. Status and next steps (reviewed 2026-10-07)

Sections 1–4 below are the original audit of 2026-10-04, kept for its evidence. Their "What I ran" notes describe the code at that time (1 test, everything in `app.py`). This section is the current picture.

### Where we are

| Phase | Done | Open | Notes |
|---|---|---|---|
| 0 Foundations | 2 / 2 | 0 | Characterization tests and the `btc_cli/` split are on `main`. |
| 1 Correctness | 1 / 21 | 20 | Record versioning is on `main`. Everything else changes trades or scoring, so it needs owner approval and `strategy_version` bumps (§2.4). |
| 2 Robustness | 5 / 14 | 9 | Model-name fix, damaged files, atomic writes and the run lock are done (the last three are waiting in PRs). |
| 3 Maintainability | 2 / 9 | 7 | `config.toml` and project-root paths are done (in PRs). The money logic is already partly separated by the split. |
| 4 Evaluation | 0 / 5 | 5 | Nothing yet. |
| 5 Deployment | 0 / 9 | 9 | Starts after Phases 1–3. |

- **Tests:** 137 offline tests, including snapshots of today's behaviour, unit tests for the Operator, ledger, storage and config, a strategy-freeze guard on `config.toml`, and a guard that fails the run if a test touches real data.
- **Waiting for the owner to merge**, in this order: #39 (safe storage; re-lands #35, which merged into a dead branch), #36 (`config.toml`), #37 (paths), #38 (run lock). All four target `main`.
- **No data is being collected yet.** `output_alpha/` holds 7 files from 2026-10-02 (legacy, warm-up). So `strategy_version` bumps are cheap until deployment, but every rule that should apply from day one of collection must be in place by `1.0`.

### Plan

**M1. Robustness groundwork (no trade changes, no version bump; the agent can do these without asking).** About 2 days.
1. ~~**Tick-data study**~~ Done 2026-10-07: feasible at any age. Binance REST covers 48 h, the daily archive covers everything older, and the data is complete. Recommended backup: OKX. See the Phase 1 same-candle item and `research/results/tick_data_2026-10-07.md`.
2. **Total AI failure exits non-zero**, including `ask` (Phase 2, S).
3. **Gemini error handling:** timeout, retries for 429/5xx, no fallback on 400/401/403, both errors logged, and the backup used for the rest of the run once the primary fails (Phase 2, M).
4. **Validate AI replies** with `pydantic` (approved): a bad reply skips only that strategy and is logged (Phase 2, M).
5. **Binance retries** and one shared exchange object with a timeout (Phase 2, S). M2.1 extends this helper.
6. **`mock` uses `compute_order`**, so it tests the real math (Phase 1, S). `mock` affects no trade, so no version bump is needed, but its snapshot changes on purpose.
7. **`operate` picks the analysis by its own UTC timestamp** from the current month only, not by file modification time across every month (Phase 2 P2, S).
8. ~~**`.gitattributes`**~~ Done 2026-10-07 (`chore/gitattributes`).

**M2. Phase 1 strategy batch: warm-up versions 0.2 → 1.0 (each step needs owner approval of its plan, §2.4).** About 6–9 days. Each step is its own PR and bumps `strategy_version` by 0.1.
1. **Market data:** USDT-M perpetual (`binanceusdm`, `BTC/USDT:USDT`) with OKX/Bybit as backup; closed candles only; enough history for EMA 144 and the 20-day RVOL baseline (about 1,000 × 4h and 2,000 × 15m, paginated); a live entry price at operate time with a `STALE_SETUP` re-check; each analysis traded at most once.
2. **Trade resolution:** 1m candles from the entry minute, paginated to now; tick re-check for the entry minute and any minute that touches both levels; `UNRESOLVED` / `UNRESOLVED_DATA_GAP` instead of guesses; 24 h `TIME_EXIT`; `resolution_method` and the first crossing trade recorded. Built with synthetic-data plan steps 1–2 (scenarios with known answers). Also records MAE/MFE (Phase 4 item), because those can't be added to past trades later.
3. **Costs and sizing:** 0.05 % fee, slippage, funding; `pnl_gross` / `fees` / `funding` / `pnl_net` / `r_multiple`; 1 % of current equity per strategy; leverage cap 3×.
4. **Operator rules:** stop floor max(1 × 15m ATR, 0.5 × 4H ATR); target candidates computed in Python for both directions; R:R ≥ 1.5 after fees; target ≤ 3 × 4H ATR; cost ≤ 0.30R; one ledger per strategy, whatever model answered (ends the alpha/beta split).
5. **Agents:** prompts use the real field names; Agent 3 gets the Python-computed stop, target and R:R for both directions as facts; the target is a number chosen from the candidates; the generation settings are decided and recorded (the consistency check costs about 20 Gemini calls: owner go-ahead).
6. **Activity gate:** RVOL < 0.5, 15m ATR < 0.15 %, 4H ATR < 0.60 %, checked before any AI call.
7. **Indicator accuracy:** a volume profile with ATR-sized bins spread over each candle's range; rounding only for display (fixes low-priced symbols).
8. **Record every decision:** a structured record for every verdict and every rejection (reason, model, R:R), replacing the free-text `operator_errors.log` for new runs (owner decision 2026-10-07). Needed before collection so the report's funnel has data from day one.

When all eight are merged, set `strategy_version = "1.0"`: the first official version.

**M3. Ready for unattended running (mostly Phases 2, 3 and 5).** About 3–4 days.
- `[gemini]` quota settings, a daily call budget, and a `check-setup` command (a new command needs approval).
- Rotating `logs/app.log` with UTC times (a log format change needs approval).
- Alerts (Telegram) and a heartbeat (healthchecks.io): the owner creates both accounts; decide whether to keep `requests` for this.
- `deploy/`: systemd service and timers, a server env file, nightly backups, and a rebuild guide. Bounded disk use (a `latest.json` pointer, compressing old months).
- Linux install steps in the README (and remove the personal path), UTC file names (a record-format change).

**M4. Deploy and collect (owner, with exact commands from the agent).** Oracle sign-up in `eu-madrid-1`; the `curl` checks; `uv sync --locked` and `uv run pytest` on the VM (also the first run of the run lock's Linux code path); start the timer.

**M5. Evaluate (Phase 4).** A `report` command (per strategy, per version, per model; R-based; sample-size warning). SQLite only if the report needs it; JSON plus atomic writes is enough until then.

### Decisions needed from the owner

1. **Merge** #39 → #36 → #37 → #38.
2. **Approve the M2 order.** Each step's detailed plan still comes for approval before it is coded.
3. **Review `synthetic_data/PLAN.md`.** Steps 1–2 are needed for M2.2.
4. **Add a non-AI baseline strategy from day one?** Recommended: same Operator and rules, direction from a plain rule or a coin flip, no Gemini calls. It is the only fair way to tell whether the AI adds anything, and it must run over the same period as the AI strategies. This is a §2.4 change, so it would join the M2 batch.
5. **Confirm the run schedule:** every 30 minutes, just after a candle closes. It is still marked "Proposed", and the schedule is under the strategy freeze.
6. **Gemini consistency check** (about 20 calls) before M2.5.
7. **Approvals for M3:** the `check-setup` command, the log format change, removing the unused Gemini key check from `operate`, and keeping or removing `requests`.

---

## 1. Summary

The project is a well-organised prototype: the pipeline runs, the CLI is pleasant, and the operator does the stop-loss and sizing math in Python. **The paper-trading results are not trustworthy yet**, though, because the code that decides whether a trade was a WIN or a LOSS has several holes, and costs are not modelled.

The three most important problems:
1. **Trade resolution can record the wrong result.** It ignores the candle the trade was opened in, it only looks back 25 hours, and the two output folders (alpha/beta) keep separate ledgers. The probe recorded a **WIN for a trade that had already hit its stop**.
2. **The AI picks the take-profit price, with almost no Python guard rails.** The target is a free-text number that is parsed with a fragile string split, it is chosen without knowing the trade direction, and there is no minimum reward-to-risk check. This already happened in your own data: Greedy said GO LONG with a target *below* the price.
3. **Indicators include the unfinished last candle, and fees are not modelled.** Volume looks like "CONTRACTION" on almost every run (4H volume was 11% of average because the candle was 34 minutes old). Fees alone would eat roughly 0.25–0.6 of each trade's risk (0.25R–0.6R).

**Verdict: not ready for unattended deployment.** Do Phases 1–3 first. The server work in Phase 5 is fairly simple once those are done.

### Owner decisions (2026-10-04)

These answers (full text in section 4) are now built into the items below.

| Topic | Decision |
|---|---|
| Market type | **USDT-M perpetual futures** (long and short). Model about 0.05% taker fees, funding, and a leverage cap. |
| Position size | **Percentage of current equity** (1% per trade to start), so risk grows or shrinks with the account. Reports still show R multiples so the A and B strategies stay comparable. |
| Holding period | **Decided: up to 24 hours**, then a time stop closes the trade at market. Stop floor = the larger of 1 × 15m ATR and 0.5 × 4H ATR. |
| Low-activity filter | **Accepted:** skip trading when RVOL < 0.5, 15m ATR < 0.15%, 4H ATR < 0.60%, or fees > 0.30R. **Recheck monthly** with `research/activity_gate_study.py`. See the Phase 1 item "Market activity gate". |
| Research folder | **Approved:** a `research/` folder for study scripts and their dated reports. |
| Change log | **Requested:** `CHANGELOG.md` records every change and decision. |
| Start of Phase 1 | **Not yet.** Settle the project foundations first. |
| Minimum R:R / target distance | **Agreed:** R:R ≥ 1.5 *after fees*, target ≤ 3 × 4H ATR. |
| Alpha/beta folders | **Agreed:** one ledger per strategy, each trade tagged with `model_used`. |
| Gemini quota | **500 requests/day** for `gemini-3.5-flash-lite` (from AI Studio). Store it in `config.toml`, and check models and limits regularly. |
| Run frequency | **Proposed:** every **30 minutes**, just after a candle closes. That is at most 192 calls/day, 38% of the quota. Every 15 minutes would be 384/day (77%), which leaves too little room for retries. |
| Oracle region | **Recommended: Spain Central (Madrid), `eu-madrid-1`.** See the Phase 5 Binance item and the answer to question 7. |
| Approvals | Granted for `uv add --dev pytest`, `uv add pydantic`, replacing `pandas-ta`, and splitting `app.py` later (the timing of the split changed on 2026-10-05, see below). |
| Working-copy edits | Committed as `eed40d8` on branch `chore/model-update-readme-rewrite` and pushed (GitHub login fixed). The PR still needs to be opened and merged. |

### Owner decisions (2026-10-05)

| Topic | Decision |
|---|---|
| Agent instructions | The revised draft replaced the old `AGENTS.md`. |
| Strategy freeze | Anything that changes which trades are taken or how they are scored needs approval and a `strategy_version` bump. Versions are analyzed separately. See `AGENTS.md` §2.4. |
| Warm-up data | Data recorded before Phase 1 is complete is warm-up data, excluded from performance figures. Batch the Phase 1 fixes. |
| Package split | **Moved earlier:** split `app.py` into a `btc_cli/` package *before* other code work, with no change in behaviour. See Phase 0. |
| Trade resolution | **Precision, not worst case:** ambiguous minutes are resolved with tick data, never by assuming the stop. See the Phase 1 same-candle item. |

---

## 2. README vs. code discrepancies

1. **Model fallback** (README L63): says it switches "if that model is down". The code switches on *any* error, including rate limits (429), a bad API key or a bad request (`app.py:L90`). The **committed** code on `main` still uses `gemini-3.1-flash-lite-preview`, which Google shut down on 2026-05-25. Only the uncommitted working copy uses `gemini-3.5-flash-lite`.
2. **"Saves that run's data to `output_beta/`"** (README L63): the code also switches the *ledger* folder for the rest of the process (`L965-L975`, `L1034`, `L1105`, `L1172`). The normal pre-run check never looks at `output_beta/`, so beta trades are never resolved. This also breaks the README's "one open position per symbol" rule (README L74).
3. **"The AI never does the arithmetic"** (README L43): the take-profit is a price the AI writes (`L56`, `L604-L612`). The Greedy prompt asks the AI to "mentally calculate R:R" (`L1150`).
4. **"Fixed risk: $100 (1% of a $10,000 account)"** (README L69): `100.0` is hardcoded (`L670`, `L715`). `account_balance_usdt` and `risk_per_trade_percent` are written into the payload but never used, and the balance never changes after wins or losses.
5. **"`mock` checks the Operator's math"** (README L234-L239): `mock` uses a *different* stop formula with no volatility floor (`L835`, `L847` vs `L630-L636`).
6. **"Checks recent 15m candles"** (README L75): only the last 100 candles (25 hours) are fetched (`L219`), and the candle the trade opened in is skipped (`L238`).
7. **Volume Profile "where 70% of volume traded"** (README L56): it is an approximation. It uses 10 price bins on closing prices only, and the "value area" can be made of separate, non-touching bins (`L396-L425`).
8. **Swing support/resistance "of the last 20 candles"** (README L58): this includes the unfinished current candle (`L393-L394`, `L428`).
9. **"Analysis older than 10 minutes is ignored"** (README L71): the age is measured with the local (non-UTC) clock from the moment the file was written (`L318`, `L573-L577`). The trade still enters at the price from analysis time, not the price at operate time.
10. **"Any Binance symbol can be passed in"** (README L50): every value is rounded to 2 decimals (`L449-L467`). For a $0.12 coin the ATR becomes `0.0` (probe). The `status` header always says "BTC/USDT" (`L491`).
11. **"Friendly error messages"** (README L80): some raw exceptions are still printed (`L221`, `L1327`, `L1330`). When the fallback model also fails, that error is not logged anywhere (`L129-L130`).
12. **Libraries table lists `requests`** (README L102): nothing in the code imports it.
13. **"Where Your Data Is Saved"** (README L205-L222): this omits `output_alpha/status/system/<YYYY-MM>/…_SYSTEM_analysis.json`, which every `status` and `auto` run writes.
14. **`operate` "reads the latest analysis"** (README L154): it also needs `GEMINI_API_KEY` and creates an AI client it never uses (`L535-L540`).
15. **`BTC-CLI PROJECT (start_go-btc).md`** (listed in README L256): it says Agent 3 outputs "Risk Allocation %, Order Types" and that stops use "the AI's ATR multipliers". Neither is true any more. **Resolved 2026-10-05:** the owner deleted the file, and its README row was removed.

---

## 3. Phased roadmap

### Phase 0 – Foundations (before Phase 1; owner decision 2026-10-05)

- [x] **[P1] Characterization tests for today's behaviour** — Confirmed (only 1 test exists) — Effort: M
  - Where: new `tests/` folder; `pyproject.toml` (dev dependency)
  - Problem: the package split must not change behaviour, but nothing checks that. `mock` only covers the order math.
  - Fix:
    - Run `uv add --dev pytest` (approved 2026-10-04).
    - Use a fake exchange with saved candles and a fake Gemini client with canned replies, and write offline tests that capture today's output:
      - the indicator values
      - the exact prompt text sent to each agent
      - the analysis JSON
      - the ticket, ledger and history files
    - Cover a long, a short and a `SIT ON HANDS` case. Write files only to `tmp_path`.
    - Known bugs are captured as they are; they are fixed later on their own branches.
  - Done when: `uv run pytest` passes offline on the current `app.py`, on its own `test/…` branch.
  - Status (2026-10-07): done on branch `test/characterization-tests`: 43 offline tests, all passing. The tests drive the real CLI (`CliRunner`) rather than internal functions, so they can stay unchanged through the split. `test_app.py` already moved to `tests/`.

- [x] **[P1] Split `app.py` into the `btc_cli/` package** — Confirmed — Effort: L
  - Where: `app.py` (1,384 lines); target layout in `AGENTS.md` §10
  - Problem: one big file is getting hard to change safely and to test piece by piece.
  - Fix: after the characterization tests, move the code into `btc_cli/` (`cli`, `config`, `data`, `indicators`, `agents`, `trade_operator`, `ledger`, `storage`, `logging_setup`). Keep `app.py` as a thin entry point so `uv run app.py <command>` still works, (`test_app.py` is already in `tests/`). No behaviour change, so no `strategy_version` bump. (Replaces the old Phase 3 item "Split `app.py` into modules (later)".)
  - Done when: the characterization tests pass unchanged, and `uv run app.py mock mock_long.json` / `mock_short.json` give the same output before and after.
  - Status (2026-10-07): done on branch `refactor/btc-cli-package`. Every characterization test and snapshot passed unchanged; only `conftest.py` changed (it reads the model names from `btc_cli.config`). Both `mock` commands and every `--help` and `commands` output are byte-identical before and after. Two modules were added to the layout with owner approval: `pipeline.py` (the command flows) and `console.py` (the shared console and panels). There are also new unit tests for `trade_operator` and `ledger`, plus a test that checks module boundaries.
  - Note: the fakes are wired in by `tests/conftest.py`, which patches `datetime`, `console` and `BASE_DIR` by name in `app` and any `btc_cli.*` module. If the split renames or restructures these, adjust `conftest.py` only. The test files and snapshots must not change.

### Phase 1 – Correctness (bugs that make paper results wrong or misleading)

- [x] **[P0] Version every record** — Confirmed (no record has version fields) — Effort: M
  - Where: analysis, execution, ledger and history writes (`app.py:L314-L346`, `L741-L786`, `L280-L296`)
  - Problem: the strategy freeze (`AGENTS.md` §2.4) needs every record tagged, so versions can be analyzed separately. Today records have no version, no per-agent model, and closed trades don't link to their analysis.
  - Fix: add `schema_version`, `strategy_version`, UTC timestamp, exchange/market, and `model_used` per agent call to every record. Add the analysis file name to each trade. Readers treat records without these fields as legacy (`schema_version` 0, warm-up). This is a record format change, so present the plan first.
  - Done when: a test shows every newly written record carries the fields, and old records still load.
  - Status (2026-10-07): done on branch `feat/record-versioning`. Analysis, status, ticket, ledger and history records now carry `schema_version` 1 and `strategy_version` "0.1", plus the exchange, the market type, UTC times and `models_used` per agent call. Tickets, ledgers and history entries also carry `trade_id`, `analysis_file` and `analysis_run_id`. The field list is in `README.md`.
  - Not covered here (owner decision 2026-10-07): `operator_errors.log` stays free text. Rejections get a structured record in the Phase 4 item "Record every decision".
  - Every later §2.4 fix must bump `STRATEGY_VERSION` in `btc_cli/config.py` (0.2, 0.3, …; 1.0 once Phase 1 is complete).

- [ ] **[P0] Trade resolution ignores the candle the trade was opened in** — Confirmed — Effort: M
  - Where: `app.py:L235-L239`
  - Problem: candle timestamps are the candle *open* time. `if ts <= entry_timestamp: continue` throws away the whole candle that contains the entry moment. If price hits the stop in the remaining minutes of that candle and later reaches the target, the trade is recorded as a WIN. The probe reproduced this: SL touched inside the entry candle, TP touched in the next one, result = **WIN**.
  - Fix: resolve trades with **1-minute candles** fetched from the entry time, `exchange.fetch_ohlcv(symbol, "1m", since=entry_ms, limit=1000)`. Include the minute that contains the entry. This also shrinks the "same candle" problem below to a one-minute window.
  - Done when: a unit test with a fake exchange where the stop is touched in the entry candle records **LOSS**.

- [ ] **[P0] Trade resolution only looks back 25 hours** — Confirmed — Effort: M
  - Where: `app.py:L219`
  - Problem: `limit=100` on 15m candles covers 25 hours. If the tool is off for longer (laptop asleep, server outage), older candles are never checked. The probe set the SL hit 30 hours ago and a later TP hit, and the result was **WIN**.
  - Fix: fetch with `since=entry_ms` and loop (paginate) until you reach the present. If candles are missing, mark the trade `UNRESOLVED_DATA_GAP` and alert instead of guessing.
  - Done when: a test with a 200-candle gap and an early SL hit records LOSS.

- [ ] **[P0] Two ledgers per strategy (alpha/beta) can mean a second open trade or a forgotten one** — Confirmed — Effort: M
  - Where: `app.py:L32`, `L965-L975`, `L1033-L1034`, `L1104-L1105`, `L1171-L1172`, `L530-L533`, `L198`
  - Problem: `BASE_DIR` is a global that flips to `output_beta` as soon as one agent uses the fallback model, and stays flipped for the rest of the process. In `auto`, `_run_operate` then checks `output_beta/<strategy>/…_paper_ledger.json`, so it does not see an OPEN trade in `output_alpha`, and it opens a second one. On the next normal run (a new process, back to `output_alpha`), the beta trade is never checked, so it stays OPEN forever and is never resolved.
  - Fix: keep **one ledger and one history per strategy**, whatever model was used. Record `"model_used"` on each analysis and trade so alpha and beta results can still be separated in reports. Stop mutating a global; pass the output folder as a function argument.
  - Done when: a test that simulates a fallback mid-run shows the existing open trade blocks a new one, and the next run still resolves it.

- [ ] **[P0] Take-profit is AI free text, chosen without knowing the trade direction, with no reward-to-risk floor** — Confirmed — Effort: M
  - Where: `app.py:L56` (schema field is `str`), `L604-L612` (parsing), `L638-L640`, `L683-L685`, `L1016` (prompt)
  - Problem:
    - Agent 2 picks one "magnet" before Agent 3 decides LONG or SHORT, so the target is often on the wrong side. Real example in `output_alpha/operator_errors.log`: Greedy **GO LONG at 84,452.62 with TP 83,939.32** (below price), rejected.
    - The number is pulled out with `split('|')[0].split(':')[1]`. The probe showed that `"TARGET: $83,939.32"` and `"TARGET: 83939.32 (POC)"` both fail and silently fall back to the 15m POC, which may also be on the wrong side.
    - The only check is "is it on the right side?". A target $5 away (R:R 0.01) or 40% away would be accepted.
  - Fix:
    - Make Python compute candidate targets for **both** directions, for example the nearest POC/VAH/VAL/swing level above and below the price.
    - Have Agent 2/3 choose among those candidates, or return a **number** field per direction (`upside_target: float`, `downside_target: float`).
    - In the operator, reject when `R:R < MIN_RR` (**1.5 after fees, agreed**) or when the target is more than `MAX_TARGET_ATR` × 4H ATR away (**3, agreed**).
    - Log the rejection reason.
  - Done when: unit tests show wrong-side, too-close (R:R < 1.5) and too-far targets are all rejected with a clear reason, and a valid one passes.

- [ ] **[P0] Indicators include the unfinished (still forming) last candle** — Confirmed — Effort: S
  - Where: `app.py:L354`, `L428` (`df.iloc[-1]`), `L393-L394`
  - Problem: Binance returns the current, unfinished candle as the last row. Every indicator, and especially **volume vs. its 20-period average**, uses it, so values change ("repaint") between runs. Your saved runs show this:
    - 4H volume/VMA was **0.108**, because the 4H candle was 34 minutes old.
    - 15m volume/VMA was **0.07**.
    - Agent 2 reported **"CONTRACTION" in all 4 runs**. That bias flows straight into the Defensive rule "volume contracting → SIT ON HANDS" and the Greedy "exhaustion" rule.
  - Fix: drop the last row when `open_time + timeframe > now`, and compute indicators on closed candles only. Use a separate live price (`fetch_ticker`) for the entry. Schedule runs just after each 15m close (see Phase 5).
  - Done when: two runs inside the same 15-minute window produce identical indicator values.

- [ ] **[P0] No trading fees, spread or slippage** — Confirmed — Effort: M
  - Where: `app.py:L266-L278` (PnL), `L669-L671`, `L714-L716`
  - Problem: PnL is gross. With a 1-ATR stop the position is large compared with the risk, so fees matter a lot.
    - Your 2026-10-02 data: price 84,452, 15m ATR 286, stop distance about 708, so the position is about $11.9k. A 0.10% fee on entry and exit costs about $24, which is **0.24R per trade**.
    - When the 1-ATR floor sets the stop (0.34% away), the position is about $29.5k, so fees are about **0.59R**.
    - A strategy that looks slightly profitable now could be losing money once fees are counted.
  - Fix: add `FEE_RATE` (**0.05%**, the USDT-M futures taker rate, since futures were chosen) and `SLIPPAGE_BPS` (e.g. 2–5 basis points) to the config. Apply them on entry and exit. Store `pnl_gross`, `fees`, `funding` (see the next item) and `pnl_net` on every closed trade, and use the net values in the R:R check. With the proposed stop floor, fees come to about 0.17R per trade at today's prices.
  - Done when: a closed trade in history shows gross, fees and net, and a unit test checks the numbers.

- [ ] **[P0] Simulate the futures market you chose: perpetual prices and funding** — Confirmed (decision 2026-10-04) — Effort: M
  - Where: `app.py:L218`, `L477`, `L903` (`ccxt.binance()` = the **spot** market), `L266-L278` (PnL)
  - Problem: you decided to paper-trade USDT-M perpetual futures, but candles, entry prices and trade resolution all come from the spot market. Perpetual prices differ slightly from spot. Every 8 hours, holders of a perpetual position also pay or receive a "funding" fee, and a 24-hour trade crosses up to 3 funding times.
  - Fix:
    - Read candles and the live price from `ccxt.binanceusdm()` with the symbol `BTC/USDT:USDT`. `https://fapi.binance.com` answered HTTP 200 from your connection on 2026-10-04.
    - At close, fetch `fetch_funding_rate_history(symbol, since=entry_ms)` and add up the payments that fall while the trade is open. Longs pay when the rate is positive; shorts receive.
    - Store `funding` on each trade and subtract it in `pnl_net`.
  - Done when: a test with fake funding rates gives the expected funding total for a long and a short, and live candles come from the perpetual market.

- [ ] **[P0] Entry price is from analysis time, entry timestamp is from operate time** — Confirmed — Effort: S
  - Where: `app.py:L593`, `L626`, `L743`, `L775`
  - Problem: `operate` can run up to 10 minutes after `analyze`. It still enters at the old analysis price but starts the clock "now". Any move in between is lost, and price may already be past the stop or target.
  - Fix: at operate time, fetch the live price and use it as the entry. Re-check that the stop and target are still on the correct sides and that R:R is still above the minimum. Otherwise reject with reason `STALE_SETUP`.
  - Done when: a test where the live price has already passed the target rejects the trade.

- [ ] **[P1] Position size can mean leverage of about 3x, with no cap** — Confirmed — Effort: S
  - Where: `app.py:L670-L671`, `L715-L716`
  - Problem: size = $100 ÷ stop distance. With a 0.34% stop that is about $29.5k on a $10k account (about 3x), with nothing limiting it. Futures allow this (you chose futures), but the leverage should be a deliberate setting, not an accident of how tight the stop is.
  - Fix: add `MAX_LEVERAGE` to the config (suggested **3.0**) and cap the notional at `equity × MAX_LEVERAGE`. If the cap applies, the real risk is smaller than 1%; record that. Store `leverage_used` on the trade. State in the README that the tool simulates USDT-M perpetual futures. With the proposed stop floor (0.5 × 4H ATR, about 0.6% today), normal trades use about 1.7x, so the cap should rarely apply.
  - Done when: a test with a tiny stop distance shows the size capped and the reduced risk recorded.

- [ ] **[P1] Account balance is fixed and the risk fields are ignored** — Confirmed — Effort: S
  - Where: `app.py:L616-L617`, `L670`, `L715`, `L843`, `L855`
  - Problem: the account balance is always `10000.0` and the risk always `100.0`. Realised PnL never changes the balance, so drawdown and equity curves cannot be computed. The `account_balance_usdt` and `risk_per_trade_percent` inputs exist but do nothing.
  - Fix (**decided: percentage of equity**):
    - Compute `equity = STARTING_BALANCE + sum(pnl_net in history)` separately for each strategy (each starts with its own $10,000).
    - Then `risk_usd = equity × RISK_PCT / 100`, with `RISK_PCT = 1.0` in the config.
    - Store `equity_before`, `risk_usd` and `r_multiple = pnl_net / risk_usd` on each trade. Because the two strategies' balances will drift apart, compare them in the report by R multiples, not dollars.
  - Done when: after a recorded loss, the next ticket's `risk_usd` is 1% of the new, lower equity, and a test checks this.

- [ ] **[P1] The `mock` command tests different math from the real operator** — Confirmed — Effort: S
  - Where: `app.py:L834-L856` vs `L628-L716`
  - Problem: `mock` has no volatility floor. With price 70,000, ATR 1,000 and threat 69,900, the probe got:
    - real operator: SL **69,000**, size **$7,000**
    - mock: SL **69,400**, size **$11,667**
    - The two mock files that ship with the project do not show this, because there the floor is not the deciding value.
  - Fix: move the order math into one pure function, `compute_order(verdict, price, atr, threat, target, equity, risk_pct, ...)`, that returns either a ticket or a rejection reason. Call it from both `operate` and `mock`. Add a `mock_json/mock_floor.json` case.
  - Done when: `mock` and `operate` give the same ticket for the same inputs, checked by a test.

- [ ] **[P1] The same analysis can be traded twice** — Confirmed — Effort: S
  - Where: `app.py:L549-L579`
  - Problem: nothing marks an analysis file as "used". If a trade opens and closes within 10 minutes (possible with a 1-ATR stop) and `operate` runs again, it opens a second trade from the same analysis at the old price.
  - Fix: store `last_consumed_analysis` (the file name) in a small per-strategy state file, or write `trade_id` back into the analysis. Skip analyses that were already used.
  - Done when: running `operate` twice on one analysis creates at most one trade.

- [ ] **[P1] Same-candle stop and target hit is always scored as a LOSS** — Confirmed — Effort: M (after the 1m change above)
  - Where: `app.py:L244-L261`
  - Problem: when a candle's high reaches the target and its low reaches the stop, the stop is checked first, so the result is LOSS (probe confirmed). The rule is hidden, and on 15m candles with 1-ATR stops it can happen often.
  - Decision (owner, 2026-10-05): **precision, not worst case.** The pessimistic rule is dropped.
  - Fix:
    - When a 1-minute candle touches both levels, fetch the exchange's individual trades for that minute (Binance aggregated trades through `ccxt` `fetch_trades`). The first trade at or beyond a level decides the result.
    - Do the same for the entry minute, counting only trades after the entry time.
    - If tick data cannot be fetched, retry on later runs. After 24 hours, close the trade as `UNRESOLVED`, alert, and exclude it from performance figures.
    - Record `resolution_method` (`1m` / `tick` / `unresolved`) and the time and price of the first crossing trade. Count each method in the report.
    - First check that tick data is available, and how far back it goes, for the chosen exchange (Binance USDT-M first, then the backup exchange).
  - Tick data checked (2026-10-07, `research/tick_data_study.py`, report `research/results/tick_data_2026-10-07.md`): **feasible at any age, with complete data.**
    - Binance USDT-M REST `aggTrades` covers the last 48 h; the daily archive `data.binance.vision` covers older days (online by about 07:10 UTC the next day), so there is never a gap.
    - A day's trades add up to the candle volume exactly.
    - Binance's 1m candles count a boundary trade in the next minute about 7 % of the time, and in 4 of 1,440 minutes that moved the high or low by one tick. So trade timestamps decide, a tick check covers the neighbouring minutes' edges, and a level within one tick of a candle's range counts as touched. Details are in the report's "Interpretation".
    - Backup exchanges: OKX REST goes back about 3 months but needs the raw endpoint, paged by trade id (ccxt's `fetch_trades` ignores `since`). Bybit has archive files only, and its candles open at the previous close. Recommended backup: **OKX**.
  - Done when: tests with fake tick data resolve both orders correctly (stop first → LOSS, target first → WIN), and the no-tick-data case ends as `UNRESOLVED`, never WIN or LOSS.

- [ ] **[P1] EMA 144 has too little history to settle** — Confirmed (simulated) — Effort: S
  - Where: `app.py:L354` (`limit=200`), `L377`
  - Problem: an EMA 144 computed on only 200 candles still remembers its starting value. Over 200 simulated price paths, the last EMA 144 from 200 candles differed from one computed on 1,000 candles by a median of **0.20%** (90th percentile 0.42%). That is the same size as one 15m ATR, so values drift from what a charting site would show. RSI 47 is fine (about 0.06 points).
  - Fix: fetch 1,000 candles (Binance allows up to 1,000 per request) and send only the latest values to the AI.
  - Done when: EMA 144 matches a 1,000+ candle reference within 0.01%.

- [ ] **[P1] Volume Profile is too coarse to use as a price target** — Confirmed — Effort: M
  - Where: `app.py:L396-L425`
  - Problem:
    - It uses 10 bins over the price range, putting each candle's whole volume at its *close*.
    - On 4H data (200 candles, about 33 days) your saved value area spans **76,750–86,704 (12%)**, and the POC is a bin midpoint (83,939.32), not a real traded price.
    - The 70% value area is made of the biggest bins, wherever they are. A standard value area grows outward from the POC without gaps.
    - Because the take-profit often *is* this POC, the target's precision is about ±1 bin width.
  - Fix: use bins about 0.25–0.5 × ATR wide, spread each candle's volume across its high–low range, and grow the value area outward from the POC. Explain in the README that this is an approximation from OHLC candles, not true tick-by-tick data.
  - Done when: a unit test with known synthetic volume puts POC/VAH/VAL at the expected levels, and the README describes the approximation.

- [ ] **[P1] Prompt and data mismatches, and the AI is asked to do R:R math** — Confirmed — Effort: S
  - Where: `app.py:L1008` vs `L925`, `L1150`, `L1089-L1090`
  - Problem:
    - Agent 2's prompt describes fields called `price_to_va_status` and `distance_to_poc_percent`, but the data it receives calls them `va_status` and `dist_poc_percent`.
    - Greedy is told to "mentally calculate R:R" using a "nearest structural threat". It never sees the stop Python will actually use (support − 0.5 ATR, floored at 1 ATR), so its "ASYMMETRY: FAVORABLE" judgement has nothing behind it. In your data, Greedy said FAVORABLE for a long whose target was below the price.
  - Fix: use the real key names in the prompts. Compute stop, target and R:R for both LONG and SHORT in Python, and pass those numbers to Agent 3 as facts, so the AI judges a setup instead of doing arithmetic.
  - Done when: the prompts reference only keys that exist in the data, and Agent 3's input includes the Python-computed R:R for both directions.

- [ ] **[P1] Low-priced symbols break the math** — Confirmed — Effort: S
  - Where: `app.py:L449-L467`
  - Problem: values are rounded to 2 decimals *before* the stop and size math. For a $0.12 coin the probe got `atr_14 = 0.0` and `support = 0.12 = price`, which makes the stop logic meaningless.
  - Fix: round only when printing, never before calculations. Until other symbols are tested, either limit the CLI to BTC/USDT or say so in the README.
  - Done when: a test on a 0.12-priced series gives a non-zero ATR and a valid ticket.

- [ ] **[P1] Stop and target come from different timeframes, and trades have no time limit** — Confirmed — Effort: M
  - Where: `app.py:L594`, `L596-L597`, `L630-L636`, `L675-L681` (15m ATR and 15m swings for the stop) vs Agent 2's target, which in your data is the **4H** POC
  - Problem: the stop is sized for about 5 hours of 15m movement (20 candles, 1 × 15m ATR), but the target can be a 4H level days away. With a target cap of 3 × 4H ATR (agreed), a 1 × 15m ATR stop could give R:R up to about 10. Normal noise would then hit the stop long before the target. A trade can also stay open forever.
  - Fix (**holding period decided 2026-10-04: 24 hours**):
    - Holding period: up to **24 hours**. After `MAX_HOLD_HOURS = 24`, close the trade at market and record `result = "TIME_EXIT"`.
    - Stop: keep the 15m structure (`swing − 0.5 × 15m ATR`), but make the floor the larger of `1 × 15m ATR` and `0.5 × 4H ATR`. At today's values that is about 495 instead of 286, roughly 0.6% of price.
    - Target: between `MIN_RR` × stop distance and `3 × 4H ATR`.
    - Put all three numbers in the config.
    - Evidence that this combination works over 24 h: in the 13-month simulation for the gate item below, these rules gave a median stop of 0.84% of price and a 1.5R target of about 1.26%. Only 14% of trades reached neither stop nor target within 24 h.
  - Done when: `compute_order` applies the new floor, `resolve_trade` closes trades older than 24 h at the market price, the README describes the rule, and tests cover both.

- [ ] **[P1] Market activity gate: skip trading when volume or volatility is too low** — Confirmed (owner request + data study) — Effort: M
  - Where: new function `market_activity_gate(...)`, called in `_run_analyze` **before** any Gemini call (`app.py:L895`), plus a cost check inside `compute_order`
  - Problem: when the market is quiet, price moves less, stops are tighter, and fees take a bigger share of each trade. Today nothing stops the tool from trading in those conditions.
  - Study (13 months of Binance BTC/USDT perpetual candles, 2025-08-30 → 2026-10-04; 17,661 decision points every 30 minutes):
    - **Method.** At every point I applied this plan's operator rules: stop floor = max(1 × 15m ATR, 0.5 × 4H ATR), 1.5R target, 24 h time stop, 0.05% fee + 0.02% slippage per side. I tried both a long and a short (a random direction), then grouped the results by volume and volatility. Random direction earns about 0R before costs in every group, so the differences between groups are pure costs and missed moves. That is exactly what a filter can save.
    - **The "volume vs. 20-candle average with an offset" idea does not work on 15m candles.**
      - BTC volume follows a strong daily cycle. In UTC, the last hour's volume compared with the previous 20 hours is about 0.5× between 22:00 and 01:00, and 2.0× at 15:00 (US market open).
      - A 20-candle average covers only 5 hours, so it mostly measures *time of day*, and a single 15m candle is noisy.
      - Blocking when `volume < 0.5 × VMA20` would block 26% of the time, yet the blocked moments were slightly *better* than average (−0.16R vs −0.19R).
    - **What works: compare the last hour with the same hour on previous days** ("seasonal relative volume", RVOL):
      - `RVOL = volume of the last 4 closed 15m candles ÷ median volume of the same clock hour over the previous 20 days`.
      - This keeps your "offset" idea: the threshold allows volume to be up to 50% below normal for that hour.
      - With RVOL < 0.5, the next 24 hours moved only 2.27% (median) vs 3.22% otherwise, and costs were 0.24R vs 0.16R.
      - Low RVOL also predicts *less movement than ATR alone suggests* (24 h range = 1.9 × 4H ATR vs 2.35× normally), so it adds information the ATR rule doesn't have.
      - The result was the same in both halves of the year (−0.25R blocked vs −0.13R / −0.21R allowed).
    - **Minimum ATR.**
      - When the 15m ATR was below **0.15% of price** (12% of the time), the next 24 hours moved only 1.83% and fees took 0.29R per trade.
      - Below 0.10%, it was 1.3% of movement and 0.31R in fees.
      - 4H ATR below 0.6% of price is rare (0.9%), but costs there reach 0.39R; keep it only as a safety net.
    - **Direct cost check.** Once Python knows the stop, it can calculate the fees exactly. Rejecting tickets where fees + slippage exceed **0.30R** (a stop closer than about 0.47% of price) removes the worst 6% (−0.35R vs −0.17R).
    - **All rules together** block **22%** of decision points: 7% on weekdays, 59% on weekends.
      - Blocked moments: 24 h range 2.21% vs 3.29%, costs 0.25R vs 0.16R, result −0.26R vs −0.16R per trade.
      - Weekends really are quieter (24 h range 2.59% vs 3.25%), and the weekend moments the gate lets through look like normal weekdays (range 3.20%).
      - The gate also saves about 22% of Gemini calls, because it runs before the AI.
  - Fix (rules **accepted by the owner on 2026-10-04**, all on **closed** candles):
    1. **Low volume (before AI):** skip if `RVOL < 0.5`.
    2. **Low volatility (before AI):** skip if 15m ATR < 0.15% of price, or 4H ATR < 0.60% of price.
    3. **Costs too high (in `compute_order`):** reject the ticket if `(FEE + SLIPPAGE) × 2 × entry ÷ stop_distance > 0.30`.
    4. Save the measured values and the reason (`LOW_VOLUME`, `LOW_VOLATILITY`, `COSTS_TOO_HIGH`) in each run's record, so the Phase 4 report can count them.
    5. Fetch at least 21 days of 15m candles (about 2,000, i.e. 2 requests) for the 20-day baseline. This also covers the EMA warm-up item.
    6. Config:
       ```toml
       [activity_gate]
       enabled = true
       rvol_min = 0.5               # last hour ÷ median of the same hour over the last 20 days
       rvol_lookback_days = 20
       atr15_min_pct = 0.15         # 15m ATR as % of price
       atr4h_min_pct = 0.60         # 4H ATR as % of price (safety net)
       max_cost_r = 0.30            # fees + slippage as a share of the risk (checked by the operator)
       # These thresholds MUST be re-checked periodically: run research/activity_gate_study.py monthly.
       thresholds_checked_on = "2026-10-04"
       recheck_every_days = 30
       ```
    7. The daily `check-setup` command (Phase 2 quota item) warns when `thresholds_checked_on` is older than `recheck_every_days`.
  - **Re-check schedule (thresholds MUST be re-checked periodically to keep the strategy well tuned):**
    - **Monthly:** run `uv run research/activity_gate_study.py`. It takes about 30 seconds, is free, and makes no Gemini calls. It saves a dated report in `research/results/`. Then update `thresholds_checked_on` in `config.toml` and add one line to `CHANGELOG.md`.
    - **Change a threshold only when** the current value shows `REVIEW` in **two monthly reports in a row**, and a neighbouring value shows `KEEP` in both halves of the year. Monthly data is noisy; changing values after a single report would chase noise (overfitting).
    - **Re-check straight away** (don't wait for the month) if the fee rate, exchange, symbol, stop/target rules or the 24 h limit change.
    - **Every 3 months, once each strategy has 30+ closed trades:** compare the thresholds with *real* paper trades (Phase 4 report: results of trades taken just above each threshold, plus MAE/MFE).
    - First baseline report: `research/results/activity_gate_2026-10-04.md`. All four rules show `KEEP`, and the 4H safety net shows `TOO FEW CASES` because it rarely fires, which is expected.
  - Limits of the study:
    - The gate does not create an edge by itself; it avoids paying more for less movement. The AI still has to pick the right direction.
    - The thresholds come from one year of BTC only, which is why the monthly re-check above is required.
    - For other symbols, percent-of-price thresholds may not fit; use each symbol's own 30-day percentiles instead.
  - Done when:
    - unit tests with synthetic candles trigger each of the three reasons and let a normal case through
    - a gated run makes **no** Gemini calls and records the reason
    - the README explains the gate and the monthly re-check in plain words

- [ ] **[P1] Synthetic market data for testing the rules** — Planned (owner request 2026-10-07; questions answered, plan awaiting review) — Effort: M–L
  - Where: new folder `synthetic_data/` (plan: [`synthetic_data/PLAN.md`](synthetic_data/PLAN.md), overview: [`synthetic_data/README.md`](synthetic_data/README.md))
  - Problem: the Phase 1 resolution, cost and gate fixes need price paths with known correct answers, and there is no way to check the rules for hidden bugs or look-ahead without spending AI calls.
  - Fix:
    - Reproducible, Bitcoin-like 1m candles (15m/4h built from them), ticks only where needed, and funding.
    - Two generators: first a block bootstrap of 3 years of public BTC USDT-M perpetual history (`binanceusdm`, with funding), then a calibrated regime-switching model.
    - Generated datasets go in `synthetic_data/output/`, ignored by Git.
    - Hand-made scenarios with expected outcomes for `tests/`.
    - A no-edge walk-forward check in `research/` with a rule-based stand-in verdict and no Gemini calls.
    - Never written to `output_alpha/`, and never counted as performance.
  - Order: steps 1–2 of the plan (format, builders, resolution scenarios) go with the 1m/tick resolution items above. Steps 3–6 (calibration, generators, no-edge check) can follow.
  - Done when: the owner has approved the plan, the resolution items above are tested against `synthetic_data/scenarios/`, and a 3-year no-edge run reports an average R consistent with −costs.

### Phase 2 – Robustness (errors, retries, state safety, tests)

- [x] **[P0] Commit the model-name fix that is sitting uncommitted** — Confirmed — Effort: S
  - Where: `git diff app.py` (adds `PRIMARY_MODEL = "gemini-3.5-flash-lite"` at `L37`)
  - Problem: `main` still calls `gemini-3.1-flash-lite-preview`. Google's deprecations page lists it as shut down on 2026-05-25. On `main`, every agent call fails over to the backup model, so every run lands in `output_beta/`. `gemini-3.5-flash-lite` is listed as a stable model (released 2026-07-21).
  - Fix: review and commit the working-copy changes (model constants, README, the "skip analysis when both positions are open" guard). Also decide what to do with the staged `venv/` deletions.
  - Status (2026-10-04): committed as `eed40d8` on branch `chore/model-update-readme-rewrite` and **pushed**. This includes `.gitignore`, `README.md`, `app.py` and the removal of the old `venv/` folder.
    - The GitHub login problem is fixed: this repo now signs in as `DariSant` (repo-only setting; see `CHANGELOG.md` → Fixed).
    - Remaining step: open the PR at <https://github.com/DariSant/bitcoin-ai-cli/pull/new/chore/model-update-readme-rewrite> and merge it.
  - Done when: the PR is merged and `main` names a live model.
  - Status (2026-10-07): done. PR #28 is merged, and `main` uses `gemini-3.5-flash-lite` with `gemini-2.5-flash` as fallback (`app.py:L37-L38`).

- [x] **[P0] A damaged history file is silently replaced, losing all closed trades** — Confirmed — Effort: S
  - Where: `app.py:L286-L296` (bare `except: pass`)
  - Problem: if `*_trade_history.json` cannot be read (half-written, hand-edited), the code starts a new empty list and **overwrites the file** with just the newest trade. The probe started with 2 old trades in a damaged file and ended with 1 entry.
  - Fix: if the file exists but cannot be read, leave it exactly where it is (no rename: `AGENTS.md` §2.3 forbids renaming or moving recorded files), log an error, raise an alert, and stop. Never replace history automatically.
  - Done when: a test with a damaged history file keeps the original bytes and the run exits with an error.
  - Status (2026-10-07): done on branch `fix/safe-storage`. The fix originally proposed here (rename to `…corrupt-<time>.json`) conflicted with §2.3, so the file is left in place instead. Nothing is written, the trade stays OPEN, the strategy is blocked, the other strategy still runs, and the command exits 1. Alerts come with Phase 5.

- [x] **[P0] A damaged ledger file is treated as "no open trade"** — Confirmed — Effort: S
  - Where: `app.py:L204-L208`
  - Problem: a `JSONDecodeError` returns `False` ("no position"). The next ticket then overwrites the ledger, and the open trade is lost (probe: `still_open = False`).
  - Fix: same as above: leave the file untouched, log the error, alert, and block new trades for that strategy until it is fixed.
  - Done when: a test with a damaged ledger blocks new trades.
  - Status (2026-10-07): done on branch `fix/safe-storage`. Invalid JSON, an empty file and a non-object all count as damaged. The command exits 1.

- [x] **[P1] File writes are not crash-safe, and a crash can duplicate a trade in history** — Confirmed — Effort: S
  - Where: `app.py:L294-L299`, `L343-L344`, `L762-L763`, `L785-L786`
  - Problem: files are written in place. If the process dies mid-write, the file is half-written. If it dies after appending to history but before deleting the ledger (`L299`), the next run resolves the same trade again and appends it a second time.
  - Fix: add a small `write_json_atomic(path, data)` helper: write to `path + ".tmp"`, `flush`/`fsync`, then `os.replace`. Give every trade a `trade_id` (e.g. UTC time + strategy) and skip appending one whose `trade_id` is already in history.
  - Done when: a test that simulates a crash between the two steps produces exactly one history entry.
  - Status (2026-10-07): done on branch `fix/safe-storage`. `storage.write_json_atomic` is used for analyses, tickets, ledgers and history, and the bytes written are identical to before. The temp file is `.<name>.<pid>.tmp` in the same folder. History de-duplicates on `trade_id`, or on symbol/entry time/verdict/entry price for legacy trades. The two append-only logs are unchanged.

- [x] **[P1] Nothing stops two runs overlapping** — Confirmed (no lock exists) — Effort: S
  - Where: whole `auto` / `operate` flow
  - Problem: a scheduled run plus a manual run, or a slow run overlapping the next scheduled one, can both read "no open trade" and both write a ledger.
  - Fix: create a lock file at start with `os.open(path, os.O_CREAT | os.O_EXCL)`. This uses only the standard library, needs no new package, and works on Windows and Linux. Store the process ID (PID) in it and treat the lock as stale if that process is gone. Release it in a `finally:` block.
  - Status (2026-10-07): done on branch `fix/run-lock` with an OS file lock (`storage.run_lock`) on `run.lock` in the data folder, instead of `O_EXCL` plus a PID check.
    - `status`, `analyze`, `operate` and `auto` hold it. `auto` holds one lock across all three steps, so nothing can slip in between them.
    - A second run prints who holds the lock and exits 3 without doing anything.
    - The OS releases the lock when the holder ends, even on a crash; a test kills a holder process to prove it. The file is never deleted.
    - Not yet exercised: the Linux `fcntl.flock` branch. These tests ran on Windows only (`msvcrt.locking`); `uv run pytest` on the VM (Phase 5 ARM64 item) covers it.
  - Caution (found 2026-10-07): don't check "is that PID alive?" with `os.kill(pid, 0)`. On Windows, signal 0 is `CTRL_C_EVENT`, so the check would interrupt a process. Prefer an OS file lock (`fcntl.flock` on Linux, `msvcrt.locking` on Windows), held on an open lock file: the OS releases it when the process dies, so stale locks can't happen and the lock file is never deleted (`AGENTS.md` §5).
  - Done when: starting a second `auto` while one is running prints "another run is in progress" and exits.

- [ ] **[P1] The AI fallback treats every error the same, with no retries or timeout** — Confirmed — Effort: M
  - Where: `app.py:L83-L130`
  - Problem:
    - Any exception switches to the backup model, including a bad API key (which will also fail on the backup) and a 429 rate limit (where waiting is better).
    - The backup model's error is thrown away (`except Exception as fallback_e: return None, None`), so you never learn why both failed.
    - No timeout is set, so a hung request can block a scheduled run.
    - Each agent tries the dead primary again, so one outage costs 4 failed calls and 4 extra entries in the health log.
  - Fix: catch `google.genai.errors.ClientError` and `ServerError` and check the status code:
    - 400/401/403: stop and alert; do not fall back.
    - 429: wait (respect the retry delay if given), retry once, then fall back.
    - 5xx or timeout: retry twice with a growing delay (backoff), then fall back.
    - Log the backup model's exception.
    - Set a request timeout via `http_options`.
    - Once the primary fails in a run, use the backup for the remaining agents.
  - Done when: tests with a fake client cover 401 (no fallback, clear message), 429 then success (retry, no fallback), and 500 ×3 (fallback, both errors logged).

- [ ] **[P1] Total AI failure exits with "success" (exit code 0)** — Confirmed — Effort: S
  - Where: `app.py:L970-L972`, `L1029-L1031`, `L1096-L1102`, `L1167-L1169`
  - Problem: "Both models unreachable" prints a message and `return`s, so the process exit code is 0. A scheduler or monitor will think the run succeeded.
  - Fix: `raise typer.Exit(code=2)` (or another non-zero code) after printing.
  - Also (found 2026-10-07): `ask` exits 0 for the same failure, because its broad `except Exception` catches its own `typer.Exit` (Click's `Exit` is a `RuntimeError`) and prints "An unexpected error occurred: ". Pinned by `test_ask_with_both_models_down_exits_0`.
  - Done when: a test with both models failing gets a non-zero exit code.

- [ ] **[P1] AI responses are not validated** — Confirmed — Effort: M
  - Where: `app.py:L978`, `L1037`, `L1108`, `L1175`, and `.get(…, 'NEUTRAL' / 'SIT ON HANDS')` defaults
  - Problem:
    - Only `json.loads` is checked. Missing keys or unexpected values become NEUTRAL / SIT ON HANDS without any record.
    - A JSON error in Agent 3 Defensive calls `typer.Exit` (`L1112`), so Greedy never runs that cycle. The two A/B arms then get different amounts of data.
  - Fix: switch the `TypedDict` schemas to `pydantic` models and use `response.parsed`. Run `uv add pydantic` (**approved 2026-10-04**; it is already installed as a dependency of `google-genai`). On failure, log the raw text, mark *that strategy* as skipped, and continue with the other one.
  - Done when: a fake response missing `final_verdict` is logged and skipped without stopping the other strategy.

- [ ] **[P1] Store the Gemini quota in config, stay under it, and check models and limits regularly** — Confirmed (owner reports 500 RPD) — Effort: M
  - Where: `app.py:L37-L38`, call pattern in `L967-L1165`; new `config.toml`; new `check-setup` command
  - Problem:
    - A full `auto` run makes **4 AI calls** (Agents 1, 2, 3-Defensive, 3-Greedy). With your limit of **500 requests/day** for `gemini-3.5-flash-lite`:
      - every 15 minutes = 384/day (77%)
      - every 30 minutes = 192/day (38%)
    - Nothing in the code knows the limit, so a busy day just fails with 429 errors and falls back.
    - Google limits are per project, and the daily count resets at **midnight Pacific time** (09:00 Spanish time, or 08:00 during the few weeks each year when the US and EU clock changes don't line up).
    - The Gemini API cannot report your quota numbers; Google shows them only in AI Studio. A program can, however, list which models your key can use.
    - Google's deprecations page says `gemini-2.5-flash` (the current backup) is "restricted to users with prior active usage", so it may stop working for you without warning.
  - Fix:
    1. **Put the numbers in `config.toml`, not `.env`.** `.env` is for secrets; quotas are normal settings you want to see, compare and track in Git. Suggested section:
       ```toml
       [gemini]
       primary_model = "gemini-3.5-flash-lite"
       fallback_model = "gemini-2.5-flash"   # replace after `check-setup` confirms a better one
       daily_request_limit = 500             # from AI Studio → Rate limits
       requests_per_minute_limit = 0         # fill in from AI Studio (0 = unknown)
       safety_margin_percent = 10            # stop new analyses at 90% of the daily limit
       limits_checked_on = "2026-10-04"      # update every time you look in AI Studio
       recheck_every_days = 30
       ```
    2. **Count calls.** Keep a small file with calls per Pacific-time day (or a table, once SQLite is in). Before Agents 1 and 2 run, skip the cycle if the 4 calls would cross `daily_request_limit × (1 − safety_margin)`. After a daily-quota 429, stop calling until the reset instead of falling back on every agent.
    3. **Add a `check-setup` command**, run by hand and once a day by the scheduler (Phase 5). It should:
       - list the models your key can use (`client.models.list()`, the same call as `list_models.py`)
       - fail loudly if `primary_model` or `fallback_model` is no longer listed, so a shut-down model is caught before it breaks runs
       - print newer Flash / Flash-Lite models than the ones configured, as upgrade candidates
       - warn if `limits_checked_on` is older than `recheck_every_days`, as a reminder to look in AI Studio and update the numbers
       - warn if `[activity_gate] thresholds_checked_on` is older than its `recheck_every_days` (monthly gate re-check, see Phase 1)
       - send its warnings through the alert channel (Phase 5)
    4. **Schedule every 30 minutes** (see Owner decisions). Keep the existing "skip AI when both strategies have open trades" guard.
    5. Delete `list_models.py` once `check-setup` replaces it.
  - Confirm before relying on it: whether `models.list` counts toward the daily request quota (expected: it does not). Check the AI Studio usage page after running it a few times.
  - Done when: limits live only in `config.toml`; a test shows the budget guard skipping a cycle at 90%; `check-setup` reports missing models, newer candidates and an out-of-date `limits_checked_on`.

- [ ] **[P1] AI answers may vary a lot on identical data** — Suspected — Effort: S (setting) / M (measurement)
  - Where: `app.py:L79-L81` (no `temperature` or other generation settings)
  - Problem: no generation settings are set, so the model's default randomness applies. Your two Greedy runs about an hour apart, on almost identical data, returned GO LONG and SIT ON HANDS with the same risk vector. That is only one example, not proof.
  - Confirm by: sending one saved analysis payload to Agent 3 five times (20 calls) and counting how often the verdicts agree.
  - Fix: save the model name and generation settings into every analysis file. Check Google's current advice for the 3.x models before lowering `temperature`; for Gemini 3 models Google has recommended keeping the default. If verdicts flip often, consider asking 3 times and taking the majority verdict (this costs 3× the calls).
  - Done when: the consistency check is recorded and the setting choice is written in the code comments.

- [ ] **[P1] Binance calls have no retry, and one failed check stops the whole run** — Confirmed — Effort: S
  - Where: `app.py:L217-L222`, `L348-L360`, `L477`, `L903`
  - Problem:
    - One timeout in `_check_open_positions` calls `typer.Exit(1)` and prints the raw exception text.
    - A new `ccxt.binance()` object is created for every fetch.
    - There is no retry on `ccxt.NetworkError`, `RequestTimeout` or `DDoSProtection` (Binance's rate-limit response).
  - Fix: create one shared exchange object with `{"timeout": 15000, "enableRateLimit": True}`. Add a `fetch_with_retry` helper (3 tries, waiting 2s, 4s, 8s) for those errors. Print a short friendly message and log the full details.
  - Done when: a fake exchange that fails twice and then succeeds completes the run.

- [ ] **[P1] Test suite covers almost nothing** — Confirmed — Effort: L
  - Where: `test_app.py:L1-L9` (only `format_pipe_string`); `pyproject.toml` has no dev dependencies
  - Problem: none of the code that decides money outcomes is tested.
  - Fix: `uv add --dev pytest` (**approved 2026-10-04**). Then add offline tests using a fake exchange object and a fake Gemini client (`monkeypatch`) for:
    - `compute_order`: long, short, floor active, wrong side, R:R too low, target too far, leverage cap
    - trade resolution: SL, TP, entry-candle hit, same-candle hit, gap longer than 25h, fees in PnL
    - the 10-minute staleness check, using UTC times
    - damaged ledger and damaged history files
    - atomic write and crash-in-the-middle cases
    - fallback routing: 401, 429, 500, and the case where an open trade blocks a new one
    - magnet / target validation
    - closed-candle filtering
    - indicators on fewer than 144 candles (this already gives a friendly error, confirmed by the probe) and on low-priced symbols
  - Done when: `uv run pytest` runs offline with at least the cases above and all pass.
  - Progress (2026-10-07, later): 137 offline tests. Added since: unit tests for `trade_operator`, `ledger`, `storage` and `config`; atomic-write and crash cases; damaged ledger and history; the run lock (including a crashed holder); a strategy-freeze guard on `config.toml`; and a session guard that fails if real data changes. Still open: the cases that only exist once the Phase 1 fixes land (magnet validation, closed candles, R:R floor, leverage cap). Each fix brings its own tests, so this item closes with Phase 1.
  - Progress (2026-10-07): `pytest` is now a dev dependency, and the offline harness (fake exchange, fake Gemini, frozen clock) exists in `tests/conftest.py`. The characterization tests already cover today's resolution (SL, TP, entry candle, same candle, 100-candle lookback), the staleness check, damaged ledger and history files, and fallback routing. Each fix still needs its own tests for the *corrected* behaviour.

- [ ] **[P2] `operate` scans every saved analysis and picks one by file modification time** — Confirmed (found 2026-10-07 while writing characterization tests) — Effort: S
  - Where: `app.py:L549-L563` (`rglob("*.json")`, then sort by `st_mtime`)
  - Problem: every `operate` run lists every analysis file ever written, across all months, so it gets slower as data grows on the small VM. It also chooses by the file's modification time, not by `metadata.timestamp`, so after a backup restore or a copy that changes mtimes, it can trade an older analysis. Pinned today by `test_operate_picks_the_analysis_by_file_mtime_not_by_name`.
  - Fix: read only the current month's folder (or keep a small "latest analysis" pointer written atomically by `analyze`), and choose by the record's own UTC timestamp.
  - Done when: a test with a newer-mtime but older-timestamp file picks the newer timestamp, and the scan reads a bounded number of files.

- [ ] **[P2] `operate` needs a Gemini key it never uses** — Confirmed — Effort: S
  - Where: `app.py:L535-L540`
  - Problem: `operate` refuses to run without `GEMINI_API_KEY`, then builds a client it never calls.
  - Fix: delete those lines.
  - Done when: `operate` works with no `.env`.

### Phase 3 – Maintainability (refactor, config, logging, packaging)

- [x] **[P1] Move hardcoded values into one config file** — Confirmed — Effort: M
  - Where: `app.py:L32` (`BASE_DIR`), `L37-L38` (models), `L219`/`L354` (candle limits), `L377-L394` (indicator lengths), `L408` (70%), `L577` (600 s), `L616-L617`, `L630-L636`/`L675-L681` (0.5 ATR, 1 ATR), `L670`/`L715` (risk $100)
  - Problem: changing risk, thresholds or models means editing numbers scattered across a 1,400-line file, and `mock` has its own copies.
  - Fix: create a `config.toml`, read with Python's built-in `tomllib` (no new library). Put in it:
    - account: starting balance, risk % (1.0)
    - costs: fee rate (0.05%), slippage
    - risk rules: max leverage (3.0), min R:R (1.5), max target distance (3 × 4H ATR), stop-floor rule, max hold hours (24)
    - the `[activity_gate]` section from the Phase 1 gate item
    - timing and data: ATR multipliers, staleness seconds, candle counts, data folder, exchange id and market type
    - the `[gemini]` section from the quota item in Phase 2
    - Keep secrets in `.env`.
    - Add a small check at start-up that rejects impossible values (e.g. risk % ≤ 0) with a friendly message.
  - Done when: no trading number is hardcoded in `app.py`, and `mock` reads the same config.
  - Status (2026-10-07): done on branch `refactor/config-toml`, with values unchanged (no strategy bump). `config.toml` has `[market]`, `[gemini]`, `[indicators]` and `[operator]`, all marked `[frozen]`. `btc_cli/config.py` checks every value at start-up. `operate` and `mock` read the same Operator settings.
    - Kept in code on purpose: the EMA 34/89/144, RSI 13/47, VMA 20 and ATR 14 lengths, because they are part of field names the prompts and records use; and `SCHEMA_VERSION` / `STRATEGY_VERSION`, which describe code and settings together.
    - Not added yet: settings for features that don't exist (fees, leverage cap, min R:R, `[activity_gate]`, the Gemini quota). Each arrives with its own item, so `config.toml` never lists a value the code ignores. The data folder comes with the paths item below.
    - `tests/test_config.py` pins the 0.1 values, so a silent edit fails a test.

- [ ] **[P1] Use one logging setup with rotation and UTC times, not several separate log files** — Confirmed — Effort: S
  - Where: `app.py:L22-L26` (`error.log`, ERROR level only), `L103-L120` (`logs/system_health.log`), `L662-L665`/`L707-L710` (`operator_errors.log`)
  - Problem:
    - Three different log files, all written in different ways.
    - Only errors are logged, so there is no record of normal runs.
    - The log files never rotate.
    - `error.log` is created in whatever folder the command is started from.
    - `operator_errors.log` stamps UTC but analysis files use local time (e.g. `22:34:43` local vs `20:34:45 UTC` for the same run).
  - Fix: set up one `logging` configuration in code. Log at INFO level to `logs/app.log` with a `RotatingFileHandler` (e.g. 5 MB × 5 files) and timestamps converted to UTC. Keep the JSON-lines health log if useful, but through `logging`. Use `datetime.now(timezone.utc)` everywhere, including file names.
  - Done when: every run writes one INFO line on start and one on finish, all timestamps are UTC, and log files rotate.
  - Progress (2026-10-07): `error.log` and `logs/` now live in the data folder, so they no longer depend on the current folder. Rotation, INFO level and UTC are still open.

- [x] **[P1] All file paths depend on the folder the command is run from** — Confirmed — Effort: S
  - Where: `app.py:L23`, `L32`, `L104`, `L804`
  - Problem: on a server, a scheduler may start the process from a different folder. Ledgers and logs would then be created somewhere else, and open trades would look "missing".
  - Fix: `PROJECT_ROOT = pathlib.Path(__file__).resolve().parent`, and build every path from it (or from `DATA_DIR` in config). Use `pathlib` instead of building paths with f-strings.
  - Done when: running `uv run /full/path/app.py status` from another folder writes into the project's data folder.
  - Status (2026-10-07): done on branch `fix/project-root-paths`. `config.PROJECT_ROOT` comes from `btc_cli/config.py`'s own location. The data folder is `[paths] data_dir` (default `"."` = the project folder, so nothing moves), overridable with the `BTC_CLI_DATA_DIR` environment variable for development runs (§2.7).
    - `output_alpha/`, `output_beta/`, `logs/` and `error.log` all sit under the data folder, and `mock_json/` is read from the project folder.
    - The console still shows paths relative to the data folder, so the output is unchanged.
    - Tests: a command started from another folder writes only into the data folder. The test harness now also fails the run if any real data file changes.

- [ ] **[P1] Keep the money logic separate from file and network code** — Confirmed — Effort: M
  - Where: `app.py:L186-L311` and `L525-L796` (long and short blocks are near-duplicates, `L628-L716`; the error-logging block is copied twice, `L641-L667` / `L686-L712`)
  - Problem: the trade math is mixed in with file reading, printing and network calls, so it cannot be tested on its own, and the copies have already drifted apart (the mock bug above).
  - Fix: pull out small functions that only take inputs and return a result, with no file or network access: `compute_order`, `resolve_trade(candles, ledger)`, `compute_pnl`, `validate_agent_response`, `is_analysis_fresh`. The command functions just load data, call these, and save.
  - Done when: each of these functions has unit tests that run without network or disk.
  - Progress (2026-10-07): the `btc_cli/` split already made `compute_order`, `find_exit`, `calculate_pnl` and `calculate_indicators` pure and unit-tested (a test checks they do no I/O). Still open: `validate_agent_response` (comes with pydantic validation, M1.4) and `is_analysis_fresh` (comes with M1.7 / M2.1).

- [ ] **[P2] Use SQLite instead of JSON files before building reports** — Suspected (needed for Phase 4) — Effort: M
  - Where: ledger, history and footprint files
  - Problem: JSON plus atomic writes is enough for 2 strategies × 1 symbol. Reporting (Phase 4) needs queries across many trades, and SQLite gives proper "all or nothing" updates (transactions) for free.
  - Fix: after Phase 2, move `trades` (open and closed), `analyses` and `rejections` into a single `data/paper.db` using Python's built-in `sqlite3` (no new library). Keep an export-to-JSON command.
  - Done when: open and close happen in one transaction, and the report command reads from the database.

- [ ] **[P2] Clean up dead files and dependencies** — Confirmed — Effort: S
  - Where: `main.py` (unused starter), `pyproject.toml:L13` (`requests`, not imported), `pyproject.toml:L4` ("Add your description here"), `bitcoin_ai_cli.egg-info/`, `BTC-CLI PROJECT (start_go-btc).md` (outdated claims), `app.py:L485`/`L911` (`except (RuntimeError, ValueError, Exception)` is the same as `except Exception`), `L491` (hardcoded "BTC/USDT" header)
  - Fix: delete `main.py` and the egg-info folder. Either remove `requests` or keep it for Telegram alerts (Phase 5). Fix the description. Mark the old status report as historical or delete it. Print the real symbol in the `status` header.
  - Status (2026-10-05): the old status report is **done** (deleted by the owner).
  - Status (2026-10-07): `main.py` and `bitcoin_ai_cli.egg-info/` deleted (owner approved), `*.egg-info/` ignored, and the description fixed. Still open: the `requests` decision (remove it, or keep it for Phase 5 alerts), the redundant `except (RuntimeError, ValueError, Exception)` (now in `pipeline.py`), and the hardcoded "BTC/USDT" `status` header. The last one changes output, so it goes with a snapshot update.
  - Done when: `pyproject.toml` lists only libraries the code imports.

- [ ] **[P2] `auto` downloads market data twice** — Confirmed — Effort: S
  - Where: `app.py:L1290-L1291`
  - Problem: `status` fetches and prints the data, then `analyze` fetches it again. The panel you see and the numbers the AI gets can differ.
  - Fix: fetch once in `auto` and pass the data to both steps.
  - Done when: the `status` file and the analysis files from one `auto` run contain identical `raw_market_data`.

- [ ] **[P2] Consider replacing `pandas-ta`** — Confirmed — Effort: M
  - Where: `pyproject.toml:L11`, `uv.lock:L753-L765`, `app.py:L375-L390`
  - Problem:
    - The locked version is `0.4.71b0`, a **beta**. Its maintainer has said there will be no more releases without more sponsorship.
    - It pulls in `numba` and `llvmlite`, which are large, slow to import (the single test took 5.3 s), and pin `numpy` below 2.3.
    - Only 4 of its indicators are used (EMA, RSI, SMA, ATR).
    - It does work today, and ARM64 wheels exist for all of its dependencies (see Phase 5).
  - Fix: write the 4 indicators in plain `pandas` (about 30 lines; `research/activity_gate_study.py` already has a hand-written ATR you can reuse), check them against `pandas-ta` output, then remove the library with `uv remove pandas-ta` (**approved 2026-10-04**).
  - Done when: the new indicators match `pandas-ta` to 1e-8 on a saved candle set, and `numba` is gone from `uv.lock`.

- Split `app.py` into modules: **moved to Phase 0** (owner decision 2026-10-05).

- [x] **[P2] Line-ending warnings** — Confirmed — Effort: S
  - Where: `git diff` warns "LF will be replaced by CRLF" for `app.py`, `README.md`, `.gitignore`
  - Fix: add a `.gitattributes` file with `* text=auto eol=lf` so Windows and Linux copies match.
  - Done when: `git diff` shows no line-ending warnings.
  - Status (2026-10-07): done on branch `chore/gitattributes`. Every tracked file was already stored with LF, so no file content changed.

### Phase 4 – Evaluation (metrics and reporting)

- [ ] **[P1] Add a `report` command** — Confirmed (does not exist) — Effort: M
  - Where: new command; data from the history files (or SQLite)
  - Problem: there is no way to tell whether Defensive or Greedy is better, or whether either beats doing nothing.
  - Fix: per strategy, and split by `model_used`, show:
    - number of trades, win rate, and **average R** (PnL ÷ risk at entry)
    - **expectancy** (average net R per trade) and **profit factor** (gross wins ÷ gross losses)
    - **max drawdown** on the net equity curve, total fees paid, and average time in a trade
    - number of ambiguous same-candle results
    - a funnel: analyses → GO verdicts → rejected tickets (by reason) → trades
  - Store `r_multiple` on each closed trade so this is a simple sum.
  - Done when: `uv run app.py report` prints this table from saved history, and a test checks it against a hand-made history file.

- [ ] **[P1] Record every decision, not just trades** — Confirmed — Effort: S
  - Where: `app.py:L584-L586` (SIT ON HANDS is not recorded by `operate`), `L641-L667` (rejections go only to a text log)
  - Problem: you cannot measure how often each strategy trades, or why tickets get rejected.
  - Fix: write one structured record per strategy per run with the verdict, the rejection reason if any, the model used and the R:R.
  - Done when: the report's funnel numbers come from these records.

- [ ] **[P2] Explain sample size in the report** — Confirmed — Effort: S
  - Problem: a 60% win rate after 10 trades means almost nothing. Fewer than about 30 closed trades per strategy is noise; aim for 100 or more before drawing conclusions.
  - Fix: print "sample too small" below 30 trades, and show a simple confidence range for the win rate.
  - Done when: the report shows the warning.

- [ ] **[P2] Compare against a simple non-AI baseline** — Confirmed (none exists) — Effort: M
  - Problem: without a baseline you cannot tell whether the AI adds anything beyond the stop and target rules.
  - Fix: add a third "strategy" that uses the same operator but takes its direction from a plain rule (e.g. EMA 34 vs 89 on 4H, or a random coin flip). Track it in the same report.
  - Done when: the report shows the baseline next to Defensive and Greedy.

- [ ] **[P2] Track how far each trade moved for and against you** — Confirmed — Effort: S
  - Problem: the largest move against the trade (MAE) and in its favour (MFE) show whether stops are too tight or targets too far.
  - Fix: record the highest high and lowest low between entry and exit on each trade.
  - Done when: these two values are saved on every closed trade.

### Phase 5 – Deployment to Oracle Cloud (only after Phases 1–3 are done)

- [ ] **[P1] Choose the Oracle region, and make sure Binance keeps working from it** — Suspected — Effort: S (test) / M (fallback)
  - Where: `app.py:L218`, `L477`, `L903` (`ccxt.binance()` hardcoded)
  - Problem:
    - On Oracle, Always Free resources can only be created in your **home region**, which you choose at sign-up and cannot change later.
    - Binance blocks some countries and many cloud IP ranges. US IPs get HTTP 451 ("unavailable for legal reasons"), and the Netherlands is on Binance's restricted list.
    - **New for EU residents:** Binance has no MiCA licence (the EU's new crypto licence). Since **2026-07-01** it no longer offers new trading, deposits or sign-ups to EU customers, and Spain's regulator (CNMV) refused any extension.
      - This tool only *reads public market data*, which needs no account, and that still works. From your connection on 2026-10-04 these all returned HTTP 200: Binance spot (`api.binance.com`), Binance futures (`fapi.binance.com`), `data-api.binance.vision`, OKX, Bybit and Kraken.
      - Binance could still block EU or cloud IP addresses later, so a fallback exchange is no longer optional.
    - Spain is on Google's list of countries where the Gemini API is available.
  - Recommended region: **Spain Central (Madrid), `eu-madrid-1`**.
    - It is in the same country as you, so the same rules apply as for your home connection, which already works. Latency is low, and your data stays under EU law.
    - Second choices: Paris or Milan.
    - Avoid US regions (Binance blocks them), Amsterdam (the Netherlands is restricted), and Frankfurt (Ampere A1 VMs are famously hard to get there).
    - Madrid has a single availability domain, and I could not find whether A1 capacity is easy to get there. If the A1 VM shows "out of capacity", retry at quiet hours or temporarily use the AMD Micro shape.
  - Confirm by: right after sign-up, open **Cloud Shell** in the Oracle console (it runs in your home region) and run:
    - `curl -s -o /dev/null -w "%{http_code}\n" https://fapi.binance.com/fapi/v1/ping`
    - `curl -s -o /dev/null -w "%{http_code}\n" https://generativelanguage.googleapis.com`

    A `200` (Binance) or a `404` (the Google address answering without a path) means you can reach it; `451` or `403` means blocked. Run the same commands again from the VM once it exists.
  - Fix:
    - Make the exchange and market a config value (`exchange_id = "binanceusdm"`, `symbol = "BTC/USDT:USDT"`).
    - Add a fallback exchange through ccxt for the perpetual market: **OKX** (`okx`, `BTC/USDT:USDT`) or **Bybit** (`bybit`, `BTC/USDT:USDT`). Both answered from Spain.
    - Record which exchange supplied each run's data, because prices differ slightly between exchanges.
    - Alert when the fallback is used.
  - Done when: the curl checks pass from the VM, and a forced Binance failure (e.g. a wrong URL in a test) falls back to the second exchange cleanly.

- [ ] **[P1] Run on a schedule with systemd** — Confirmed (no scheduler exists) — Effort: M
  - Where: new files in `deploy/`
  - Problem: there is no loop or scheduler. Running `while True` in a terminal stops at the first crash or reboot.
  - Fix:
    - Create a `systemd` service with `Type=oneshot` that runs `uv run --locked app.py auto`, with `WorkingDirectory=` set to the project and `User=` set to a dedicated user.
    - Add a matching `.timer` with `OnCalendar=*:0/30:30` (30 seconds after every second 15m close, i.e. every 30 minutes, matching the quota decision) and `Persistent=true`.
    - Add a second, daily timer that runs `app.py check-setup` (see the quota item in Phase 2).
    - systemd will not start a oneshot service that is still running, and the lock file covers manual runs.
    - Set `TZ=UTC`. Add `TimeoutStartSec=` so a hung run is killed.
  - Done when: `systemctl list-timers` shows the timer, and runs survive a reboot.

- [ ] **[P1] Remove Windows-only assumptions** — Confirmed — Effort: S
  - Where: README L108-L125 (PowerShell-only install), README L119 (`cd C:\Users\dsant\…`), `app.py:L318`/`L743` (local-time file names), `open()` calls without `encoding=` (`L119`, `L205`, `L289`, `L295`, `L343`, `L566`, `L664`, `L709`, `L762`, `L785`)
  - Fix:
    - Add Linux install steps to the README (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
    - Remove the personal path.
    - Use UTC in file names (`…Z`).
    - Pass `encoding="utf-8"` to every `open()`.
  - Done when: a fresh clone on Ubuntu installs and runs using only the README.
  - Progress (2026-10-07): every `open()` now passes `encoding="utf-8"`, and paths are built from the project root. Still open: Linux install steps and the personal `cd C:\Users\dsant\…` path in the README, and UTC file names.

- [ ] **[P1] Python 3.13 and dependencies on ARM64 (aarch64)** — Confirmed (lockfile) / Suspected (runtime) — Effort: S
  - Where: `uv.lock`
  - Problem: none found in the lockfile. Every compiled dependency has a `cp313` Linux ARM64 wheel: `numpy 2.2.6`, `pandas 2.3.3`, `numba 0.61.2` (manylinux_2_28), `llvmlite 0.44.0` (manylinux_2_27/2_28). Those need glibc ≥ 2.28, which Ubuntu 22.04/24.04 and Oracle Linux 8/9 all have. `uv` downloads Python 3.13.12 for aarch64 by itself.
  - Confirm by: running `uv sync --locked`, then `uv run pytest`, on the VM. This is also the first run of the run lock's Linux (`fcntl.flock`) code path; `tests/test_storage.py` covers it.
  - Fix: nothing needed now. If you choose the small AMD Micro VM instead (1 GB RAM), check memory use; `numba` makes `pandas-ta` heavy.
  - Done when: the tests pass on the VM.

- [ ] **[P1] Secrets on the server** — Confirmed — Effort: S
  - Where: `app.py:L19` (`load_dotenv()` from the current folder)
  - Fix: keep the key in `/etc/btc-cli/env` with `chmod 600`, owned by the service user, loaded with `EnvironmentFile=` in the systemd unit. Never commit it or print it in logs. Use a separate Gemini API key for the server so it can be revoked on its own.
  - Done when: `ls -l` shows the env file as `-rw-------`, and the service runs without a `.env` in the repo folder.

- [ ] **[P1] Alerts and a heartbeat so silent failures get noticed** — Confirmed (none exist) — Effort: M
  - Problem: on an unattended server, nobody sees the terminal. A run that stops working looks exactly like a run that chose SIT ON HANDS.
  - Fix:
    - Send a Telegram bot message (simple HTTP POST; `requests` is already a dependency) on: run errors, a switch to the backup model or backup exchange, quota exhaustion, `check-setup` warnings, damaged files, trade opened, trade closed (with net PnL and R).
    - After each successful run, ping a free heartbeat service such as healthchecks.io. It emails you if no ping arrives for e.g. 45 minutes.
    - Send a daily summary.
  - Done when: stopping the timer triggers a "missed heartbeat" email, and a forced error sends a Telegram message.

- [ ] **[P1] Limit disk growth** — Confirmed — Effort: S
  - Where: `app.py:L314-L346`, `L742-L763`; `L549` (`rglob` over every analysis file on each `operate`)
  - Problem: one `auto` run writes about 7.4 KB in 3 files. Every 15 minutes that is about 0.7 MB/day but **about 105,000 files a year**, and `operate` scans all of them on every run.
  - Fix: keep a `latest.json` pointer per strategy so `operate` doesn't scan. Compress months older than 2 into a `.tar.gz`. Use `RotatingFileHandler` for logs. Moving to SQLite (Phase 3) also solves this.
  - Done when: the time for `operate` does not grow with history, and old months are compressed automatically.

- [ ] **[P1] Back up the ledger and trade history** — Confirmed (no backups exist) — Effort: S
  - Fix: back up the ledger/history files (or `paper.db` via `sqlite3 .backup`) nightly with a systemd timer. Copy them to OCI Object Storage (part of Always Free) or to a private Git repository. Keep 30 days.
  - Done when: restoring last night's backup onto a fresh checkout reproduces the report.

- [ ] **[P2] Oracle Always Free gotchas** — Suspected — Effort: S
  - Problem:
    - Oracle may reclaim an Always Free VM it considers idle: over 7 days, 95th-percentile CPU, network and memory (A1 shapes) all under 20%. This workload is very light, so it could qualify.
    - ARM (A1) capacity is often "out of host capacity" in popular regions.
  - Fix:
    - Upgrade the account to Pay-As-You-Go. Oracle states idle reclamation applies to free-tier accounts; resources within the Always Free limits stay free. Check this against Oracle's current docs before relying on it.
    - Keep the backups above, so a reclaimed VM is a short rebuild.
    - Write a `deploy/README.md` with the rebuild steps.
    - If A1 is unavailable, retry creating it later, or use the AMD Micro shape temporarily.
  - Done when: the rebuild steps are documented and have been tested once.

### Later / nice to have

- [ ] **[P2] Partial exits and trailing stops** (already on the README roadmap). Only after fees and resolution are correct. — Effort: M
- [ ] **[P2] Replay test for the rules (backtest).** Replay stored candles through `compute_order` and `resolve_trade` to test stop and target rules without spending AI calls. The AI part cannot be backtested honestly, because the model may have seen that price history in training. The synthetic-data item in Phase 1 covers the rules part of this on generated data. — Effort: L
- [ ] **[P2] Feedback loop** that gives the agents their past results (README roadmap). Only after you have 100+ trades and a baseline. — Effort: L
- [ ] **[P2] Multiple symbols** once the rounding and per-symbol settings are fixed. — Effort: M

---

## 4. Open questions for the owner

1. **Spot or futures?** Should the paper account act like a spot account (long only, no leverage, 0.10% fees) or a USDT-M perpetual (shorts allowed, leverage, about 0.05% fees plus funding)? Today it quietly assumes the second without modelling it.
  - Futures. We might want to operate short as well as long.
2. **How should position size be set?** Fixed $100 risk per trade (simplest to compare A vs B), or 1% of current equity (more realistic)? I lean towards fixed $ risk with results in R multiples for the A/B test.
  - Keep it percentage. We might want to increase/decrease the risked amounts with the progressive variation of the funds
3. **Holding period.** Are these intraday trades (target within a few 15m ATRs) or multi-day swings (4H levels)? This decides which ATR sets the stop and how far the target may be.
  - 
  - **Reply (Claude, 2026-10-04):** no answer yet, so I've proposed a default that fits your other answers.
    - Since you agreed to targets up to 3 × 4H ATR, these are **short swings: a few hours up to 48 hours**, closed at market after 48 h.
    - Stop floor = the larger of 1 × 15m ATR and 0.5 × 4H ATR.
    - Details are in the Phase 1 item "Stop and target come from different timeframes".
    - **Please confirm or change the 48 h limit.**
  - **Owner (2026-10-04):** 24 hours holding. Also add a low-volume / low-ATR flag that cancels trading.
  - **Reply (Claude, 2026-10-04):** 24 h is now the plan. The filter design and the data study behind it are in the Phase 1 item "Market activity gate". Please confirm the proposed thresholds:
    - RVOL < 0.5
    - 15m ATR < 0.15%
    - 4H ATR < 0.60%
    - fees > 0.30R
  - **Owner (2026-10-04):** thresholds accepted. They must be re-checked periodically. `research/` folder approved. Don't start Phase 1 yet; settle the foundations first. Add a documentation file for changes.
  - **Reply (Claude, 2026-10-04):**
    - Re-check schedule: **monthly** (Phase 1 gate item), with a threshold changed only after two `REVIEW` reports in a row, plus a quarterly check against real trades.
    - The study now lives in `research/activity_gate_study.py`.
    - Changes and decisions are recorded in the new `CHANGELOG.md`.
4. **Minimum R:R and maximum target distance.** Suggested starting values: R:R ≥ 1.5 after fees, target ≤ 3 × 4H ATR. Do you agree?
  - I agree
5. **Alpha/beta folders.** Should backup-model runs keep trading on their own, or share one ledger per strategy and just be tagged with `model_used`? I recommend sharing and tagging; separate ledgers cause the duplicate/orphan bug.
  - Follow your approach.
6. **Run frequency.** Every 15m candle close (about 384 AI calls/day at most) or hourly? This depends on your project's real free-tier daily limit shown in AI Studio.
  - For the moment Gemini 3.5 Flash Lite allows me 500 Requests per day. I will update this information if the rate limits change in the future. Let's have the rate limit stated somewherw in the project. It can be in .env or a specific file (choose the optimal approach). The rate limits mut be checked periodically in order to get access to better model's eventually and to avoid using deprecated models or failed API calls.
  - **Reply (Claude, 2026-10-04):**
    - The limit goes in **`config.toml`** (a `[gemini]` section), not `.env`. `.env` is only for secrets, while limits are normal settings you want to see and track in Git.
    - A new **`check-setup`** command, run daily by the scheduler, will:
      - catch models that have been shut down
      - list newer Flash / Flash-Lite models you could switch to
      - remind you to recheck the numbers in AI Studio every 30 days (Google does not let programs read the quota numbers)
    - A daily call counter will stop new analyses at 90% of the 500 limit.
    - With 4 calls per run, I recommend running **every 30 minutes** (192 calls/day at most).
    - See the Phase 2 item "Store the Gemini quota in config…".
7. **Oracle home region and your country.** Which region will you use? It has to allow Binance and the Gemini API, and it cannot be changed after sign-up.
  - Which region should I establish? I live in Spain at the moment, but I don't know whether it has restrictions. Suggest a region's choice.
  - **Reply (Claude, 2026-10-04): choose Spain Central (Madrid), `eu-madrid-1`.**
    - Spain is supported by the Gemini API.
    - Binance stopped *offering trading* to EU customers on 2026-07-01 (no MiCA licence), but its *public price data*, which is all this tool uses, still answered from your connection today, for both spot and futures.
    - Second choices: Paris or Milan. Avoid US regions, Amsterdam and Frankfurt.
    - Because Binance could block EU addresses in future, the plan now includes a backup exchange (OKX or Bybit), and both are reachable from Spain.
    - After sign-up, run the two `curl` checks in Oracle's Cloud Shell before building anything (Phase 5, first item).
8. **Approvals needed under `AGENTS.md`:**
   - `uv add --dev pytest`
   - possibly `uv add pydantic` (already installed indirectly)
   - removing `pandas-ta` and writing the indicators by hand
   - eventually splitting `app.py` into modules
  - You have green light to proceed with these changes if needed.
9. **Uncommitted changes.** Should the current working-copy edits (model names, README rewrite, staged `venv/` deletions) be committed as they are before Phase 1 starts?
  - Yes. Commited and pushed thorugh a PR.
  - **Reply (Claude, 2026-10-04):**
    - Committed as `eed40d8` on branch `chore/model-update-readme-rewrite`.
    - The push was refused (403), because this PC is logged in to GitHub as `Bahia90Studio`, not `DariSant`. The steps to fix it are in the first Phase 2 item.
    - `TODO.md` and `BTC-CLI PROJECT (start_go-btc).md` were left out of the commit.
