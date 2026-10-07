# Lessons

Shared memory for every agent working on this repo (`AGENTS.md` §12). Newest first. Keep each lesson short: date, what happened, the rule, and the source.

## 2026-10-07: importing `app` has side effects
- What: `app.py` calls `load_dotenv()` (reads the real `.env` and its key) and `logging.basicConfig(filename="error.log")` (creates `error.log` in the current folder) as soon as it is imported.
- Rule: tests must import `app` only after `tests/conftest.py` has run. It disables `load_dotenv` and puts a `NullHandler` on the root logger first. Don't add another conftest or test entry point that imports `app` earlier.
- Source: branch `test/characterization-tests`.

## 2026-10-07: time-zone bugs can be tested on Windows
- What: the CLI uses naive local time for file names and `metadata.timestamp`, so the same instant looks different in Madrid and on a UTC server. Python on Windows honours the `TZ` environment variable (for example `EST5` or `JST-9`).
- Rule: after touching anything time-related, run the suite under at least two zones. In bash: `TZ=EST5 uv run pytest` and `TZ=JST-9 uv run pytest`. In PowerShell: `$env:TZ = "EST5"; uv run pytest; Remove-Item Env:TZ`.
- Source: branch `test/characterization-tests`.

## 2026-10-07: snapshot diffs are behaviour changes
- What: `tests/characterization/snapshots/` records today's behaviour, including known bugs.
- Rule: never regenerate snapshots (`UPDATE_SNAPSHOTS=1`) to make a failing test pass. A diff is either an unintended change (fix the code) or an approved §2.4 change (regenerate, review the diff, and mention it in `CHANGELOG.md`).
- Source: branch `test/characterization-tests`.
