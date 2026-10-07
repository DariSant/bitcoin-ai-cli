# Lessons

Shared memory for every agent working on this repo (`AGENTS.md` §12). Newest first. Keep each lesson short: date, what happened, the rule, and the source.

## 2026-10-07: data paths are absolute now, so test isolation lives in `conftest.py`
- What: before `fix/project-root-paths`, tests stayed away from real data only because they `chdir` into `tmp_path` and every path was relative. Now every path is absolute (`config.DATA_DIR`, `BASE_DIR`, `BETA_DIR`, `LOGS_DIR`, `ERROR_LOG`, `MOCK_DIR`).
- Rule: any new path setting must also be pointed at `tmp_path` in the `isolated_env` fixture. The session guard `real_data_untouched` catches a miss, but only after the damage is done. Build new paths from these settings, never from `os.getcwd()` or a bare relative string.
- Source: branch `fix/project-root-paths`.

## 2026-10-07: merging is the owner's job, even when asked
- What: the owner asked the agent to "merge everything into main". `gh pr merge` was then blocked by the Claude Code permission check ("merge without review"). `AGENTS.md` §9 also says the agent never merges.
- Rule: push the branch, open the PR, and hand the owner the link to merge. Don't look for another way to merge. Stack follow-up work on a branch made from the unmerged one, and say so in its PR.
- Source: PR #34, conversation 2026-10-07.

## 2026-10-07: `os.kill(pid, 0)` is not a harmless "is it alive?" check on Windows
- What: on Windows, signal 0 is `CTRL_C_EVENT`, so `os.kill(pid, 0)` sends Ctrl+C to the process instead of just probing it.
- Rule: for the run lock, use an OS file lock (`fcntl.flock` / `msvcrt.locking`), never PID probing.
- Source: planning the run lock on branch `fix/safe-storage`.

## 2026-10-07: `UPDATE_SNAPSHOTS=1` rewrites every snapshot, not just the changed ones
- What: this checkout has `core.autocrlf=true`, so snapshots sit on disk with CRLF. The snapshot writer writes LF, so regenerating marks all 21 files as modified even when only 4 changed. Git itself shows no content diff for the rest.
- Rule: after regenerating, run `git diff --stat` (it ignores the CRLF/LF difference) to see the real changes, then restore the untouched files with `git checkout -- <files>`. To prove a record change is additive, copy the snapshots aside first and compare key by key, normalising line endings.
- Source: branch `feat/record-versioning`.

## 2026-10-07: importing `app` has side effects
- What: `app.py` calls `load_dotenv()` (reads the real `.env` and its key) and `configure_logging()` (creates `error.log` in the current folder) as soon as it is imported. The `btc_cli` modules themselves have no import side effects.
- Rule: tests must import `app` only after `tests/conftest.py` has run. It disables `load_dotenv` and puts a `NullHandler` on the root logger first. Don't add another conftest or test entry point that imports `app` earlier. Running `uv run app.py mock <missing file>` by hand writes to the real `error.log`, so clear what you added afterwards.

## 2026-10-07: `config.BASE_DIR` must be read at call time
- What: the fallback path rebinds `config.BASE_DIR` to `output_beta` for the rest of the process (known P0 bug), and `auto` depends on that today.
- Rule: until that bug is fixed, write `config.BASE_DIR`, never `from btc_cli.config import BASE_DIR`. The second form silently changes where `auto` writes, and `test_auto_after_a_fallback_operates_on_output_beta` catches it.

## 2026-10-07: shell pitfalls seen on this Windows setup
- What: in the Bash tool, a Python `"\\n"` (escaped backslash-n) inside a heredoc reached Python as a real newline, which broke generated code and a test. The same happened with backslash escapes in a `sed -i` replacement and in Python string literals passed through a heredoc (branch `feat/record-versioning`). `git add a b missing` aborts the whole add, so changes staged earlier (e.g. a `git mv`) end up in the next commit. `git worktree remove` fails with "Filename too long" once a worktree has its own `.venv`.
- Rule: write scripts that contain backslash escapes with the file-writing tool (or edit them with the editor), never through a heredoc or `sed`. Check `git status` before every commit. For a scratch worktree, set `UV_PROJECT_ENVIRONMENT` to the repo's `.venv` so no second `.venv` is created.
- Source: branch `test/characterization-tests`.

## 2026-10-07: time-zone bugs can be tested on Windows
- What: the CLI uses naive local time for file names and `metadata.timestamp`, so the same instant looks different in Madrid and on a UTC server. Python on Windows honours the `TZ` environment variable (for example `EST5` or `JST-9`).
- Rule: after touching anything time-related, run the suite under at least two zones. In bash: `TZ=EST5 uv run pytest` and `TZ=JST-9 uv run pytest`. In PowerShell: `$env:TZ = "EST5"; uv run pytest; Remove-Item Env:TZ`.
- Source: branch `test/characterization-tests`.

## 2026-10-07: snapshot diffs are behaviour changes
- What: `tests/characterization/snapshots/` records today's behaviour, including known bugs.
- Rule: never regenerate snapshots (`UPDATE_SNAPSHOTS=1`) to make a failing test pass. A diff is either an unintended change (fix the code) or an approved §2.4 change (regenerate, review the diff, and mention it in `CHANGELOG.md`).
- Source: branch `test/characterization-tests`.
