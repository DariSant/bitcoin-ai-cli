"""Shared test harness: isolation, fakes and snapshots.

Every test runs inside its own tmp_path (the CLI writes relative paths), with no
network, no .env, a frozen clock, a fake exchange and a fake Gemini client.

The fakes are wired in through `_patch_app_modules`, which patches the CLI's
module globals by name. When `app.py` is split into `btc_cli/`, only this file
should need to change, never the tests themselves.
"""

import json
import logging
import os
import socket
import sys
from datetime import datetime as real_datetime
from datetime import timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator

import pytest

# --- Import-time isolation (must run before anything imports `app`) ---

# app.py calls load_dotenv() on import. Tests must never read the real .env or its key.
import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False

# app.py calls logging.basicConfig(filename="error.log") on import, which would create
# error.log in the repo. basicConfig is a no-op once the root logger has a handler.
logging.getLogger().addHandler(logging.NullHandler())

import ccxt  # noqa: E402
from google import genai  # noqa: E402
from rich.console import Console  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

import app as app_module  # noqa: E402
from btc_cli import config  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
SNAPSHOT_DIR = TESTS_DIR / "characterization" / "snapshots"

# The last fixture candle opens at this instant (see tests/fixtures/make_candles.py).
FROZEN_UTC = real_datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
FROZEN_EPOCH = FROZEN_UTC.timestamp()

PRIMARY_MODEL = config.PRIMARY_MODEL
FALLBACK_MODEL = config.FALLBACK_MODEL


def load_candles(timeframe: str) -> list[list[float]]:
    """Return the saved fixture candles for a timeframe."""
    with open(FIXTURES_DIR / f"candles_{timeframe}.json", encoding="utf-8") as f:
        return json.load(f)


# --- Frozen clock ---

class Clock:
    """Controls what `datetime.now()` returns inside the CLI."""

    def __init__(self) -> None:
        self.epoch = FROZEN_EPOCH

    def advance(self, seconds: float) -> None:
        self.epoch += seconds

    def local_naive(self) -> real_datetime:
        """What the CLI's naive `datetime.now()` returns: local wall time, machine dependent."""
        return real_datetime.fromtimestamp(self.epoch)


def _make_frozen_datetime(clock: Clock) -> type[real_datetime]:
    class FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls.fromtimestamp(clock.epoch, tz)

    return FrozenDatetime


# --- Fake exchange ---

class FakeExchange:
    """Stands in for `ccxt.binance()`: serves saved candles and records every call."""

    def __init__(self) -> None:
        self.candles: dict[str, list[list[float]]] = {"4h": load_candles("4h"), "15m": load_candles("15m")}
        self.calls: list[dict[str, Any]] = []
        self.instances = 0
        self.error: Exception | None = None

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", since: int | None = None, limit: int | None = None, params: dict | None = None) -> list[list[float]]:
        self.calls.append({"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit})
        if self.error is not None:
            raise self.error
        rows = self.candles[timeframe]
        if limit:
            rows = rows[-limit:]
        return [list(row) for row in rows]


# --- Fake Gemini ---

class FakeGemini:
    """Stands in for `genai.Client`.

    `script` is consumed in call order: a str is returned as the reply text,
    an Exception is raised. Every call is recorded, including failed ones.
    """

    def __init__(self) -> None:
        self.script: list[str | Exception] = []
        self.calls: list[dict[str, Any]] = []
        self.clients_created = 0
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def client_factory(self, *args: Any, **kwargs: Any) -> "FakeGemini":
        self.clients_created += 1
        return self

    def _generate_content(self, model: str, contents: str, config: dict | None = None) -> SimpleNamespace:
        schema = config.get("response_schema") if config else None
        self.calls.append({
            "model": model,
            "contents": contents,
            "response_mime_type": config.get("response_mime_type") if config else None,
            "response_schema": schema.__name__ if schema else None,
        })
        if not self.script:
            raise AssertionError(f"Unexpected Gemini call #{len(self.calls)} (model={model}); the test script is exhausted.")
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)

    def models_used(self) -> list[str]:
        return [call["model"] for call in self.calls]


# --- Wiring ---

def _app_modules() -> list[ModuleType]:
    """The CLI's own modules: `app` today, plus `btc_cli.*` after the split."""
    return [m for name, m in list(sys.modules.items()) if m is not None and (name == "app" or name == "btc_cli" or name.startswith("btc_cli."))]


def _patch_app_modules(monkeypatch: pytest.MonkeyPatch, name: str, matches: Callable[[Any], bool], value: Any) -> None:
    for module in _app_modules():
        if hasattr(module, name) and matches(getattr(module, name)):
            monkeypatch.setattr(module, name, value)


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Tests must not touch the network (AGENTS.md §7).")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# --- Guard: the real dataset must not change while tests run (AGENTS.md §7) ---

REAL_DATA = [REPO_ROOT / "output_alpha", REPO_ROOT / "output_beta", REPO_ROOT / "logs", REPO_ROOT / "error.log"]


def _real_data_state() -> dict[str, tuple[int, int]]:
    """Size and modification time of every real data file (only metadata is read, never content)."""
    state = {}
    for root in REAL_DATA:
        files = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        for path in files:
            info = path.stat()
            state[path.relative_to(REPO_ROOT).as_posix()] = (info.st_size, info.st_mtime_ns)
    return state


@pytest.fixture(scope="session", autouse=True)
def real_data_untouched() -> Iterator[None]:
    before = _real_data_state()
    yield
    after = _real_data_state()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"Tests changed real data files: {changed}"


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def exchange() -> FakeExchange:
    return FakeExchange()


@pytest.fixture
def gemini() -> FakeGemini:
    return FakeGemini()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, clock: Clock, exchange: FakeExchange, gemini: FakeGemini) -> Iterator[Path]:
    """Run every test in tmp_path with no network, a frozen clock and fake services."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    _block_network(monkeypatch)

    def make_exchange(*args: Any, **kwargs: Any) -> FakeExchange:
        exchange.instances += 1
        return exchange

    monkeypatch.setattr(ccxt, "binance", make_exchange)
    monkeypatch.setattr(genai, "Client", gemini.client_factory)

    _patch_app_modules(monkeypatch, "datetime", lambda v: v is real_datetime, _make_frozen_datetime(clock))
    # A fixed-width, colourless console that writes to whatever sys.stdout is (CliRunner's buffer).
    test_console = Console(width=100, color_system=None, force_terminal=False, legacy_windows=False)
    _patch_app_modules(monkeypatch, "console", lambda v: isinstance(v, Console), test_console)
    # Every data path points into tmp_path. monkeypatch also restores BASE_DIR, which the
    # fallback path rebinds to BETA_DIR (known bug).
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path / "output_alpha")
    monkeypatch.setattr(config, "BETA_DIR", tmp_path / "output_beta")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "ERROR_LOG", tmp_path / "error.log")
    monkeypatch.setattr(config, "MOCK_DIR", tmp_path / "mock_json")
    # The run lock is not data: keep it out of tmp_path so the file listings in snapshots are unchanged.
    monkeypatch.setattr(config, "LOCK_FILE", tmp_path_factory.mktemp("lock") / "run.lock")

    yield tmp_path


# --- CLI runner ---

@pytest.fixture
def run_cli() -> Callable[..., Any]:
    """Invoke the real Typer app, e.g. run_cli("analyze", "--def")."""
    runner = CliRunner()

    def run(*args: str) -> Any:
        return runner.invoke(app_module.app, list(args))

    return run


# --- Snapshots ---

def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content if content.endswith("\n") else content + "\n"
    return json.dumps(content, indent=2, ensure_ascii=False) + "\n"


@pytest.fixture
def snapshot() -> Callable[[str, Any], None]:
    """Compare content with tests/characterization/snapshots/<name>.

    Regenerate with UPDATE_SNAPSHOTS=1 only when a behaviour change is intended
    and approved; a snapshot diff is exactly what these tests exist to catch.
    """

    def check(name: str, content: Any) -> None:
        path = SNAPSHOT_DIR / name
        actual = _to_text(content)
        if os.environ.get("UPDATE_SNAPSHOTS") == "1":
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(actual)
            return
        if not path.exists():
            pytest.fail(f"Missing snapshot {path.name}. Create it with UPDATE_SNAPSHOTS=1 after reviewing the output.")
        with open(path, encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected, f"Output differs from snapshot {path.name}"

    return check
