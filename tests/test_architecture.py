"""Checks the module boundaries from AGENTS.md §10 by reading each module's imports."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "btc_cli"
PURE_MODULES = ["indicators", "trade_operator", "ledger"]
# Anything that touches the network, the disk, the terminal or the CLI.
IMPURE_IMPORTS = {"ccxt", "google", "os", "pathlib", "shutil", "socket", "requests", "rich", "typer", "logging"}


def imports_of(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            if node.module == "btc_cli":
                names.update(f"btc_cli.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("module", PURE_MODULES)
def test_pure_modules_do_no_io(module):
    roots = {name.split(".")[0] for name in imports_of(module)}
    assert not roots & IMPURE_IMPORTS
    assert not any(name.startswith("btc_cli") and name != "btc_cli.config" for name in imports_of(module))


def test_nothing_imports_the_cli():
    modules = [p.stem for p in PACKAGE.glob("*.py") if p.stem not in ("cli", "__init__")]
    assert modules, "package not found"
    for module in modules:
        assert "btc_cli.cli" not in imports_of(module), f"{module} imports btc_cli.cli"
