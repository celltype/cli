"""Shared pytest configuration and fixtures."""

import os
import sys
from pathlib import Path
import pytest

# Ensure `import ct` works when running from repo root without editable install.
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e", action="store_true", default=False,
        help="Run end-to-end tests that hit real APIs",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e"):
        return  # Run all tests including e2e
    # Skip e2e tests by default
    skip_e2e = pytest.mark.skip(reason="Need --run-e2e to run")
    for item in items:
        if "test_e2e" in item.nodeid or "e2e" in item.keywords:
            item.add_marker(skip_e2e)


def has_api_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _has_cellxgene():
    try:
        import cellxgene_census
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Rich console capture fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def captured_console():
    """Yield a (console, buffer) tuple for capturing Rich output."""
    from io import StringIO
    from rich.console import Console
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    return console, buf
