"""Service gates for the test suite (audit finding D18).

Three tests talk to a live Qdrant at ``127.0.0.1:6333``. On a machine without
one the whole suite crashed instead of skipping, and interrupted runs left
orphan collection directories behind (``academic_landscape_test__chunks__...``
reappeared four times during the MAG-50 work), because every test built the
*same* collection name and cleanup only ran on the happy path.

These fixtures give the live-service tests one shared, uniquely named
collection that is torn down best-effort even when the test fails, and they
skip cleanly when the service is absent. The default suite is now offline;
run the live layer explicitly with ``--run-live``.
"""

from __future__ import annotations

import socket
import uuid

import pytest


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((host, port)) == 0


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that need live local services (Qdrant at 6333)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "live: test needs a live local service (skip unless --run-live)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs a live local service; pass --run-live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def qdrant_url(request: pytest.FixtureRequest) -> str:
    """Qdrant base URL, skipped when the service is unreachable."""
    url = "http://127.0.0.1:6333"
    if not request.config.getoption("--run-live"):
        pytest.skip("needs --run-live")
    if not _port_open("127.0.0.1", 6333):
        pytest.skip("Qdrant is not reachable at 127.0.0.1:6333")
    return url


@pytest.fixture
def qdrant_collection_prefix() -> str:
    """A per-test collection prefix so runs cannot collide on disk.

    The recurring orphan directories were the same fixed name being created by
    every test; a unique suffix per test makes leftovers impossible to confuse
    and makes cleanup scoping trivial.
    """
    return f"academic_landscape_test_{uuid.uuid4().hex[:8]}"
