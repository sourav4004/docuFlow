"""Global pytest configuration (Phase 24).

ENVIRONMENT GATE for native pgvector tests.

A small set of legacy retrieval suites (Phases 4–6) intentionally use the
REAL application PostgreSQL and native pgvector operators (``<=>``). They
cannot run against the shared SQLite fixture and cannot run at all when the
``vector`` extension is absent. Every test in those modules is automatically
marked ``env_pgvector``; the report hook converts ONLY the specific
``type "vector" does not exist`` infrastructure error into an explicit skip
with a documented reason. Any other failure — assertion, KeyError, TypeError
— still fails loudly. When pgvector is installed the marker is inert and
every test runs for real.

This is deliberately narrow: a blanket exception→skip would hide real
regressions in the JSON fallback path (which is validated by the Phase 16+
vector suites against the shared fixture).

The gated-module list lives in one place (PGVECTOR_GATED_MODULES) and is
listed in docs/FINAL_ACTIVATION_CHECKLIST.md.
"""

from __future__ import annotations

import pytest

# Modules whose tests intentionally exercise native pgvector on the REAL
# application PostgreSQL. Environment-gated, not disabled.
PGVECTOR_GATED_MODULES = frozenset({
    "test_vector_search.py",
    "test_hybrid_retrieval.py",
    "test_retrieval.py",
    "test_retrieval_quality.py",
    "test_message_sources.py",
})

_PGVECTOR_MISSING_REASONS = (
    'type "vector" does not exist',
    "type 'vector' does not exist",
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "env_pgvector: requires the pgvector extension on the real app "
        "PostgreSQL (skipped with an explicit reason when absent)",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.fspath.basename in PGVECTOR_GATED_MODULES:
            item.add_marker(pytest.mark.env_pgvector)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Convert the missing-pgvector infrastructure error into an explicit skip.

    Only the exact UndefinedObject error for the ``vector`` type is
    converted; every other failure propagates normally.
    """
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call" or not rep.failed:
        return
    if item.get_closest_marker("env_pgvector") is None:
        return
    longrepr = str(rep.longrepr) if rep.longrepr is not None else ""
    if any(reason in longrepr for reason in _PGVECTOR_MISSING_REASONS):
        rep.outcome = "skipped"
        # Must be the canonical (path, lineno, reason) tuple — pytest 8's
        # terminal reporter asserts tuple format for skipped reports and an
        # INTERNALERROR otherwise (found and fixed in Phase 24).
        rep.longrepr = (
            str(item.fspath),
            item.location[1],
            "ENVIRONMENT GATE (pgvector): native operator test could not run "
            "because the pgvector extension is not installed on the connected "
            "PostgreSQL. This is an environment limitation, not a product "
            "defect - the JSON fallback path is validated by the Phase 16+ "
            "vector suites. See docs/FINAL_ACTIVATION_CHECKLIST.md step 1.",
        )
