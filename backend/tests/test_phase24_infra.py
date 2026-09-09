"""Phase 24 tests — test infrastructure + unified error model (Steps 1, 31, 107).

Covers: the conftest pgvector environment gate (marker present, non-pgvector
failures NOT masked, gate list integrity), the validation runner's exit-code
propagation contract, and the unified error model (structured envelope,
legacy compatibility, validation errors, correlation IDs).
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

from fastapi.testclient import TestClient

from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

from app.core.database import get_db  # noqa: E402
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402

app.dependency_overrides[get_db] = override_get_db

BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


_counter = [0]


def _mkuser(db):
    _counter[0] += 1
    u = User(email=f"p24i{_counter[0]}@example", name="u", password_hash="x")
    db.add(u)
    db.commit()
    return u


# ===========================================================================
# Conftest pgvector environment gate (Step 1 / 2 integrity)
# ===========================================================================


class TestConftestGate:
    def test_gated_modules_get_marker(self, pytester):
        pytester.makeconftest(
            """
            import pytest
            PGVECTOR_GATED_MODULES = frozenset({"test_gated.py"})
            def pytest_collection_modifyitems(config, items):
                for item in items:
                    if item.fspath.basename in PGVECTOR_GATED_MODULES:
                        item.add_marker(pytest.mark.env_pgvector)
            """
        )
        pytester.makepyfile(
            test_gated="""
            import pytest

            def test_marked(request):
                assert request.node.get_closest_marker("env_pgvector") is not None
            """,
            test_ungated="""
            def test_unmarked(request):
                assert request.node.get_closest_marker("env_pgvector") is None
            """,
        )
        result = pytester.runpytest_subprocess("-q")
        result.assert_outcomes(passed=2)

    def test_pgvector_missing_error_is_gated(self, pytester):
        """The exact missing-pgvector error becomes a skip with a reason."""
        pytester.makeconftest(
            """
            import pytest
            PGVECTOR_GATED_MODULES = frozenset({"test_gate_demo.py"})
            _PGVECTOR_MISSING_REASONS = ('type "vector" does not exist',)

            def pytest_collection_modifyitems(config, items):
                for item in items:
                    if item.fspath.basename in PGVECTOR_GATED_MODULES:
                        item.add_marker(pytest.mark.env_pgvector)

            @pytest.hookimpl(hookwrapper=True, tryfirst=True)
            def pytest_runtest_makereport(item, call):
                outcome = yield
                rep = outcome.get_result()
                if rep.when != "call" or not rep.failed:
                    return
                if item.get_closest_marker("env_pgvector") is None:
                    return
                longrepr = str(rep.longrepr) if rep.longrepr is not None else ""
                if any(r in longrepr for r in _PGVECTOR_MISSING_REASONS):
                    rep.outcome = "skipped"
                    rep.longrepr = "ENVIRONMENT GATE (pgvector): not installed"
            """
        )
        pytester.makepyfile(
            test_gate_demo="""
            import pytest

            def test_vector_missing():
                raise TypeError('type "vector" does not exist')

            def test_other_failure():
                assert 1 == 2, "must NOT be masked"
            """,
            test_ungated_fail="""
            def test_no_marker_failure():
                raise TypeError('type "vector" does not exist')
            """,
        )
        result = pytester.runpytest_subprocess("-q")
        outcomes = result.parseoutcomes()
        # gated+missing -> skipped; other assertion -> failed; unmarked -> failed
        assert outcomes.get("skipped", 0) == 1
        assert outcomes.get("failed", 0) == 2

    def test_real_modules_are_in_gate_list(self):
        from tests.conftest import PGVECTOR_GATED_MODULES

        for name in (
            "test_vector_search.py",
            "test_hybrid_retrieval.py",
            "test_retrieval.py",
            "test_retrieval_quality.py",
            "test_message_sources.py",
        ):
            assert name in PGVECTOR_GATED_MODULES

    def test_gate_list_files_exist(self):
        from tests.conftest import PGVECTOR_GATED_MODULES

        tests_dir = BACKEND_DIR / "tests"
        for name in PGVECTOR_GATED_MODULES:
            assert (tests_dir / name).exists(), name


# ===========================================================================
# Validation runner exit-code propagation (Step 1 / Step 125)
# ===========================================================================


@pytest.mark.skipif(
    sys.platform.startswith("win") is False and not Path("/usr/bin/env").exists(),
    reason="runner contract validated via bash",
)
class TestRunnerExitCode:
    """The reusable runner must propagate pytest's exit code."""

    def _run(self, *args):
        bash = "C:/Program Files/Git/bin/bash.exe" if sys.platform == "win32" else "/usr/bin/bash"
        proc = subprocess.run(
            [bash, str(BACKEND_DIR / "run_tests.sh"), *args],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=BACKEND_DIR,
        )
        return proc.returncode

    def test_pass_yields_zero(self):
        rc = self._run(
            "tests/runner_probe/probe_pass.py",
            "-p", "no:cacheprovider", "-q", "--no-header",
        )
        assert rc == 0, "runner must propagate pytest success as exit 0"

    def test_failure_yields_nonzero(self):
        rc = self._run(
            "tests/runner_probe/probe_fail.py",
            "-p", "no:cacheprovider", "-q",
        )
        assert rc != 0, "runner must propagate pytest failure as nonzero exit"

    def test_deselected_yields_five(self):
        rc = self._run(
            "tests/runner_probe/probe_pass.py", "-k", "definitely_no_such_test",
            "-p", "no:cacheprovider", "-q",
        )
        assert rc == 5, "deselected/no-tests run must surface exit 5"


# ===========================================================================
# Unified error model (Step 31)
# ===========================================================================


class TestUnifiedErrorModel:
    def test_401_has_structured_envelope(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 401
        body = resp.json()
        assert body["detail"]
        assert body["error"]["code"]
        assert body["error"]["message"]

    def test_legacy_detail_preserved(self, client):
        resp = client.get("/workspaces")
        assert resp.status_code == 401
        assert isinstance(resp.json()["detail"], str)

    def test_validation_error_is_structured_422(self, client):
        resp = client.post("/auth/register", json={"name": 12345})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "validation_error"
        assert "validation failed" in body["error"]["message"].lower()

    def test_malformed_json_is_422_not_500(self, client):
        resp = client.post(
            "/auth/register",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, "malformed JSON must be structured 422"

    def test_correlation_id_in_error_envelope(self, client):
        resp = client.get("/documents", headers={"X-Request-ID": "corr-p24-1"})
        assert resp.status_code == 401
        assert resp.headers.get("X-Request-ID") == "corr-p24-1"
        assert resp.json()["error"].get("request_id") == "corr-p24-1"

    def test_no_stack_trace_in_client_errors(self, client):
        resp = client.get("/documents")
        text = resp.text.lower()
        assert "traceback" not in text
        assert ".py" not in text or "error" not in text

    def test_domain_error_shape(self):
        from app.core.errors import DomainError

        e = DomainError("conflict", "already exists")
        assert e.status_code == 409
        assert e.code == "conflict"
        e2 = DomainError("custom_kind", "boom", status_code=418)
        assert e2.status_code == 418
