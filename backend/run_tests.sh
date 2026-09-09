#!/usr/bin/env bash
# ============================================================
# DocuFlow test validation runner (Phase 24, Step 1)
#
# Guarantees pytest's exit code is the script's exit code.
# Filtered output (tail/grep) must NEVER mask a failure: the
# full log is tee'd to .testreports/last_run.txt and the exit
# code comes from pytest via pipefail — never from the filter.
#
# Usage:
#   ./run_tests.sh                          full suite
#   ./run_tests.sh tests/test_auth.py ...   specific files
#   ./run_tests.sh -k "isolation"           pytest arg passthrough
#
# Exit code: pytest's exit code, always. CI-correct.
# ============================================================
set -u
cd "$(dirname "$0")"

PY="${PYTHON_BIN:-python}"

# Reuse the committed virtualenv when present (Windows checkout layout).
if [ -z "${PYTHON_BIN:-}" ] && [ -x "venv/Scripts/python.exe" ]; then
  PY="venv/Scripts/python.exe"
elif [ -z "${PYTHON_BIN:-}" ] && [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
fi

mkdir -p .testreports

set -o pipefail
"$PY" -m pytest "$@" ${PYTEST_ADDOPTS:-} --tb=short 2>&1 | tee .testreports/last_run.txt
exit "${PIPESTATUS[0]}"
