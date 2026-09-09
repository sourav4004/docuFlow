@echo off
rem ============================================================
rem DocuFlow test validation runner (Phase 24, Step 1)
rem Guarantees pytest's exit code is the script's exit code.
rem Output is written to .testreports\last_run.txt and then
rem displayed; the exit code comes directly from pytest.
rem
rem Usage:
rem   run_tests.bat                 full suite
rem   run_tests.bat tests\test_auth.py ...
rem ============================================================
setlocal
cd /d "%~dp0"

set PY=python
if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe

if not exist ".testreports" mkdir ".testreports"

%PY% -m pytest %* --tb=short > ".testreports\last_run.txt" 2>&1
set RC=%ERRORLEVEL%
type ".testreports\last_run.txt"

endlocal & exit /b %RC%
