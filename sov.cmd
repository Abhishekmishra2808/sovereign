@echo off
setlocal
set PYTHONPATH=%~dp0
python -m agent.cli %*
