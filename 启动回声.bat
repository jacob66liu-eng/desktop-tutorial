@echo off
cd /d "%~dp0"
set OLLAMA_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=qwen3:8b
start "Jacob 回声助手服务" /min python server.py
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8765
