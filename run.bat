@echo off
rem Starts the backend (:8000) and the viewer (:5173) in their own windows, then opens the viewer.
rem Close those windows to stop the servers.
cd /d "%~dp0"
start "playground backend" cmd /k uv run uvicorn backend.app:app --port 8000
start "playground frontend" cmd /k "cd frontend && corepack pnpm install --frozen-lockfile && corepack pnpm dev"
timeout /t 8 /nobreak >nul
start "" http://localhost:5173
