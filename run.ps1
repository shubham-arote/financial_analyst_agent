# run.ps1 — set up the venv (first run) and launch the Live SRR Document Engine.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}

if (-not (Test-Path "samples\report_page.png")) {
    Write-Host "Generating sample report..." -ForegroundColor Cyan
    & $py samples\make_sample.py
}

Write-Host "`nServer starting -> http://127.0.0.1:8000  (Ctrl+C to stop)`n" -ForegroundColor Green
& $py -m uvicorn app.server:app --host 127.0.0.1 --port 8000
