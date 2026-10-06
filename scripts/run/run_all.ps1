# TrustLens Run All Script

Write-Host "Starting TrustLens..."

if (-Not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    exit 1
}

# Backend
Write-Host "Starting Backend..."
Start-Process -NoNewWindow -FilePath "venv\Scripts\uvicorn.exe" -ArgumentList "app.main:app", "--reload"

# Frontend
Write-Host "Starting Frontend..."
Start-Process -NoNewWindow -FilePath "venv\Scripts\streamlit.exe" -ArgumentList "run", "frontend/streamlit_app.py"

# Tests
Write-Host "Running Tests..."
& "venv\Scripts\pytest.exe" -q
