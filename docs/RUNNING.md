# Running TrustLens

## VS Code Task Execution (Recommended)
To run all subsystems simultaneously, use the integrated VS Code Tasks system:
1. Open Command Palette (`Ctrl+Shift+P`)
2. `Tasks: Run Task`
3. Select `TrustLens: Run All`

This will automatically launch the Backend, the Streamlit Frontend, and run the Pytest suite in parallel integrated terminals.

## PowerShell Execution
Alternatively, you can run the bootstrap script from PowerShell:
```powershell
./scripts/run/run_all.ps1
```

## Manual Execution
- **Backend**: `uvicorn app.main:app --reload`
- **Frontend**: `streamlit run frontend/streamlit_app.py`
- **Tests**: `pytest -q`
