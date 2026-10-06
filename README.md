# TrustLens

TrustLens is an evidence-first, multilingual, context-aware social-media scam-risk verification system.

## Running TrustLens

You can run TrustLens in three different ways depending on your workflow.

### 1. VS Code Task Execution (Recommended)
This method is the cleanest. It launches 3 separate integrated terminals in VS Code automatically (Backend, Frontend, and Tests).
1. Open the Command Palette (`Ctrl+Shift+P`)
2. Select **`Tasks: Run Task`**
3. Select **`TrustLens: Run All`**

### 2. Automated PowerShell Script
This script validates your environment and launches all 3 components automatically.
Run the following from your terminal:
```powershell
./scripts/run/run_all.ps1
```

### 3. Manual Terminal Commands
If you want to manually run individual components, open your terminal, activate your virtual environment (`.\venv\Scripts\activate`), and use the following commands:

**Run the Backend (FastAPI):**
```bash
uvicorn app.main:app --reload
```
*The backend will be available at `http://127.0.0.1:8000/api/health`*

**Run the Frontend (Streamlit):**
```bash
streamlit run frontend/streamlit_app.py
```
*The frontend will automatically open in your default browser.*

**Run the Test Suite:**
```bash
pytest -q
```

## Project Documentation
- [Architecture Audit](docs/ARCHITECTURE_AUDIT.md)
- [System Flow](docs/SYSTEM_FLOW.md)
- [Model Inventory](docs/MODEL_INVENTORY.md)
