from fastapi import FastAPI
from app.api.routes import input as routes_input

app = FastAPI(title="TrustLens API")

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok"}

app.include_router(routes_input.router, prefix="/api")

