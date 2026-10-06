from fastapi import FastAPI

app = FastAPI(title="TrustLens API")

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
