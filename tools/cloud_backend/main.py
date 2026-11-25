from fastapi import FastAPI

app = FastAPI(
    title="RAD-2 Cloud Backend",
    version="0.1.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rad2-cloud"
    }