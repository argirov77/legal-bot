from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI(title="Legal Bot API")


@app.get("/", response_class=PlainTextResponse)
def read_root() -> str:
    """Healthcheck endpoint for the service."""
    return "ok"
