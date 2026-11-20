"""
R U Socrates — FastAPI application entry point.

Start with:
        uvicorn services.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before any other imports
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .database import create_tables
from .routes.tasks import router as tasks_router
from .routes.results import router as results_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("R U Socrates API starting up…")
    create_tables()
    logger.info("Database tables ready (SQLite WAL mode)")
    if API_TOKEN:
        logger.info("API token required for /api/* requests")
    else:
        logger.warning(
            "RUS_API_TOKEN is not set — /api/* is unauthenticated. "
            "Keep the server on 127.0.0.1."
        )
    yield
    logger.info("R U Socrates API shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="R U Socrates",
    description="The transparent research engine API",
    version="0.1.0",
    lifespan=lifespan,
)

# Origins allowed to call the API. Override with a comma-separated list in
# RUS_ALLOWED_ORIGINS.
#
# The defaults cover the Next.js dev server and the Electron shell, which serves
# the static export from loopback. A file:// page would send `Origin: null` and
# be rejected here — that is why the shell uses an HTTP server.
_DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("RUS_ALLOWED_ORIGINS", ",".join(_DEFAULT_ORIGINS)).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional bearer-token auth
# ---------------------------------------------------------------------------
#
# The API is designed to run on loopback. If you expose it beyond that, set
# RUS_API_TOKEN and every client must send `Authorization: Bearer <token>`.
# The web app sends the same value from Settings → Connection.
API_TOKEN = os.getenv("RUS_API_TOKEN", "").strip()


async def require_token(
    authorization: str = Header(default=""),
    token: str = Query(default=""),
) -> None:
    """Reject unauthenticated calls once a token is configured.

    ``?token=`` exists because EventSource cannot send headers; prefer the
    Authorization header everywhere else, since query strings end up in logs.
    """
    if not API_TOKEN:
        return
    if authorization == f"Bearer {API_TOKEN}" or token == API_TOKEN:
        return
    raise HTTPException(status_code=401, detail="Missing or invalid API token")


app.include_router(tasks_router, dependencies=[Depends(require_token)])
app.include_router(results_router, dependencies=[Depends(require_token)])


@app.get("/health")
def health():
    # Unauthenticated on purpose: it carries no data and is used as a probe.
    return {"status": "ok", "service": "R U Socrates API", "auth": bool(API_TOKEN)}