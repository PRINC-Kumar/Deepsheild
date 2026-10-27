"""
main.py — FastAPI application entry point
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db
from backend.routes import auth, detect, history, stats
from backend.services import ml_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("DeepShield starting up…")

    # Ensure storage directories exist
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.HEATMAP_DIR).mkdir(parents=True, exist_ok=True)

    # Init database tables
    init_db()
    logger.info("Database initialised.")

    # Load ML model (once)
    ml_adapter.load_model()

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("DeepShield shutting down.")


app = FastAPI(
    title="DeepShield API",
    description="Deepfake detection API — image, video & webcam",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.APP_ENV == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(detect.router)
app.include_router(history.router)
app.include_router(stats.router)

# ── Static files ──────────────────────────────────────────────────────────────
# Serve uploaded files and heatmaps
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/heatmaps", StaticFiles(directory=settings.HEATMAP_DIR), name="heatmaps")

# Serve frontend last (catch-all)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

