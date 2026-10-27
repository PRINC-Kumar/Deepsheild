"""
routes/detect.py — Image, video and webcam-frame detection endpoints.
Rate-limited; saves every scan to history.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.models.scan import ScanHistory
from backend.models.user import User
from backend.routes.deps import get_current_user
from backend.schemas.scan import ScanResult
from backend.services import ml_adapter
from backend.services.storage import (
    generate_thumbnail,
    generate_video_thumbnail,
    validate_and_save_image,
    validate_and_save_video,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/detect", tags=["detect"])

# ── Simple in-memory rate limiter ─────────────────────────────────────────────
_rate_store: dict[int, list[float]] = defaultdict(list)


def _check_rate_limit(user_id: int) -> None:
    now = time.time()
    window = 60  # seconds
    limit = settings.RATE_LIMIT_DETECT_PER_MINUTE
    calls = [t for t in _rate_store[user_id] if now - t < window]
    if len(calls) >= limit:
        raise HTTPException(
            429,
            f"Rate limit exceeded: max {limit} detections per minute.",
        )
    calls.append(now)
    _rate_store[user_id] = calls


def _save_scan(
    db: Session,
    user: User,
    result: dict,
    original_filename: str,
    saved_filename: str,
    file_type: str,
    file_hash: str | None,
    file_size: int | None,
    thumbnail_path: str | None,
) -> ScanHistory:
    scan = ScanHistory(
        user_id=user.id,
        filename=saved_filename,
        original_filename=original_filename,
        file_hash=file_hash,
        file_type=file_type,
        file_size_bytes=file_size,
        thumbnail_path=thumbnail_path,
        label=result["label"],
        confidence=result["confidence"],
        fake_probability=result["fake_probability"],
        faces_detected=result["faces_detected"],
        heatmap_path=result.get("heatmap_path"),
        frame_scores=json.dumps(result.get("frame_scores", [])),
        processing_time_ms=result["processing_time_ms"],
        warnings=json.dumps(result.get("warnings", [])),
    )
    db.add(scan)
    return scan


# ── Duplicate-hash helper ─────────────────────────────────────────────────────
def _find_cached(db: Session, user_id: int, file_hash: str) -> ScanHistory | None:
    from sqlalchemy import select
    result = db.execute(
        select(ScanHistory)
        .where(ScanHistory.user_id == user_id, ScanHistory.file_hash == file_hash)
        .order_by(ScanHistory.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ─────────────────────────────────────────────────────────────────────────────
@router.post("/image", response_model=ScanResult)
async def detect_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_rate_limit(user.id)

    save_path, file_hash, file_size = await validate_and_save_image(file)

    # Duplicate check
    cached = _find_cached(db, user.id, file_hash)
    if cached:
        logger.info(f"Returning cached result for hash {file_hash}")
        return ScanResult(
            label=cached.label,
            confidence=cached.confidence,
            fake_probability=cached.fake_probability,
            faces_detected=cached.faces_detected,
            heatmap_path=cached.heatmap_path,
            frame_scores=json.loads(cached.frame_scores or "[]"),
            processing_time_ms=cached.processing_time_ms,
            warnings=json.loads(cached.warnings or "[]")
            + ["Cached result — identical file uploaded previously."],
        )

    result = ml_adapter.predict_image(str(save_path))
    thumbnail = generate_thumbnail(save_path)

    _save_scan(
        db,
        user,
        result,
        original_filename=file.filename or "upload",
        saved_filename=save_path.name,
        file_type="image",
        file_hash=file_hash,
        file_size=file_size,
        thumbnail_path=thumbnail,
    )

    return ScanResult(**result)


@router.post("/video", response_model=ScanResult)
async def detect_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_rate_limit(user.id)

    save_path, file_hash, file_size = await validate_and_save_video(file)

    cached = _find_cached(db, user.id, file_hash)
    if cached:
        return ScanResult(
            label=cached.label,
            confidence=cached.confidence,
            fake_probability=cached.fake_probability,
            faces_detected=cached.faces_detected,
            heatmap_path=cached.heatmap_path,
            frame_scores=json.loads(cached.frame_scores or "[]"),
            processing_time_ms=cached.processing_time_ms,
            warnings=json.loads(cached.warnings or "[]")
            + ["Cached result — identical file uploaded previously."],
        )

    result = ml_adapter.predict_video(str(save_path))
    thumbnail = generate_video_thumbnail(save_path)

    _save_scan(
        db,
        user,
        result,
        original_filename=file.filename or "upload",
        saved_filename=save_path.name,
        file_type="video",
        file_hash=file_hash,
        file_size=file_size,
        thumbnail_path=thumbnail,
    )

    return ScanResult(**result)


@router.post("/frame", response_model=ScanResult)
async def detect_frame(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Accepts JSON: {"image": "<base64-encoded JPEG>"}
    Used by the webcam page.
    """
    _check_rate_limit(user.id)

    body = await request.json()
    b64 = body.get("image", "")
    if not b64:
        raise HTTPException(400, "Missing 'image' field in request body.")

    # Strip data URL prefix if present
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        import cv2
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise HTTPException(400, "Could not decode image frame.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Could not decode image frame.")

    result = ml_adapter.predict_frame(frame_bgr)
    # Webcam frames are NOT saved to history (privacy)
    return ScanResult(**result)

