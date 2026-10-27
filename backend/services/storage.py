"""
services/storage.py — File storage, validation, hashing, thumbnail generation
"""
from __future__ import annotations

import hashlib
import io
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.config import settings

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
ALLOWED_IMAGE_MIMES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"
}
ALLOWED_VIDEO_MIMES = {
    "video/mp4", "video/x-msvideo", "video/quicktime",
    "video/x-matroska", "video/webm", "video/x-flv"
}

# Simple magic-byte sniffers
IMAGE_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",  # Simplified; real check needs offset 8
    b"BM": "image/bmp",
}


def _safe_filename(original: str) -> str:
    """Return a UUID-based filename preserving the sanitised extension."""
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS:
        suffix = ".bin"
    return f"{uuid.uuid4()}{suffix}"


def _compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sniff_mime(header: bytes) -> str | None:
    for magic, mime in IMAGE_MAGIC.items():
        if header.startswith(magic):
            return mime
    return None


async def validate_and_save_image(file: UploadFile) -> tuple[Path, str, int]:
    """
    Validate image upload, save with UUID name.
    Returns (saved_path, file_hash, file_size_bytes).
    Raises HTTPException on any validation failure.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported image extension '{ext}'. "
            f"Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    data = await file.read()
    size = len(data)

    if size == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    if size > settings.max_image_bytes:
        raise HTTPException(
            413,
            f"Image too large ({size // 1_048_576} MB). "
            f"Maximum: {settings.MAX_IMAGE_SIZE_MB} MB.",
        )

    # MIME sniff
    sniffed = _sniff_mime(data[:16])
    declared = file.content_type or ""
    if (
        sniffed
        and declared
        and declared not in ALLOWED_IMAGE_MIMES
        and "octet-stream" not in declared
    ):
        # Don't block if sniffed doesn't match declared — just warn
        pass

    # Try opening with PIL to confirm it's a valid image
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(400, f"File is not a valid image: {e}")

    file_hash = _compute_hash(data)
    filename = _safe_filename(file.filename or "upload.jpg")
    save_path = settings.upload_path / filename
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(data)

    return save_path, file_hash, size


async def validate_and_save_video(file: UploadFile) -> tuple[Path, str, int]:
    """
    Validate video upload, save with UUID name.
    Returns (saved_path, file_hash, file_size_bytes).
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported video extension '{ext}'. "
            f"Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    data = await file.read()
    size = len(data)

    if size == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    if size > settings.max_video_bytes:
        raise HTTPException(
            413,
            f"Video too large ({size // 1_048_576} MB). "
            f"Maximum: {settings.MAX_VIDEO_SIZE_MB} MB.",
        )

    file_hash = _compute_hash(data)
    filename = _safe_filename(file.filename or "upload.mp4")
    save_path = settings.upload_path / filename
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(data)

    # Verify OpenCV can open it
    cap = cv2.VideoCapture(str(save_path))
    if not cap.isOpened():
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, "Cannot open video — file may be corrupted.")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frames == 0:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, "Video has zero frames.")

    return save_path, file_hash, size


def generate_thumbnail(image_path: Path, size: tuple[int, int] = (200, 200)) -> str | None:
    """Create a thumbnail and return its path string."""
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(size)
        thumb_name = f"thumb_{image_path.stem}.jpg"
        thumb_path = settings.upload_path / thumb_name
        img.save(thumb_path, "JPEG", quality=70)
        return str(thumb_path)
    except Exception as e:
        logger.warning(f"Thumbnail generation failed: {e}")
        return None


def generate_video_thumbnail(video_path: Path, size: tuple[int, int] = (200, 200)) -> str | None:
    """Extract first face frame as thumbnail."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail(size)
        thumb_name = f"thumb_{video_path.stem}.jpg"
        thumb_path = settings.upload_path / thumb_name
        img.save(thumb_path, "JPEG", quality=70)
        return str(thumb_path)
    except Exception as e:
        logger.warning(f"Video thumbnail generation failed: {e}")
        return None

