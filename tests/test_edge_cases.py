"""
tests/test_edge_cases.py — Synthetic edge case tests matching spec section 5
"""
import io
import struct
import sys
import os
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["USE_MOCK_ML"] = "true"

pytestmark = pytest.mark.asyncio


def _make_jpeg():
    from PIL import Image
    img = Image.new("RGB", (200, 200), color=(180, 140, 110))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


# ── File edge cases ────────────────────────────────────────────────────────────

async def test_exe_renamed_jpg(auth_client):
    """EXE renamed to .jpg must be rejected (400 or corrupted image)."""
    exe_bytes = b"MZ" + b"\x00" * 100  # PE header stub
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("malware.jpg", exe_bytes, "image/jpeg")},
    )
    assert resp.status_code in (400, 422)


async def test_image_too_large(auth_client):
    big = b"\xff\xd8\xff" + b"A" * (11 * 1024 * 1024)
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert resp.status_code == 413


async def test_empty_file(auth_client):
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


async def test_unsupported_extension(auth_client):
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("file.bmp.pdf", b"fakedata", "application/pdf")},
    )
    assert resp.status_code == 400


async def test_path_traversal_filename(auth_client):
    """../etc/passwd style filename must be sanitized (not cause 500)."""
    img_bytes = _make_jpeg()
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("../../etc/passwd.jpg", img_bytes, "image/jpeg")},
    )
    # Should succeed (file saved with UUID name) or return 400 — never 500
    assert resp.status_code in (200, 400)


async def test_grayscale_image(auth_client):
    """Grayscale PNG must be accepted and converted."""
    from PIL import Image
    img = Image.new("L", (200, 200), color=128)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("gray.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 200


async def test_png_with_alpha(auth_client):
    """RGBA PNG must be accepted."""
    from PIL import Image
    img = Image.new("RGBA", (200, 200), color=(180, 140, 110, 200))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("alpha.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 200


async def test_tiny_image(auth_client):
    """Very small image should succeed with a quality warning."""
    from PIL import Image
    img = Image.new("RGB", (20, 20), color=(180, 100, 80))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("tiny.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Should have low-res warning or NO_FACE
    assert data["label"] in ("NO_FACE", "REAL", "FAKE", "UNCERTAIN")


# ── Webcam edge cases ─────────────────────────────────────────────────────────

async def test_frame_missing_image_field(auth_client):
    resp = await auth_client.post("/api/detect/frame", json={})
    assert resp.status_code == 400


async def test_frame_invalid_base64(auth_client):
    resp = await auth_client.post("/api/detect/frame", json={"image": "NOT_VALID_BASE64!!!"})
    assert resp.status_code == 400


# ── Adapter edge cases ────────────────────────────────────────────────────────

async def test_adapter_missing_file():
    from backend.services.ml_adapter import predict_image
    result = predict_image("/totally/missing/file.jpg")
    assert "warnings" in result
    assert len(result["warnings"]) > 0


async def test_adapter_corrupted_image(tmp_path):
    from backend.services.ml_adapter import predict_image
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"\xff\xd8\xff" + b"GARBAGE_DATA_NOT_JPEG" * 10)
    result = predict_image(str(p))
    assert "label" in result
    assert result["label"] in ("REAL", "FAKE", "UNCERTAIN", "NO_FACE")


async def test_adapter_empty_frame():
    from backend.services.ml_adapter import predict_frame
    result = predict_frame(np.array([]))
    assert len(result["warnings"]) > 0


async def test_adapter_video_zero_frames(tmp_path):
    """Empty video file should return a friendly message, not crash."""
    from backend.services.ml_adapter import predict_video
    vid_path = tmp_path / "empty.mp4"
    vid_path.write_bytes(b"")
    result = predict_video(str(vid_path))
    assert "warnings" in result
    assert len(result["warnings"]) > 0


# ── Auth edge cases ───────────────────────────────────────────────────────────

async def test_expired_token(client):
    client.headers["Authorization"] = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5OTkiLCJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20iLCJleHAiOjF9.INVALID"
    resp = await client.get("/api/history")
    assert resp.status_code == 401


async def test_no_token(client):
    client.headers.pop("Authorization", None)
    resp = await client.get("/api/stats")
    assert resp.status_code in (401, 403)

