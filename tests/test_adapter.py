"""
tests/test_adapter.py — ML Adapter contract tests (MOCK mode)
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["USE_MOCK_ML"] = "true"  # force mock mode

from backend.services import ml_adapter

REQUIRED_KEYS = {"label", "confidence", "fake_probability", "faces_detected",
                 "heatmap_path", "frame_scores", "processing_time_ms", "warnings"}
VALID_LABELS = {"REAL", "FAKE", "UNCERTAIN", "NO_FACE"}


def assert_contract(result: dict):
    """Assert all required keys exist and values are within valid ranges."""
    assert REQUIRED_KEYS.issubset(result.keys()), f"Missing keys: {REQUIRED_KEYS - result.keys()}"
    assert result["label"] in VALID_LABELS, f"Invalid label: {result['label']}"
    assert 0.0 <= result["confidence"] <= 1.0, f"Confidence out of range: {result['confidence']}"
    assert 0.0 <= result["fake_probability"] <= 1.0, f"fake_probability out of range: {result['fake_probability']}"
    assert isinstance(result["faces_detected"], int)
    assert isinstance(result["frame_scores"], list)
    assert isinstance(result["warnings"], list)
    assert isinstance(result["processing_time_ms"], int)
    assert result["processing_time_ms"] >= 0


def setup_module():
    ml_adapter.load_model()


# ─── Image tests ──────────────────────────────────────────────────────────────

def test_predict_image_real(tmp_path):
    """Real-looking image (solid color, simulated face)."""
    from PIL import Image, ImageDraw
    img_path = tmp_path / "real_face.jpg"
    img = Image.new("RGB", (224, 224), color=(200, 160, 130))
    draw = ImageDraw.Draw(img)
    draw.ellipse([70, 60, 154, 164], fill=(210, 170, 140))  # rough face oval
    img.save(img_path)
    result = ml_adapter.predict_image(str(img_path))
    assert_contract(result)


def test_predict_image_fake(tmp_path):
    """Image that looks artificial — adapter should return valid contract."""
    import cv2
    img_path = tmp_path / "fake_face.jpg"
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), arr)
    result = ml_adapter.predict_image(str(img_path))
    assert_contract(result)


def test_predict_image_no_face(tmp_path):
    """Solid color image — no face should be detected."""
    from PIL import Image
    img_path = tmp_path / "no_face.jpg"
    Image.new("RGB", (224, 224), color=(50, 100, 200)).save(img_path)
    result = ml_adapter.predict_image(str(img_path))
    assert_contract(result)
    assert result["label"] == "NO_FACE"
    assert result["faces_detected"] == 0


def test_predict_image_corrupted(tmp_path):
    """Corrupted file — adapter must not raise, must return result with warning."""
    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_bytes(b"\xff\xd8\xff\xe0JUNK_DATA_GARBAGE")
    result = ml_adapter.predict_image(str(bad_path))
    assert_contract(result)
    # Should either warn or return NO_FACE
    has_warning = len(result["warnings"]) > 0 or result["label"] == "NO_FACE"
    assert has_warning, "Corrupted file should produce a warning or NO_FACE"


def test_predict_image_missing_file():
    result = ml_adapter.predict_image("/nonexistent/path/image.jpg")
    assert_contract(result)
    assert len(result["warnings"]) > 0


def test_predict_image_probabilities_in_range(tmp_path):
    """Run 5 predictions, all probabilities must be 0–1."""
    from PIL import Image
    for i in range(5):
        img_path = tmp_path / f"test_{i}.jpg"
        Image.new("RGB", (224, 224), color=(i*40, 80, 120)).save(img_path)
        result = ml_adapter.predict_image(str(img_path))
        assert 0.0 <= result["fake_probability"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0


# ─── Video test ───────────────────────────────────────────────────────────────

def test_predict_video_short(tmp_path):
    """Short synthetic video (10 frames)."""
    import cv2
    vid_path = tmp_path / "test_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 10, (224, 224))
    for _ in range(10):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    result = ml_adapter.predict_video(str(vid_path))
    assert_contract(result)


def test_predict_video_missing():
    result = ml_adapter.predict_video("/nonexistent/video.mp4")
    assert_contract(result)
    assert len(result["warnings"]) > 0


# ─── Frame test ───────────────────────────────────────────────────────────────

def test_predict_frame_bgr():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = ml_adapter.predict_frame(frame)
    assert_contract(result)
    assert result["heatmap_path"] is None  # heatmap not generated for frames


def test_predict_frame_empty():
    result = ml_adapter.predict_frame(np.array([]))
    assert_contract(result)
    assert len(result["warnings"]) > 0


# ─── Multi-face test ──────────────────────────────────────────────────────────

def test_predict_image_multiface(tmp_path):
    """Multi-face image: adapter returns valid contract."""
    from PIL import Image, ImageDraw
    img_path = tmp_path / "multiface.jpg"
    img = Image.new("RGB", (640, 480), color=(180, 140, 110))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 100, 150, 250], fill=(200, 160, 130))
    draw.ellipse([300, 100, 400, 250], fill=(200, 160, 130))
    img.save(img_path)
    result = ml_adapter.predict_image(str(img_path))
    assert_contract(result)

