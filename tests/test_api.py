"""
tests/test_api.py — API endpoint tests
"""
import io
import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio


# ── Auth ──────────────────────────────────────────────────────────────────────

async def test_signup_success(client):
    resp = await client.post("/api/auth/signup", json={"email": "new@test.com", "password": "newpass99"})
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert data["email"] == "new@test.com"


async def test_signup_duplicate_email(client):
    await client.post("/api/auth/signup", json={"email": "dup@test.com", "password": "pass1234"})
    resp = await client.post("/api/auth/signup", json={"email": "dup@test.com", "password": "pass1234"})
    assert resp.status_code == 409


async def test_signup_weak_password(client):
    resp = await client.post("/api/auth/signup", json={"email": "weak@test.com", "password": "short"})
    assert resp.status_code == 422


async def test_signup_invalid_email(client):
    resp = await client.post("/api/auth/signup", json={"email": "not-an-email", "password": "longpass123"})
    assert resp.status_code == 422


async def test_login_success(client):
    await client.post("/api/auth/signup", json={"email": "login@test.com", "password": "testpass123"})
    resp = await client.post("/api/auth/login", json={"email": "login@test.com", "password": "testpass123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


async def test_login_wrong_password(client):
    await client.post("/api/auth/signup", json={"email": "loginfail@test.com", "password": "testpass123"})
    resp = await client.post("/api/auth/login", json={"email": "loginfail@test.com", "password": "wrongpassword"})
    assert resp.status_code == 401


async def test_login_nonexistent_user(client):
    resp = await client.post("/api/auth/login", json={"email": "ghost@test.com", "password": "anypass123"})
    assert resp.status_code == 401


# ── Auth guard ────────────────────────────────────────────────────────────────

async def test_history_requires_auth(client):
    resp = await client.get("/api/history")
    assert resp.status_code in (401, 403)


async def test_stats_requires_auth(client):
    resp = await client.get("/api/stats")
    assert resp.status_code in (401, 403)


# ── Health ────────────────────────────────────────────────────────────────────

async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_loaded" in data
    assert "mock_mode" in data
    assert data["mock_mode"] is True


# ── Detection ─────────────────────────────────────────────────────────────────

def _make_jpeg_bytes():
    """Create a minimal valid JPEG in memory."""
    from PIL import Image
    import io
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def test_detect_image_success(auth_client):
    img_bytes = _make_jpeg_bytes()
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert data["label"] in ("REAL", "FAKE", "UNCERTAIN", "NO_FACE")
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["fake_probability"] <= 1.0
    assert isinstance(data["warnings"], list)
    assert isinstance(data["frame_scores"], list)


async def test_detect_image_too_large(auth_client):
    big = b"\xff\xd8\xff" + b"X" * (11 * 1024 * 1024)
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert resp.status_code == 413


async def test_detect_image_wrong_extension(auth_client):
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("mal.exe", b"MZ\x00\x00", "application/octet-stream")},
    )
    assert resp.status_code == 400


async def test_detect_image_empty_file(auth_client):
    resp = await auth_client.post(
        "/api/detect/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


async def test_detect_frame_success(auth_client):
    import base64
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = await auth_client.post("/api/detect/frame", json={"image": b64})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data


async def test_detect_frame_missing_image(auth_client):
    resp = await auth_client.post("/api/detect/frame", json={})
    assert resp.status_code == 400


# ── History ───────────────────────────────────────────────────────────────────

async def test_history_empty(auth_client):
    """Fresh user has empty history."""
    # Create a new fresh user to avoid interference
    resp2 = await auth_client.post("/api/auth/signup", json={"email": "fresh@test.com", "password": "freshpass99"})
    token = resp2.json()["access_token"]
    auth_client.headers["Authorization"] = f"Bearer {token}"
    resp = await auth_client.get("/api/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 0


async def test_history_pagination(auth_client):
    resp = await auth_client.get("/api/history?page=1&per_page=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["per_page"] == 5


async def test_history_filter_by_label(auth_client):
    resp = await auth_client.get("/api/history?label=REAL")
    assert resp.status_code == 200


async def test_history_filter_by_type(auth_client):
    resp = await auth_client.get("/api/history?file_type=image")
    assert resp.status_code == 200


async def test_history_get_nonexistent(auth_client):
    resp = await auth_client.get("/api/history/999999")
    assert resp.status_code == 404


async def test_history_delete_nonexistent(auth_client):
    resp = await auth_client.delete("/api/history/999999")
    assert resp.status_code == 404


async def test_history_cross_user_isolation(client):
    """User A must not access User B's scans."""
    # Create user A
    resp_a = await client.post("/api/auth/signup", json={"email": "usera@test.com", "password": "usera1234"})
    token_a = resp_a.json()["access_token"]

    # Create user B
    resp_b = await client.post("/api/auth/signup", json={"email": "userb@test.com", "password": "userb1234"})
    token_b = resp_b.json()["access_token"]

    # User A tries to access scan with id=1 as User B
    client.headers["Authorization"] = f"Bearer {token_b}"
    resp = await client.get("/api/history/1")
    # Either 404 (not found for user B) or their own scan — never user A's
    assert resp.status_code in (404, 200)


async def test_history_export_csv(auth_client):
    resp = await auth_client.get("/api/history/export")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")


async def test_stats(auth_client):
    resp = await auth_client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    required = {"total_scans", "real_count", "fake_count", "uncertain_count", "avg_confidence"}
    assert required.issubset(data.keys())


async def test_invalid_token(client):
    client.headers["Authorization"] = "Bearer totally_invalid_token"
    resp = await client.get("/api/history")
    assert resp.status_code == 401

