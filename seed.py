"""
seed.py — Create demo user + 20 sample history records
Run: python seed.py
"""
import json
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select
from backend.config import settings
from backend.database import Base, SessionLocal, engine, init_db
from backend.models.user import User
from backend.models.scan import ScanHistory
from backend.services.auth_service import hash_password

DEMO_EMAIL = "demo@deepshield.com"
DEMO_PASSWORD = "demo1234"

LABELS = ["REAL", "REAL", "REAL", "FAKE", "FAKE", "UNCERTAIN", "NO_FACE"]
FILE_TYPES = ["image", "image", "image", "video"]
FILENAMES = [
    "portrait_001.jpg", "face_scan.png", "interview_clip.mp4",
    "social_media_pic.jpg", "news_anchor.webp", "celebrity_video.mp4",
    "profile_photo.jpg", "zoom_screenshot.png", "tiktok_clip.mp4",
    "passport_photo.jpg",
]


def seed():
    init_db()

    with SessionLocal() as db:
        # Create demo user if not exists
        result = db.execute(select(User).where(User.email == DEMO_EMAIL))
        user = result.scalar_one_or_none()

        if not user:
            user = User(email=DEMO_EMAIL, hashed_password=hash_password(DEMO_PASSWORD))
            db.add(user)
            db.flush()
            print(f"[OK] Created demo user: {DEMO_EMAIL} / {DEMO_PASSWORD}")
        else:
            print(f"[INFO] Demo user already exists: {DEMO_EMAIL}")

        # Create 20 sample scans
        count = 0
        for i in range(20):
            label = random.choice(LABELS)
            fake_prob = {
                "REAL": random.uniform(0.0, 0.35),
                "FAKE": random.uniform(0.65, 0.99),
                "UNCERTAIN": random.uniform(0.4, 0.6),
                "NO_FACE": 0.5,
            }[label]
            confidence = abs(fake_prob - 0.5) * 2 if label == "UNCERTAIN" else (fake_prob if label == "FAKE" else 1 - fake_prob)
            file_type = random.choice(FILE_TYPES)
            filename = random.choice(FILENAMES)
            frame_scores = []
            if file_type == "video":
                frame_scores = [{"frame": j*10, "timestamp_s": j*0.4, "fake_probability": round(random.uniform(0,1),3)} for j in range(15)]

            scan = ScanHistory(
                user_id=user.id,
                filename=f"seed_{i:03d}_{filename}",
                original_filename=filename,
                file_hash=f"seed_hash_{i:040x}",
                file_type=file_type,
                file_size_bytes=random.randint(50_000, 5_000_000),
                thumbnail_path=None,
                label=label,
                confidence=round(confidence, 4),
                fake_probability=round(fake_prob, 4),
                faces_detected=random.randint(0 if label == "NO_FACE" else 1, 3),
                heatmap_path=None,
                frame_scores=json.dumps(frame_scores),
                processing_time_ms=random.randint(200, 2500),
                warnings=json.dumps(["Low image quality detected — result may be less reliable."] if random.random() < 0.3 else []),
                created_at=datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30)),
            )
            db.add(scan)
            count += 1

        db.commit()
        print(f"[OK] Created {count} sample scan records.")
        print(f"\nLogin at http://localhost:8000/pages/login.html")
        print(f"  Email: {DEMO_EMAIL}")
        print(f"  Password: {DEMO_PASSWORD}")


if __name__ == "__main__":
    seed()
