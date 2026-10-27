"""
routes/stats.py — Dashboard stats and health
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.scan import ScanHistory
from backend.models.user import User
from backend.routes.deps import get_current_user
from backend.schemas.scan import StatsOut
from backend.services import ml_adapter
from backend.services.auth_service import hash_password, verify_password

router = APIRouter(tags=["stats"])


@router.get("/api/stats", response_model=StatsOut)
def get_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = db.execute(
        select(
            func.count().label("total"),
            func.sum(case((ScanHistory.label == "REAL", 1), else_=0)).label("real"),
            func.sum(case((ScanHistory.label == "FAKE", 1), else_=0)).label("fake"),
            func.sum(case((ScanHistory.label == "UNCERTAIN", 1), else_=0)).label("uncertain"),
            func.sum(case((ScanHistory.label == "NO_FACE", 1), else_=0)).label("no_face"),
            func.avg(ScanHistory.confidence).label("avg_conf"),
        ).where(ScanHistory.user_id == user.id)
    )
    row = result.one()

    # Scans over time (last 30 days, grouped by date)
    time_result = db.execute(
        text(
            "SELECT date(created_at) as d, count(*) as cnt "
            "FROM scan_history "
            "WHERE user_id = :uid "
            "AND created_at >= date('now', '-30 days') "
            "GROUP BY d ORDER BY d"
        ),
        {"uid": user.id},
    )
    scans_over_time = [
        {"date": r.d, "count": r.cnt} for r in time_result
    ]

    return StatsOut(
        total_scans=row.total or 0,
        real_count=row.real or 0,
        fake_count=row.fake or 0,
        uncertain_count=row.uncertain or 0,
        no_face_count=row.no_face or 0,
        avg_confidence=round(float(row.avg_conf or 0), 4),
        scans_over_time=scans_over_time,
    )


@router.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ml_adapter.is_model_loaded(),
        "mock_mode": ml_adapter.is_mock_mode(),
    }


@router.get("/api/user/me")
def me(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return {
        "id": user.id,
        "email": user.email,
        "confidence_threshold": user.confidence_threshold,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.put("/api/user/me")
def update_profile(
    body: dict,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if "confidence_threshold" in body:
        t = float(body["confidence_threshold"])
        if not 0.0 <= t <= 1.0:
            raise HTTPException(400, "Threshold must be between 0 and 1.")
        user.confidence_threshold = t

    if "new_password" in body:
        old_pw = body.get("current_password", "")
        if not verify_password(old_pw, user.hashed_password):
            raise HTTPException(400, "Current password is incorrect.")
        if len(body["new_password"]) < 8:
            raise HTTPException(400, "New password must be at least 8 characters.")
        user.hashed_password = hash_password(body["new_password"])

    db.add(user)
    return {"message": "Profile updated."}


@router.delete("/api/user/me", status_code=204)
def delete_account(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete user and all their data."""
    db.delete(user)
