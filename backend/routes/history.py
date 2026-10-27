"""
routes/history.py — Scan history CRUD + CSV export
"""
from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.scan import ScanHistory
from backend.models.user import User
from backend.routes.deps import get_current_user
from backend.schemas.scan import HistoryPage, ScanHistoryOut

router = APIRouter(prefix="/api/history", tags=["history"])


def _to_out(scan: ScanHistory) -> ScanHistoryOut:
    return ScanHistoryOut(
        id=scan.id,
        filename=scan.filename,
        original_filename=scan.original_filename,
        file_type=scan.file_type,
        file_size_bytes=scan.file_size_bytes,
        thumbnail_path=scan.thumbnail_path,
        label=scan.label,
        confidence=scan.confidence,
        fake_probability=scan.fake_probability,
        faces_detected=scan.faces_detected,
        heatmap_path=scan.heatmap_path,
        frame_scores=json.loads(scan.frame_scores or "[]"),
        processing_time_ms=scan.processing_time_ms,
        warnings=json.loads(scan.warnings or "[]"),
        created_at=scan.created_at,
    )


@router.get("", response_model=HistoryPage)
def list_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    label: str | None = Query(None),
    file_type: str | None = Query(None),
    search: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = select(ScanHistory).where(ScanHistory.user_id == user.id)

    if label:
        q = q.where(ScanHistory.label == label.upper())
    if file_type:
        q = q.where(ScanHistory.file_type == file_type.lower())
    if search:
        q = q.where(ScanHistory.original_filename.ilike(f"%{search}%"))
    if date_from:
        try:
            q = q.where(ScanHistory.created_at >= datetime.fromisoformat(date_from))
        except ValueError:
            pass
    if date_to:
        try:
            q = q.where(ScanHistory.created_at <= datetime.fromisoformat(date_to))
        except ValueError:
            pass

    count_q = select(func.count()).select_from(q.subquery())
    total_result = db.execute(count_q)
    total = total_result.scalar_one()

    q = q.order_by(ScanHistory.created_at.desc())
    q = q.offset((page - 1) * per_page).limit(per_page)
    result = db.execute(q)
    scans = result.scalars().all()

    return HistoryPage(
        items=[_to_out(s) for s in scans],
        total=total,
        page=page,
        pages=math.ceil(total / per_page) if total else 1,
        per_page=per_page,
    )


@router.get("/export")
def export_csv(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = db.execute(
        select(ScanHistory)
        .where(ScanHistory.user_id == user.id)
        .order_by(ScanHistory.created_at.desc())
    )
    scans = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id", "filename", "file_type", "label", "confidence",
            "fake_probability", "faces_detected", "processing_time_ms",
            "warnings", "created_at",
        ]
    )
    for s in scans:
        writer.writerow(
            [
                s.id,
                s.original_filename,
                s.file_type,
                s.label,
                f"{s.confidence:.4f}",
                f"{s.fake_probability:.4f}",
                s.faces_detected,
                s.processing_time_ms,
                s.warnings,
                s.created_at.isoformat() if s.created_at else "",
            ]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=deepshield_history.csv"},
    )


@router.get("/{scan_id}", response_model=ScanHistoryOut)
def get_scan(
    scan_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = db.execute(
        select(ScanHistory).where(
            ScanHistory.id == scan_id, ScanHistory.user_id == user.id
        )
    )
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(404, "Scan not found.")
    return _to_out(scan)


@router.delete("/{scan_id}", status_code=204)
def delete_scan(
    scan_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = db.execute(
        select(ScanHistory).where(
            ScanHistory.id == scan_id, ScanHistory.user_id == user.id
        )
    )
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(404, "Scan not found.")
    db.delete(scan)


@router.delete("", status_code=204)
def clear_history(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    db.execute(
        delete(ScanHistory).where(ScanHistory.user_id == user.id)
    )
