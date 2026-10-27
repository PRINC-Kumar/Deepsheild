"""
schemas/scan.py — Pydantic schemas for scan/history
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ScanResult(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    faces_detected: int
    heatmap_path: str | None
    frame_scores: list[dict]
    processing_time_ms: int
    warnings: list[str]


class ScanHistoryOut(BaseModel):
    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size_bytes: int | None
    thumbnail_path: str | None
    label: str
    confidence: float
    fake_probability: float
    faces_detected: int
    heatmap_path: str | None
    frame_scores: list[Any]
    processing_time_ms: int
    warnings: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryPage(BaseModel):
    items: list[ScanHistoryOut]
    total: int
    page: int
    pages: int
    per_page: int


class StatsOut(BaseModel):
    total_scans: int
    real_count: int
    fake_count: int
    uncertain_count: int
    no_face_count: int
    avg_confidence: float
    scans_over_time: list[dict]   # [{date, count}]

