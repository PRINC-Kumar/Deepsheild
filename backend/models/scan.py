"""
models/scan.py — SQLAlchemy ORM model for ScanHistory
"""
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class ScanHistory(Base):
    __tablename__ = "scan_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # File metadata
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)  # image | video | frame
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)
    thumbnail_path: Mapped[str] = mapped_column(String(512), nullable=True)

    # ML result
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    fake_probability: Mapped[float] = mapped_column(Float, nullable=False)
    faces_detected: Mapped[int] = mapped_column(Integer, default=0)
    heatmap_path: Mapped[str] = mapped_column(String(512), nullable=True)
    frame_scores: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string
    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    warnings: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    user = relationship("User", back_populates="scans")

