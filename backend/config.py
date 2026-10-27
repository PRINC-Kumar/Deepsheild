"""
config.py — central settings loaded from .env
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    APP_ENV: str = "development"
    SECRET_KEY: str = "CHANGE_ME"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Database
    DATABASE_URL: str = "sqlite:///./deepshield.db"

    # File limits
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 100
    MAX_VIDEO_DURATION_S: int = 60
    AUTO_DELETE_UPLOADS_DAYS: int = 30

    # ML
    USE_MOCK_ML: bool = True
    ML_MODULE_PATH: str = ""
    ML_WEIGHTS_PATH: str = ""
    ML_GRADCAM_LAYER: str = "layer4"
    ML_LABEL_FAKE_INDEX: int = 1
    ML_THRESHOLD: float = 0.5
    UNCERTAIN_LOW: float = 0.4
    UNCERTAIN_HIGH: float = 0.6

    # Rate limiting
    RATE_LIMIT_DETECT_PER_MINUTE: int = 20

    # Directories
    UPLOAD_DIR: str = "uploads"
    HEATMAP_DIR: str = "heatmaps"

    @property
    def upload_path(self) -> Path:
        return Path(self.UPLOAD_DIR)

    @property
    def heatmap_path(self) -> Path:
        return Path(self.HEATMAP_DIR)

    @property
    def max_image_bytes(self) -> int:
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024

    @property
    def max_video_bytes(self) -> int:
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024


settings = Settings()

