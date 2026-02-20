"""Configuration for Computer Vision & OCR Pipeline."""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    azure_vision_endpoint: str = Field(...)
    azure_vision_api_key: str = Field(...)
    azure_openai_endpoint: str = Field(...)
    azure_openai_api_key: str = Field(...)
    azure_openai_deployment_name: str = Field(default="gpt-4o")
    azure_openai_api_version: str = Field(default="2024-12-01-preview")
    min_quality_score: float = Field(default=0.6)
    blur_threshold: float = Field(default=100.0)
    min_resolution_dpi: int = Field(default=200)

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
