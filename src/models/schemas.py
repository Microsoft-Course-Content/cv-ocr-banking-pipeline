"""Pydantic models for Computer Vision & OCR Pipeline API."""

from pydantic import BaseModel, Field
from typing import Optional


class QualityResponse(BaseModel):
    overall_score: float
    is_acceptable: bool
    blur_score: float
    is_blurry: bool
    skew_angle: float
    resolution_dpi_estimate: int
    noise_level: float
    brightness: float
    contrast: float
    has_sufficient_text: bool
    recommendations: list[str]


class ChequeResponse(BaseModel):
    payee: str = ""
    amount_figures: str = ""
    amount_words: str = ""
    date: str = ""
    cheque_number: str = ""
    micr_code: str = ""
    bank_name: str = ""
    account_number: str = ""
    signature_detected: bool = False
    fraud_flags: list[str] = []
    confidence: float = 0.0
    quality_score: float = 0.0


class IDCardResponse(BaseModel):
    document_type: str = ""
    full_name: str = ""
    first_name: str = ""
    last_name: str = ""
    date_of_birth: str = ""
    gender: str = ""
    nationality: str = ""
    document_number: str = ""
    expiry_date: str = ""
    issuing_country: str = ""
    face_detected: bool = False
    is_expired: bool = False
    confidence: float = 0.0
    quality_score: float = 0.0
    warnings: list[str] = []


class SignatureResponse(BaseModel):
    signature_detected: bool = False
    signature_region: list[int] = []
    similarity_score: float = 0.0
    match_verdict: str = ""
    keypoints_found: int = 0
    keypoints_matched: int = 0
    confidence: float = 0.0
    warnings: list[str] = []


class FraudResponse(BaseModel):
    is_suspicious: bool = False
    risk_level: str = "low"
    overall_score: float = 0.0
    checks: list[dict] = []
    flags: list[str] = []
    recommendations: list[str] = []


class OCRResponse(BaseModel):
    full_text: str
    line_count: int
    processing_time_ms: float
