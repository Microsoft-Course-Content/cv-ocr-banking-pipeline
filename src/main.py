"""
Computer Vision & OCR Pipeline for Banking — FastAPI Application.
Cheque reading, ID verification, signature matching, fraud detection.

Author: Jalal Ahmed Khan

Run locally:   uvicorn src.main:app --reload --port 8002
Run on Azure:  Deployed as Azure App Service (see README)
"""

import logging
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .models.schemas import (
    QualityResponse, ChequeResponse, IDCardResponse,
    SignatureResponse, FraudResponse, OCRResponse,
)
from .services.quality_assessor import QualityAssessor
from .services.ocr_engine import OCREngine
from .services.cheque_reader import ChequeReader
from .services.id_card_reader import IDCardReader
from .services.signature_verifier import SignatureVerifier
from .services.fraud_detector import FraudDetector
from .services.blob_storage import BlobStorageConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Computer Vision & OCR Pipeline — Banking",
    description=(
        "Production-grade CV and OCR system for banking documents. "
        "Cheque reading, ID card verification, signature matching, "
        "and fraud detection using Azure AI Vision, OpenCV, and GPT-4o."
    ),
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize services
quality_assessor = QualityAssessor()
ocr_engine = OCREngine()
cheque_reader = ChequeReader()
id_card_reader = IDCardReader()
signature_verifier = SignatureVerifier()
fraud_detector = FraudDetector()
blob_storage = BlobStorageConnector()

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/tiff", "image/bmp", "application/pdf"}

# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the Web UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"service": "CV & OCR Pipeline — Banking", "docs": "/docs"}


def _validate_upload(file: UploadFile):
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")


@app.post("/api/v1/quality/assess", response_model=QualityResponse)
async def assess_quality(file: UploadFile = File(...)):
    """Assess image quality for banking document processing."""
    _validate_upload(file)
    image_bytes = await file.read()
    report = quality_assessor.assess(image_bytes)
    return QualityResponse(**report.__dict__)


@app.post("/api/v1/cheque/read", response_model=ChequeResponse)
async def read_cheque(file: UploadFile = File(...)):
    """
    Process a cheque image — extract MICR code, amounts, payee,
    date, bank details, detect signature, and run fraud checks.
    """
    _validate_upload(file)
    image_bytes = await file.read()

    # Store in blob storage
    await blob_storage.store_image(image_bytes, file.filename or "cheque.png", file.content_type or "image/png")

    result = await cheque_reader.read(image_bytes)
    response = ChequeResponse(**result.__dict__)

    # Store result
    doc_id = f"cheque_{uuid.uuid4().hex[:8]}"
    await blob_storage.store_result(doc_id, response.model_dump())

    return response


@app.post("/api/v1/id-card/verify", response_model=IDCardResponse)
async def verify_id_card(file: UploadFile = File(...)):
    """
    Verify an ID document — parse MRZ, extract fields,
    detect face, check expiry, identify document type.
    """
    _validate_upload(file)
    image_bytes = await file.read()

    await blob_storage.store_image(image_bytes, file.filename or "id_card.png", file.content_type or "image/png")

    result = await id_card_reader.read(image_bytes)
    response = IDCardResponse(
        document_type=result.document_type,
        full_name=result.full_name,
        first_name=result.first_name,
        last_name=result.last_name,
        date_of_birth=result.date_of_birth,
        gender=result.gender,
        nationality=result.nationality,
        document_number=result.document_number,
        expiry_date=result.expiry_date,
        issuing_country=result.issuing_country,
        face_detected=result.face_detected,
        is_expired=result.is_expired,
        confidence=result.confidence,
        quality_score=result.quality_score,
        warnings=result.warnings,
    )

    doc_id = f"idcard_{uuid.uuid4().hex[:8]}"
    await blob_storage.store_result(doc_id, response.model_dump())

    return response


@app.post("/api/v1/signature/verify", response_model=SignatureResponse)
async def verify_signature(
    document: UploadFile = File(..., description="Document with signature"),
    reference: UploadFile = File(None, description="Reference signature (optional)"),
):
    """
    Detect signature on a document and optionally compare
    against a reference signature for verification.
    """
    _validate_upload(document)
    doc_bytes = await document.read()
    ref_bytes = await reference.read() if reference else None

    result = await signature_verifier.verify(doc_bytes, ref_bytes)
    return SignatureResponse(**result.__dict__)


@app.post("/api/v1/fraud/detect", response_model=FraudResponse)
async def detect_fraud(file: UploadFile = File(...)):
    """
    Run fraud detection analysis on a document image.
    Checks for tampering, copy-move forgery, noise
    inconsistencies, and compression artifacts.
    """
    _validate_upload(file)
    image_bytes = await file.read()
    report = await fraud_detector.analyze(image_bytes)
    return FraudResponse(**report.__dict__)


@app.post("/api/v1/ocr/extract", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(...)):
    """General-purpose OCR text extraction using Azure AI Vision."""
    _validate_upload(file)
    image_bytes = await file.read()
    result = await ocr_engine.extract_text(image_bytes)
    return OCRResponse(
        full_text=result.full_text,
        line_count=len(result.lines),
        processing_time_ms=result.processing_time_ms,
    )


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "service": "CV & OCR Pipeline — Banking"}


@app.get("/")
async def root():
    return {
        "service": "Computer Vision & OCR Pipeline — Banking",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "cheque_read": "POST /api/v1/cheque/read",
            "id_verify": "POST /api/v1/id-card/verify",
            "signature_verify": "POST /api/v1/signature/verify",
            "fraud_detect": "POST /api/v1/fraud/detect",
            "quality_assess": "POST /api/v1/quality/assess",
            "ocr_extract": "POST /api/v1/ocr/extract",
        },
    }
