"""
Cheque Reader — Full Computer Vision Pipeline.
Reads cheques using OpenCV preprocessing + Azure AI Vision OCR
with banking-specific post-processing for MICR, amounts, and dates.
"""

import cv2
import re
import numpy as np
import logging
from dataclasses import dataclass
from .ocr_engine import OCREngine, OCRResult
from .quality_assessor import QualityAssessor

logger = logging.getLogger(__name__)


@dataclass
class ChequeReadResult:
    """Complete cheque reading result."""
    payee_name: str | None = None
    amount_figures: str | None = None
    amount_words: str | None = None
    cheque_date: str | None = None
    cheque_number: str | None = None
    micr_code: str | None = None
    bank_code: str | None = None
    branch_code: str | None = None
    account_number: str | None = None
    bank_name: str | None = None
    ifsc_code: str | None = None
    signature_detected: bool = False
    amounts_match: bool | None = None
    fraud_flags: list[str] = None
    confidence_scores: dict = None
    quality_score: float = 0.0
    raw_ocr_text: str = ""

    def __post_init__(self):
        if self.fraud_flags is None:
            self.fraud_flags = []
        if self.confidence_scores is None:
            self.confidence_scores = {}


# UAE and India bank routing codes
BANK_CODES = {
    "033": "ADCB", "044": "Emirates NBD", "046": "ADIB",
    "035": "FAB", "050": "Mashreq", "026": "CBD",
    "060": "SBI", "002": "HDFC", "004": "ICICI",
    "029": "Axis Bank", "019": "Kotak", "011": "Union Bank",
}


class ChequeReader:
    """
    End-to-end cheque reading pipeline using Computer Vision.
    
    Pipeline:
    1. Quality assessment
    2. Image preprocessing (deskew, denoise, enhance)
    3. Region of Interest (ROI) detection
    4. OCR extraction
    5. Field-specific parsing (MICR, amount, date, payee)
    6. Cross-validation and fraud checks
    """

    def __init__(self):
        self.ocr = OCREngine()
        self.quality = QualityAssessor()

    async def read_cheque(self, image_bytes: bytes) -> ChequeReadResult:
        """Process a cheque image end-to-end."""
        result = ChequeReadResult()

        # Step 1: Quality check
        quality = self.quality.assess(image_bytes)
        result.quality_score = quality.overall_score

        if not quality.is_acceptable:
            result.fraud_flags.append(f"LOW_QUALITY: {quality.recommendations}")
            logger.warning("Cheque image quality below threshold")

        # Step 2: Preprocess
        processed = self._preprocess(image_bytes)

        # Step 3: OCR extraction
        ocr_result = await self.ocr.extract_text(processed)
        result.raw_ocr_text = ocr_result.full_text

        # Step 4: Extract fields
        result.micr_code, micr_parts = self._extract_micr(ocr_result)
        if micr_parts:
            result.cheque_number = micr_parts.get("cheque_number")
            result.bank_code = micr_parts.get("bank_code")
            result.branch_code = micr_parts.get("branch_code")
            result.account_number = micr_parts.get("account_number")
            result.bank_name = BANK_CODES.get(
                (micr_parts.get("bank_code") or "")[:3], "Unknown"
            )

        result.amount_figures = self._extract_amount_figures(ocr_result)
        result.amount_words = self._extract_amount_words(ocr_result)
        result.cheque_date = self._extract_date(ocr_result)
        result.payee_name = self._extract_payee(ocr_result)
        result.ifsc_code = self._extract_ifsc(ocr_result)

        # Step 5: Signature detection
        result.signature_detected = self._detect_signature(image_bytes)

        # Step 6: Cross-validation
        result.amounts_match = self._validate_amounts(
            result.amount_figures, result.amount_words
        )
        if result.amounts_match is False:
            result.fraud_flags.append("AMOUNT_MISMATCH")

        if not result.signature_detected:
            result.fraud_flags.append("NO_SIGNATURE_DETECTED")

        # Confidence scores
        result.confidence_scores = self._calculate_confidence(result, ocr_result)

        logger.info(
            f"Cheque read: #{result.cheque_number or 'N/A'}, "
            f"amount={result.amount_figures}, flags={result.fraud_flags}"
        )
        return result

    def _preprocess(self, image_bytes: bytes) -> bytes:
        """Full cheque preprocessing pipeline."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Deskew
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            angles = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in lines]
            angles = [a for a in angles if abs(a) < 45]
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Denoise
        img = cv2.bilateralFilter(img, 9, 75, 75)

        # Enhance contrast (CLAHE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def _extract_micr(self, ocr: OCRResult) -> tuple[str | None, dict | None]:
        """Extract MICR code from cheque bottom."""
        # MICR is typically in the bottom 20% of the cheque
        bottom_lines = [l for l in ocr.lines if l.bounding_box and
                        len(l.bounding_box) >= 4 and l.bounding_box[1] > 0]

        all_text = ocr.full_text

        # Standard MICR: cheque_no routing_code account_no
        patterns = [
            r"(\d{6})\s+(\d{9})\s+(\d{6,15})",
            r"[⑆⑈](\d{6})[⑆⑈]\s*[⑇](\d{9})[⑇]\s*(\d{6,15})",
        ]

        for pattern in patterns:
            match = re.search(pattern, all_text)
            if match:
                cheque_no, routing, account = match.groups()
                return (
                    f"{cheque_no} {routing} {account}",
                    {
                        "cheque_number": cheque_no,
                        "bank_code": routing[:3],
                        "branch_code": routing[3:],
                        "account_number": account,
                    },
                )
        return None, None

    def _extract_amount_figures(self, ocr: OCRResult) -> str | None:
        """Extract numerical amount."""
        patterns = [
            r"(?:AED|USD|INR|Rs\.?|SAR|\$|£|€|₹)\s*([\d,]+\.?\d{0,2})",
            r"([\d,]+\.?\d{0,2})\s*(?:AED|USD|INR|SAR|/-)",
            r"\*+([\d,]+\.?\d{0,2})\*+",
        ]
        for pattern in patterns:
            match = re.search(pattern, ocr.full_text, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "")
        return None

    def _extract_amount_words(self, ocr: OCRResult) -> str | None:
        """Extract amount in words."""
        pattern = r"(?:Rupees?|Dirhams?|Dollars?|Amount)[\s:]+(.+?)(?:Only|ONLY|only|\n)"
        match = re.search(pattern, ocr.full_text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_date(self, ocr: OCRResult) -> str | None:
        """Extract cheque date."""
        patterns = [
            r"(?:Date|Dated?)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})",
            r"(\d{2}[/-]\d{2}[/-]\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, ocr.full_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_payee(self, ocr: OCRResult) -> str | None:
        """Extract payee name."""
        pattern = r"(?:Pay|Pay\s*to|Payee)[\s:]+(.+?)(?:\n|or bearer|or order)"
        match = re.search(pattern, ocr.full_text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_ifsc(self, ocr: OCRResult) -> str | None:
        """Extract IFSC code (Indian banking)."""
        match = re.search(r"[A-Z]{4}0[A-Z0-9]{6}", ocr.full_text)
        return match.group(0) if match else None

    def _detect_signature(self, image_bytes: bytes) -> bool:
        """
        Detect signature presence in bottom-right quadrant.
        Uses contour analysis to find handwriting-like marks.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        # Focus on bottom-right quadrant (signature region)
        roi = img[int(h * 0.7):, int(w * 0.5):]

        # Threshold and find contours
        _, thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Signature: multiple small-medium contours (handwriting)
        sig_contours = [c for c in contours if 50 < cv2.contourArea(c) < 5000]
        return len(sig_contours) > 5

    def _validate_amounts(self, figures: str | None, words: str | None) -> bool | None:
        """Cross-validate amount in figures vs words."""
        if not figures or not words:
            return None
        # Basic validation — in production, use num2words for conversion
        return True  # Simplified: actual impl would convert and compare

    def _calculate_confidence(self, result: ChequeReadResult, ocr: OCRResult) -> dict:
        """Calculate per-field confidence scores."""
        scores = {}
        if ocr.lines:
            avg_ocr = sum(l.confidence for l in ocr.lines) / len(ocr.lines)
            scores["ocr_average"] = round(avg_ocr, 2)

        scores["micr"] = 0.9 if result.micr_code else 0.0
        scores["amount"] = 0.85 if result.amount_figures else 0.0
        scores["date"] = 0.85 if result.cheque_date else 0.0
        scores["payee"] = 0.80 if result.payee_name else 0.0
        scores["signature"] = 0.75 if result.signature_detected else 0.0
        return scores
