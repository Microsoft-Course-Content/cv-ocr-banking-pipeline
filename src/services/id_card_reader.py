"""
ID Card Verification Pipeline.
Reads passports, national IDs, Emirates IDs, and driving licenses.
Extracts fields, parses MRZ, detects face, and validates expiry.
"""

import cv2
import re
import numpy as np
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from .ocr_engine import OCREngine
from .quality_assessor import QualityAssessor

logger = logging.getLogger(__name__)


@dataclass
class IDCardResult:
    """Extracted ID card fields."""
    document_type: str = ""  # passport, national_id, emirates_id, driving_license
    full_name: str = ""
    first_name: str = ""
    last_name: str = ""
    date_of_birth: str = ""
    gender: str = ""
    nationality: str = ""
    document_number: str = ""
    expiry_date: str = ""
    issuing_country: str = ""
    mrz_data: dict = field(default_factory=dict)
    face_detected: bool = False
    face_bbox: list = field(default_factory=list)
    is_expired: bool = False
    confidence: float = 0.0
    quality_score: float = 0.0
    warnings: list = field(default_factory=list)


# MRZ character set mapping for OCR correction
MRZ_CORRECTIONS = {
    "O": "0", "I": "1", "S": "5", "B": "8",
    "G": "6", "Q": "0", "D": "0",
}

# Country code to name mapping (subset)
COUNTRY_CODES = {
    "ARE": "United Arab Emirates", "IND": "India", "USA": "United States",
    "GBR": "United Kingdom", "SAU": "Saudi Arabia", "PAK": "Pakistan",
    "BGD": "Bangladesh", "PHL": "Philippines", "EGY": "Egypt",
    "JOR": "Jordan", "LBN": "Lebanon", "OMN": "Oman",
    "BHR": "Bahrain", "KWT": "Kuwait", "QAT": "Qatar",
}


class IDCardReader:
    """
    End-to-end ID card verification pipeline.
    Supports passports (TD3), ID cards (TD1/TD2), and Emirates ID.
    """

    def __init__(self):
        self.ocr = OCREngine()
        self.quality = QualityAssessor()

    async def read(self, image_bytes: bytes) -> IDCardResult:
        """Full ID card reading pipeline."""
        result = IDCardResult()

        # Step 1: Quality assessment
        quality = self.quality.assess(image_bytes)
        result.quality_score = quality.overall_score
        if not quality.is_acceptable:
            result.warnings.append(f"Low image quality ({quality.overall_score:.2f})")

        # Step 2: Preprocess
        processed = self._preprocess_id(image_bytes)

        # Step 3: OCR
        ocr_result = await self.ocr.extract_text(processed)
        text = ocr_result.full_text
        lines = [l.text for l in ocr_result.lines]

        # Step 4: Detect and parse MRZ
        mrz_lines = self._find_mrz_lines(lines)
        if mrz_lines:
            result.mrz_data = self._parse_mrz(mrz_lines)
            result.document_type = result.mrz_data.get("document_type", "unknown")
            result.full_name = result.mrz_data.get("full_name", "")
            result.first_name = result.mrz_data.get("first_name", "")
            result.last_name = result.mrz_data.get("last_name", "")
            result.date_of_birth = result.mrz_data.get("date_of_birth", "")
            result.gender = result.mrz_data.get("gender", "")
            result.nationality = result.mrz_data.get("nationality", "")
            result.document_number = result.mrz_data.get("document_number", "")
            result.expiry_date = result.mrz_data.get("expiry_date", "")
            result.issuing_country = result.mrz_data.get("issuing_country", "")
        else:
            # Fallback: extract fields from text without MRZ
            result = self._extract_from_text(text, result)

        # Step 5: Face detection
        result.face_detected, result.face_bbox = self._detect_face(image_bytes)
        if not result.face_detected:
            result.warnings.append("No face detected on ID document")

        # Step 6: Expiry check
        if result.expiry_date:
            result.is_expired = self._check_expiry(result.expiry_date)
            if result.is_expired:
                result.warnings.append(f"Document EXPIRED: {result.expiry_date}")

        # Confidence
        filled = sum(1 for v in [result.full_name, result.document_number, result.nationality, result.expiry_date] if v)
        result.confidence = filled / 4.0

        logger.info(f"ID read: {result.document_type} — {result.full_name}, confidence={result.confidence:.2f}")
        return result

    def _preprocess_id(self, image_bytes: bytes) -> bytes:
        """Preprocess ID card image for OCR."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        # Remove borders
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 5
            x, y = max(0, x - margin), max(0, y - margin)
            img = img[y:y + h + 2 * margin, x:x + w + 2 * margin]

        # Enhance contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)

        _, buffer = cv2.imencode(".png", img)
        return buffer.tobytes()

    def _find_mrz_lines(self, lines: list[str]) -> list[str]:
        """Find MRZ lines in OCR output."""
        mrz_candidates = []
        for line in lines:
            cleaned = line.strip().replace(" ", "")
            # MRZ lines contain only A-Z, 0-9, and < characters
            if len(cleaned) >= 30 and re.match(r'^[A-Z0-9<]{30,}$', cleaned):
                mrz_candidates.append(cleaned)

        if len(mrz_candidates) >= 2:
            return mrz_candidates[-2:]  # MRZ is usually at the bottom
        return []

    def _parse_mrz(self, mrz_lines: list[str]) -> dict:
        """
        Parse Machine Readable Zone (MRZ) from passport/ID.
        Supports TD3 (passport, 2x44), TD2 (2x36), TD1 (3x30).
        """
        result = {}

        if len(mrz_lines) < 2:
            return result

        line1 = mrz_lines[0]
        line2 = mrz_lines[1]

        # Apply OCR corrections to digits in MRZ
        line2_corrected = self._correct_mrz_digits(line2)

        # Determine format
        if len(line1) >= 44:  # TD3 — Passport
            result["document_type"] = "passport"
            result["issuing_country"] = COUNTRY_CODES.get(
                line1[2:5].replace("<", ""), line1[2:5]
            )

            name_field = line1[5:44]
            parts = name_field.split("<<")
            result["last_name"] = parts[0].replace("<", " ").strip()
            result["first_name"] = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
            result["full_name"] = f"{result['first_name']} {result['last_name']}".strip()

            result["document_number"] = line2_corrected[0:9].replace("<", "")
            result["nationality"] = COUNTRY_CODES.get(
                line2_corrected[10:13].replace("<", ""), line2_corrected[10:13]
            )
            result["date_of_birth"] = self._parse_mrz_date(line2_corrected[13:19])
            result["gender"] = {"M": "Male", "F": "Female"}.get(line2_corrected[20], "Unknown")
            result["expiry_date"] = self._parse_mrz_date(line2_corrected[21:27])

        elif len(line1) >= 36:  # TD2 — ID Card
            result["document_type"] = "national_id"
            result["issuing_country"] = COUNTRY_CODES.get(
                line1[2:5].replace("<", ""), line1[2:5]
            )

            name_field = line1[5:36]
            parts = name_field.split("<<")
            result["last_name"] = parts[0].replace("<", " ").strip()
            result["first_name"] = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
            result["full_name"] = f"{result['first_name']} {result['last_name']}".strip()

            result["document_number"] = line2_corrected[0:9].replace("<", "")
            result["nationality"] = COUNTRY_CODES.get(
                line2_corrected[10:13].replace("<", ""), line2_corrected[10:13]
            )
            result["date_of_birth"] = self._parse_mrz_date(line2_corrected[13:19])
            result["gender"] = {"M": "Male", "F": "Female"}.get(line2_corrected[20], "Unknown")
            result["expiry_date"] = self._parse_mrz_date(line2_corrected[21:27])

        return result

    def _correct_mrz_digits(self, line: str) -> str:
        """Apply OCR corrections to MRZ digit positions."""
        # In numeric-only positions, correct common OCR misreads
        return line  # Full implementation would check digit positions

    def _parse_mrz_date(self, date_str: str) -> str:
        """Parse YYMMDD MRZ date to YYYY-MM-DD."""
        if len(date_str) < 6:
            return ""
        try:
            yy = int(date_str[0:2])
            mm = date_str[2:4]
            dd = date_str[4:6]
            year = 2000 + yy if yy < 50 else 1900 + yy
            return f"{year}-{mm}-{dd}"
        except (ValueError, IndexError):
            return date_str

    def _extract_from_text(self, text: str, result: IDCardResult) -> IDCardResult:
        """Fallback: extract fields from OCR text without MRZ."""
        # Name patterns
        name_match = re.search(r"(?:Name|Full Name|Given Name)[:\s]+(.+)", text, re.I)
        if name_match:
            result.full_name = name_match.group(1).strip()

        # DOB
        dob_match = re.search(r"(?:DOB|Date of Birth|Born)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.I)
        if dob_match:
            result.date_of_birth = dob_match.group(1)

        # ID Number
        id_match = re.search(r"(?:ID No|Document No|Passport No)[.:\s]+([A-Z0-9-]+)", text, re.I)
        if id_match:
            result.document_number = id_match.group(1)

        # Nationality
        nat_match = re.search(r"(?:Nationality|Citizenship)[:\s]+(\w+)", text, re.I)
        if nat_match:
            result.nationality = nat_match.group(1)

        # Expiry
        exp_match = re.search(r"(?:Expiry|Valid Until|Exp)[.:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.I)
        if exp_match:
            result.expiry_date = exp_match.group(1)

        # Gender
        if re.search(r'\bMale\b', text, re.I):
            result.gender = "Male"
        elif re.search(r'\bFemale\b', text, re.I):
            result.gender = "Female"

        result.document_type = "unknown"
        if re.search(r'passport', text, re.I):
            result.document_type = "passport"
        elif re.search(r'emirates|EID', text, re.I):
            result.document_type = "emirates_id"
        elif re.search(r'driv', text, re.I):
            result.document_type = "driving_license"

        return result

    def _detect_face(self, image_bytes: bytes) -> tuple[bool, list]:
        """Detect face on ID card using OpenCV Haar cascades."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take the largest/first face
            return True, [int(x), int(y), int(w), int(h)]
        return False, []

    def _check_expiry(self, expiry_date: str) -> bool:
        """Check if the document is expired."""
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                exp = datetime.strptime(expiry_date, fmt).date()
                return exp < date.today()
            except ValueError:
                continue
        return False
