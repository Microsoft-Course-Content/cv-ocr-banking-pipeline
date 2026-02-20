"""
OCR Engine using Azure AI Vision Read API.
Extracts text from banking document images with line-level
bounding boxes and confidence scores.
"""

import logging
import httpx
import time
from dataclasses import dataclass, field
from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class OCRLine:
    """A single line of extracted text."""
    text: str
    confidence: float
    bounding_box: list[float]
    page_number: int = 1


@dataclass
class OCRResult:
    """Complete OCR extraction result."""
    full_text: str
    lines: list[OCRLine]
    page_count: int
    language: str = "en"
    processing_time_ms: float = 0.0


class OCREngine:
    """
    Azure AI Vision Read API wrapper for banking document OCR.
    Supports PDF, PNG, JPEG, TIFF, and BMP formats.
    """

    def __init__(self):
        settings = get_settings()
        self.endpoint = settings.azure_vision_endpoint.rstrip("/")
        self.api_key = settings.azure_vision_api_key

    async def extract_text(self, image_bytes: bytes) -> OCRResult:
        """
        Extract text from a document image using Azure AI Vision Read API.

        Args:
            image_bytes: Raw image bytes

        Returns:
            OCRResult with full text, lines, and bounding boxes
        """
        start_time = time.time()

        # Step 1: Submit read request
        url = f"{self.endpoint}/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read"

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=headers, content=image_bytes, timeout=30.0
            )
            response.raise_for_status()
            result = response.json()

        # Parse results
        lines = []
        full_text_parts = []

        read_result = result.get("readResult", {})
        blocks = read_result.get("blocks", [])

        for block in blocks:
            for line_data in block.get("lines", []):
                text = line_data.get("text", "")
                bbox = line_data.get("boundingPolygon", [])
                
                # Calculate line confidence from word confidences
                words = line_data.get("words", [])
                if words:
                    avg_confidence = sum(w.get("confidence", 0) for w in words) / len(words)
                else:
                    avg_confidence = 0.0

                bbox_flat = []
                for point in bbox:
                    bbox_flat.extend([point.get("x", 0), point.get("y", 0)])

                lines.append(OCRLine(
                    text=text,
                    confidence=avg_confidence,
                    bounding_box=bbox_flat,
                ))
                full_text_parts.append(text)

        processing_time = (time.time() - start_time) * 1000

        return OCRResult(
            full_text="\n".join(full_text_parts),
            lines=lines,
            page_count=1,
            processing_time_ms=round(processing_time, 2),
        )

    async def extract_text_with_layout(self, image_bytes: bytes) -> dict:
        """
        Extract text with full layout information including
        paragraphs, tables, and spatial positioning.
        Uses Document Intelligence Read model for richer output.
        """
        url = f"{self.endpoint}/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read,caption"

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=headers, content=image_bytes, timeout=30.0
            )
            response.raise_for_status()
            return response.json()
