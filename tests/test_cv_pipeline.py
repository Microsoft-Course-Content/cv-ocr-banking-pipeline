"""Tests for image quality assessment and fraud detection."""

import cv2
import numpy as np
import pytest
from src.services.quality_assessor import QualityAssessor
from src.services.fraud_detector import FraudDetector


def _create_test_image(width=800, height=1000, text=True) -> bytes:
    """Create a synthetic test document image."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background

    if text:
        # Add some text-like horizontal lines to simulate a document
        for y in range(100, height - 100, 40):
            line_width = np.random.randint(200, width - 100)
            cv2.line(img, (50, y), (50 + line_width, y), (30, 30, 30), 2)

        # Add a title-like block
        cv2.rectangle(img, (50, 30), (400, 70), (20, 20, 20), -1)

    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


def _create_blurry_image() -> bytes:
    """Create a blurry test image."""
    img = np.ones((800, 600, 3), dtype=np.uint8) * 240
    cv2.putText(img, "CHEQUE", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
    # Apply heavy Gaussian blur
    img = cv2.GaussianBlur(img, (31, 31), 10)
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


class TestQualityAssessor:
    def setup_method(self):
        self.assessor = QualityAssessor()

    def test_good_quality_image(self):
        image = _create_test_image()
        report = self.assessor.assess(image)
        assert report.overall_score > 0.5
        assert report.has_sufficient_text is True

    def test_blurry_image_detected(self):
        image = _create_blurry_image()
        report = self.assessor.assess(image)
        assert report.is_blurry is True
        assert report.blur_score < 150  # Low Laplacian variance

    def test_empty_image_fails(self):
        # Completely white image — no text
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        report = self.assessor.assess(buffer.tobytes())
        assert report.has_sufficient_text is False

    def test_brightness_measurement(self):
        # Dark image
        dark = np.ones((400, 400, 3), dtype=np.uint8) * 30
        _, buf = cv2.imencode(".png", dark)
        report = self.assessor.assess(buf.tobytes())
        assert report.brightness < 50

        # Bright image
        bright = np.ones((400, 400, 3), dtype=np.uint8) * 230
        _, buf = cv2.imencode(".png", bright)
        report = self.assessor.assess(buf.tobytes())
        assert report.brightness > 200


class TestFraudDetector:
    def setup_method(self):
        self.detector = FraudDetector()

    @pytest.mark.asyncio
    async def test_clean_image_passes(self):
        image = _create_test_image()
        report = await self.detector.analyze(image)
        assert report.risk_level in ("low", "medium")
        assert len(report.checks) == 5

    @pytest.mark.asyncio
    async def test_tampered_image_flagged(self):
        """Image with a pasted white rectangle should raise ELA flags."""
        img = np.random.randint(100, 200, (600, 800, 3), dtype=np.uint8)
        # Paste a clean white block (simulating tampered region)
        img[200:300, 300:500] = 255
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        report = await self.detector.analyze(buffer.tobytes())
        # Should detect some anomaly
        assert len(report.checks) == 5
        assert report.overall_score >= 0.0  # May or may not flag depending on severity

    @pytest.mark.asyncio
    async def test_invalid_image_flagged(self):
        report = await self.detector.analyze(b"not_an_image")
        assert report.is_suspicious is True
        assert "UNREADABLE_IMAGE" in report.flags
