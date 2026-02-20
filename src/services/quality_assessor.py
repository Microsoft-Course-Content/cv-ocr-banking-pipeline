"""
Image Quality Assessment for Banking Documents.
Evaluates blur, skew, resolution, noise, and overall quality
to determine if a document image is suitable for OCR processing.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Image quality assessment results."""
    overall_score: float  # 0.0 - 1.0
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


class QualityAssessor:
    """
    Assesses document image quality using multiple OpenCV-based metrics.
    Banking documents require high quality for accurate OCR.
    """

    def __init__(self, blur_threshold: float = 100.0, min_dpi: int = 200):
        self.blur_threshold = blur_threshold
        self.min_dpi = min_dpi

    def assess(self, image_bytes: bytes) -> QualityReport:
        """
        Run full quality assessment on a document image.

        Args:
            image_bytes: Raw image bytes

        Returns:
            QualityReport with all metrics and recommendations
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return QualityReport(
                overall_score=0.0, is_acceptable=False,
                blur_score=0.0, is_blurry=True,
                skew_angle=0.0, resolution_dpi_estimate=0,
                noise_level=1.0, brightness=0.0, contrast=0.0,
                has_sufficient_text=False,
                recommendations=["Image could not be decoded"],
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Individual assessments
        blur_score, is_blurry = self._check_blur(gray)
        skew_angle = self._measure_skew(gray)
        dpi_estimate = self._estimate_dpi(img)
        noise_level = self._measure_noise(gray)
        brightness = self._measure_brightness(gray)
        contrast = self._measure_contrast(gray)
        has_text = self._check_text_presence(gray)

        # Calculate overall score
        recommendations = []
        score = 1.0

        if is_blurry:
            score -= 0.3
            recommendations.append(f"Image is blurry (score: {blur_score:.0f}). Re-scan at higher quality.")

        if abs(skew_angle) > 3.0:
            score -= 0.15
            recommendations.append(f"Document is skewed by {skew_angle:.1f}°. Will auto-correct.")

        if dpi_estimate < self.min_dpi:
            score -= 0.2
            recommendations.append(f"Low resolution (~{dpi_estimate} DPI). Scan at 300+ DPI.")

        if noise_level > 0.3:
            score -= 0.1
            recommendations.append("High noise detected. Use cleaner scan.")

        if brightness < 80 or brightness > 220:
            score -= 0.1
            recommendations.append(f"Poor brightness ({brightness:.0f}). Adjust lighting.")

        if contrast < 40:
            score -= 0.1
            recommendations.append("Low contrast. Consider enhancing before processing.")

        if not has_text:
            score -= 0.2
            recommendations.append("Insufficient text detected. Verify document content.")

        score = max(0.0, min(1.0, score))

        report = QualityReport(
            overall_score=round(score, 2),
            is_acceptable=score >= 0.6,
            blur_score=round(blur_score, 2),
            is_blurry=is_blurry,
            skew_angle=round(skew_angle, 2),
            resolution_dpi_estimate=dpi_estimate,
            noise_level=round(noise_level, 3),
            brightness=round(brightness, 1),
            contrast=round(contrast, 1),
            has_sufficient_text=has_text,
            recommendations=recommendations if recommendations else ["Image quality is acceptable"],
        )

        logger.info(f"Quality assessment: score={score:.2f}, acceptable={report.is_acceptable}")
        return report

    def _check_blur(self, gray: np.ndarray) -> tuple[float, bool]:
        """
        Detect blur using Laplacian variance.
        Higher variance = sharper image.
        """
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < self.blur_threshold
        return laplacian_var, is_blurry

    def _measure_skew(self, gray: np.ndarray) -> float:
        """Measure document skew angle using Hough Line Transform."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

        return float(np.median(angles)) if angles else 0.0

    def _estimate_dpi(self, img: np.ndarray) -> int:
        """
        Estimate DPI based on image dimensions.
        Assumes standard document sizes (A4, Letter).
        """
        h, w = img.shape[:2]
        # A4 is 8.27 x 11.69 inches
        dpi_w = w / 8.27
        dpi_h = h / 11.69
        return int(min(dpi_w, dpi_h))

    def _measure_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level using Laplacian standard deviation."""
        noise = cv2.Laplacian(gray, cv2.CV_64F).std()
        # Normalize to 0-1 range (higher = noisier)
        return min(noise / 50.0, 1.0)

    def _measure_brightness(self, gray: np.ndarray) -> float:
        """Measure average brightness."""
        return float(np.mean(gray))

    def _measure_contrast(self, gray: np.ndarray) -> float:
        """Measure contrast using standard deviation of pixel values."""
        return float(np.std(gray))

    def _check_text_presence(self, gray: np.ndarray) -> bool:
        """
        Check if sufficient text content is present using edge density.
        Text-rich areas have high edge density.
        """
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        return edge_density > 0.02  # At least 2% edge density
