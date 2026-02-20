"""
Fraud Detection Service for Banking Documents.
Uses computer vision techniques to detect tampered, forged,
or manipulated document images.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FraudReport:
    """Fraud detection analysis results."""
    is_suspicious: bool = False
    risk_level: str = "low"  # low, medium, high
    overall_score: float = 0.0  # 0 (clean) to 1 (likely fraud)
    checks: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


class FraudDetector:
    """
    Detects document fraud using computer vision analysis:
    - Copy-move forgery detection (ELA)
    - Edge inconsistency analysis
    - Noise level inconsistency
    - Metadata anomalies
    - Digital manipulation artifacts
    """

    async def analyze(self, image_bytes: bytes) -> FraudReport:
        """Run full fraud detection analysis."""
        report = FraudReport()
        score = 0.0

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            report.flags.append("UNREADABLE_IMAGE")
            report.is_suspicious = True
            report.risk_level = "high"
            return report

        # Check 1: Error Level Analysis (ELA)
        ela_suspicious, ela_detail = self._error_level_analysis(img)
        report.checks.append({"check": "Error Level Analysis", "passed": not ela_suspicious, "detail": ela_detail})
        if ela_suspicious:
            score += 0.3
            report.flags.append("ELA_ANOMALY")

        # Check 2: Noise inconsistency
        noise_suspicious, noise_detail = self._noise_inconsistency(img)
        report.checks.append({"check": "Noise Consistency", "passed": not noise_suspicious, "detail": noise_detail})
        if noise_suspicious:
            score += 0.25
            report.flags.append("NOISE_INCONSISTENCY")

        # Check 3: Edge analysis
        edge_suspicious, edge_detail = self._edge_analysis(img)
        report.checks.append({"check": "Edge Consistency", "passed": not edge_suspicious, "detail": edge_detail})
        if edge_suspicious:
            score += 0.2
            report.flags.append("EDGE_ANOMALY")

        # Check 4: Copy-move detection
        copy_suspicious, copy_detail = self._copy_move_detection(img)
        report.checks.append({"check": "Copy-Move Detection", "passed": not copy_suspicious, "detail": copy_detail})
        if copy_suspicious:
            score += 0.35
            report.flags.append("COPY_MOVE_DETECTED")

        # Check 5: JPEG artifact analysis
        jpeg_suspicious, jpeg_detail = self._jpeg_artifact_analysis(img)
        report.checks.append({"check": "Compression Artifacts", "passed": not jpeg_suspicious, "detail": jpeg_detail})
        if jpeg_suspicious:
            score += 0.15
            report.flags.append("COMPRESSION_ANOMALY")

        # Final scoring
        report.overall_score = min(score, 1.0)
        report.is_suspicious = report.overall_score >= 0.3

        if report.overall_score >= 0.6:
            report.risk_level = "high"
            report.recommendations.append("REJECT — Multiple fraud indicators. Escalate to fraud team.")
        elif report.overall_score >= 0.3:
            report.risk_level = "medium"
            report.recommendations.append("REVIEW — Some anomalies detected. Manual verification recommended.")
        else:
            report.risk_level = "low"
            report.recommendations.append("PASS — No significant fraud indicators detected.")

        logger.info(f"Fraud analysis: risk={report.risk_level}, score={report.overall_score:.2f}, flags={report.flags}")
        return report

    def _error_level_analysis(self, img: np.ndarray) -> tuple[bool, str]:
        """
        Error Level Analysis (ELA) — detects regions that were
        edited/pasted by looking at JPEG compression artifacts.
        Edited regions compress differently than original content.
        """
        # Simulate re-compression and compare
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        _, encoded = cv2.imencode(".jpg", img, encode_params)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Calculate difference
        diff = cv2.absdiff(img, recompressed)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Analyze variance of differences across regions
        h, w = diff_gray.shape
        block_size = 64
        variances = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = diff_gray[y:y + block_size, x:x + block_size]
                variances.append(np.var(block))

        if not variances:
            return False, "Image too small for ELA analysis"

        # High variance difference between blocks suggests editing
        var_range = max(variances) - min(variances)
        mean_var = np.mean(variances)

        suspicious = var_range > mean_var * 3.0

        detail = f"ELA variance range: {var_range:.2f}, mean: {mean_var:.2f}"
        return suspicious, detail

    def _noise_inconsistency(self, img: np.ndarray) -> tuple[bool, str]:
        """
        Check for noise level inconsistency across the image.
        Different noise patterns suggest parts of the image
        came from different sources (composite/paste).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        block_size = 64
        noise_levels = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y + block_size, x:x + block_size].astype(np.float64)
                # Estimate noise using Laplacian
                laplacian = cv2.Laplacian(block, cv2.CV_64F)
                noise = laplacian.std()
                noise_levels.append(noise)

        if len(noise_levels) < 4:
            return False, "Image too small for noise analysis"

        noise_std = np.std(noise_levels)
        noise_mean = np.mean(noise_levels)
        cv = noise_std / max(noise_mean, 0.001)  # Coefficient of variation

        suspicious = cv > 0.8  # High variation suggests inconsistency

        detail = f"Noise CV: {cv:.3f} (mean={noise_mean:.2f}, std={noise_std:.2f})"
        return suspicious, detail

    def _edge_analysis(self, img: np.ndarray) -> tuple[bool, str]:
        """
        Detect unnatural edges that may indicate pasting/splicing.
        Look for sharp edge discontinuities that don't match
        the document's natural structure.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect edges at multiple scales
        edges_fine = cv2.Canny(gray, 100, 200)
        edges_coarse = cv2.Canny(gray, 50, 100)

        # Compare edge density across quadrants
        h, w = gray.shape
        quadrants = [
            edges_fine[0:h // 2, 0:w // 2],
            edges_fine[0:h // 2, w // 2:],
            edges_fine[h // 2:, 0:w // 2],
            edges_fine[h // 2:, w // 2:],
        ]

        densities = [np.count_nonzero(q) / max(q.size, 1) for q in quadrants]
        density_range = max(densities) - min(densities)

        # Unusually high density difference could indicate splicing
        suspicious = density_range > 0.15

        detail = f"Edge density range: {density_range:.3f}, quadrants: {[f'{d:.3f}' for d in densities]}"
        return suspicious, detail

    def _copy_move_detection(self, img: np.ndarray) -> tuple[bool, str]:
        """
        Detect copy-move forgery using ORB feature matching.
        If large groups of features match within the same image,
        a region may have been duplicated to cover something.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 50:
            return False, "Insufficient features for copy-move analysis"

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des, des, k=3)

        # Self-matching: skip identity match (k=0), look at k=1,2
        suspicious_matches = 0
        min_distance_threshold = 30  # Feature distance threshold

        for match_group in matches:
            if len(match_group) >= 3:
                # Skip self-match (distance=0), check next matches
                m1, m2 = match_group[1], match_group[2]
                if m1.distance < min_distance_threshold:
                    # Check spatial distance (matching features should be far apart for copy-move)
                    pt1 = kp[m1.queryIdx].pt
                    pt2 = kp[m1.trainIdx].pt
                    spatial_dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                    if spatial_dist > 50:  # Features are far apart but look the same
                        suspicious_matches += 1

        ratio = suspicious_matches / max(len(matches), 1)
        suspicious = ratio > 0.05  # More than 5% suspicious matches

        detail = f"Copy-move matches: {suspicious_matches}/{len(matches)} ({ratio:.3f})"
        return suspicious, detail

    def _jpeg_artifact_analysis(self, img: np.ndarray) -> tuple[bool, str]:
        """
        Analyze JPEG blocking artifacts.
        Inconsistent block artifacts across the image suggest
        that parts were saved at different compression levels.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        # Analyze 8x8 block boundaries (JPEG uses 8x8 DCT blocks)
        horizontal_diffs = []
        vertical_diffs = []

        for y in range(8, h - 8, 8):
            row_diff = np.mean(np.abs(gray[y, :] - gray[y - 1, :]))
            horizontal_diffs.append(row_diff)

        for x in range(8, w - 8, 8):
            col_diff = np.mean(np.abs(gray[:, x] - gray[:, x - 1]))
            vertical_diffs.append(col_diff)

        if not horizontal_diffs or not vertical_diffs:
            return False, "Image too small for JPEG analysis"

        h_std = np.std(horizontal_diffs)
        v_std = np.std(vertical_diffs)
        avg_std = (h_std + v_std) / 2

        suspicious = avg_std > 5.0  # High variation in block boundaries

        detail = f"Block boundary std: h={h_std:.2f}, v={v_std:.2f}"
        return suspicious, detail
