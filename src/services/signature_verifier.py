"""
Signature Verification Service.
Detects signature regions, extracts features using ORB,
and compares against reference signatures for banking use cases.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SignatureResult:
    """Signature verification result."""
    signature_detected: bool = False
    signature_region: list = None  # [x, y, w, h]
    similarity_score: float = 0.0  # 0-1, how similar to reference
    match_verdict: str = ""  # "match", "mismatch", "inconclusive"
    keypoints_found: int = 0
    keypoints_matched: int = 0
    confidence: float = 0.0
    warnings: list = None

    def __post_init__(self):
        if self.signature_region is None:
            self.signature_region = []
        if self.warnings is None:
            self.warnings = []


class SignatureVerifier:
    """
    Signature detection and verification using OpenCV feature matching.
    
    Pipeline:
    1. Detect signature region (bottom-right quadrant analysis)
    2. Extract signature using contour detection
    3. Extract ORB features from signature
    4. Match against reference signature using BFMatcher
    5. Score similarity and determine match/mismatch
    """

    def __init__(self, min_match_ratio: float = 0.25):
        self.min_match_ratio = min_match_ratio
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    async def verify(
        self, document_bytes: bytes, reference_bytes: bytes | None = None
    ) -> SignatureResult:
        """
        Detect and optionally verify a signature.

        Args:
            document_bytes: Document image containing signature
            reference_bytes: Reference signature image for comparison

        Returns:
            SignatureResult with detection and match info
        """
        result = SignatureResult()

        # Step 1: Detect signature region
        sig_region, sig_image = self._detect_signature_region(document_bytes)
        if sig_image is None:
            result.signature_detected = False
            result.match_verdict = "no_signature"
            result.warnings.append("No signature region detected")
            return result

        result.signature_detected = True
        result.signature_region = sig_region

        # Step 2: Extract features from detected signature
        sig_processed = self._preprocess_signature(sig_image)
        kp1, des1 = self.orb.detectAndCompute(sig_processed, None)
        result.keypoints_found = len(kp1) if kp1 else 0

        if result.keypoints_found < 10:
            result.warnings.append("Too few signature features detected")
            result.confidence = 0.3
            result.match_verdict = "inconclusive"
            return result

        # Step 3: Compare with reference if provided
        if reference_bytes is not None:
            ref_img = self._load_image(reference_bytes, grayscale=True)
            if ref_img is not None:
                ref_processed = self._preprocess_signature(ref_img)
                kp2, des2 = self.orb.detectAndCompute(ref_processed, None)

                if des2 is not None and len(kp2) >= 10:
                    similarity = self._match_features(des1, des2)
                    result.similarity_score = similarity
                    result.keypoints_matched = int(similarity * min(len(kp1), len(kp2)))

                    if similarity >= 0.4:
                        result.match_verdict = "match"
                        result.confidence = min(0.95, 0.5 + similarity)
                    elif similarity >= self.min_match_ratio:
                        result.match_verdict = "inconclusive"
                        result.confidence = 0.4 + similarity * 0.3
                    else:
                        result.match_verdict = "mismatch"
                        result.confidence = 0.7
                        result.warnings.append("Signature does not match reference")
                else:
                    result.match_verdict = "inconclusive"
                    result.warnings.append("Reference signature has insufficient features")
            else:
                result.match_verdict = "inconclusive"
                result.warnings.append("Could not load reference signature image")
        else:
            result.match_verdict = "detected_only"
            result.confidence = 0.8

        logger.info(
            f"Signature verification: {result.match_verdict} "
            f"(similarity={result.similarity_score:.2f}, keypoints={result.keypoints_found})"
        )
        return result

    def _detect_signature_region(self, image_bytes: bytes) -> tuple[list, np.ndarray | None]:
        """
        Detect signature region using contour analysis.
        Signatures are typically in the bottom-right of banking documents.
        """
        img = self._load_image(image_bytes, grayscale=True)
        if img is None:
            return [], None

        h, w = img.shape

        # Focus on bottom-right quadrant (where signatures usually are)
        roi_y_start = int(h * 0.65)
        roi_x_start = int(w * 0.4)
        roi = img[roi_y_start:, roi_x_start:]

        # Adaptive threshold to find ink marks
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to connect signature strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for signature-like contours
        sig_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            x, y, cw, ch = cv2.boundingRect(c)
            aspect_ratio = cw / max(ch, 1)

            # Signature characteristics: wider than tall, medium area
            if (area > 500 and area < roi.size * 0.4 and
                    aspect_ratio > 1.5 and aspect_ratio < 10):
                sig_contours.append(c)

        if not sig_contours:
            # Fallback: look for any ink concentration
            ink_density = np.count_nonzero(binary) / binary.size
            if ink_density > 0.01:
                # Return the whole ROI as signature region
                return [roi_x_start, roi_y_start, w - roi_x_start, h - roi_y_start], roi
            return [], None

        # Combine all signature contours into one bounding box
        all_points = np.vstack(sig_contours)
        x, y, sw, sh = cv2.boundingRect(all_points)

        # Map back to full image coordinates
        abs_x = roi_x_start + x
        abs_y = roi_y_start + y

        # Extract signature image with padding
        pad = 10
        sig_img = img[
            max(0, abs_y - pad):min(h, abs_y + sh + pad),
            max(0, abs_x - pad):min(w, abs_x + sw + pad)
        ]

        return [abs_x, abs_y, sw, sh], sig_img

    def _preprocess_signature(self, img: np.ndarray) -> np.ndarray:
        """Preprocess signature image for feature extraction."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to standard size for consistent feature extraction
        img = cv2.resize(img, (300, 150), interpolation=cv2.INTER_CUBIC)

        # Binarize
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Thin the strokes for better feature matching
        kernel = np.ones((2, 2), np.uint8)
        thinned = cv2.erode(binary, kernel, iterations=1)

        return thinned

    def _match_features(self, des1: np.ndarray, des2: np.ndarray) -> float:
        """Match ORB descriptors and return similarity score."""
        if des1 is None or des2 is None:
            return 0.0

        matches = self.bf_matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        max_possible = min(len(des1), len(des2))
        if max_possible == 0:
            return 0.0

        return len(good_matches) / max_possible

    def _load_image(self, image_bytes: bytes, grayscale: bool = False) -> np.ndarray | None:
        """Load image from bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        return cv2.imdecode(nparr, flag)
