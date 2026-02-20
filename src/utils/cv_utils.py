"""OpenCV utility functions for banking document image processing."""

import cv2
import numpy as np


def bytes_to_cv2(image_bytes: bytes, grayscale: bool = False) -> np.ndarray | None:
    """Convert raw bytes to OpenCV image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(nparr, flag)


def cv2_to_bytes(img: np.ndarray, fmt: str = ".png") -> bytes:
    """Convert OpenCV image to bytes."""
    _, buffer = cv2.imencode(fmt, img)
    return buffer.tobytes()


def auto_deskew(img: np.ndarray) -> np.ndarray:
    """Auto-correct document skew using Hough lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return img

    angles = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
              for l in lines if abs(np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) < 45]

    if not angles:
        return img

    angle = np.median(angles)
    if abs(angle) < 0.5:
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for OCR: denoise + enhance + sharpen."""
    # Bilateral filter preserves edges
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # CLAHE contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return img


def extract_roi(img: np.ndarray, x_pct: float, y_pct: float, w_pct: float, h_pct: float) -> np.ndarray:
    """Extract region of interest by percentage coordinates."""
    h, w = img.shape[:2]
    x1 = int(w * x_pct)
    y1 = int(h * y_pct)
    x2 = int(w * (x_pct + w_pct))
    y2 = int(h * (y_pct + h_pct))
    return img[y1:y2, x1:x2]


def compute_image_hash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) for image deduplication."""
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return "".join(str(int(b)) for b in diff.flatten())
