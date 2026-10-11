"""Face analysis with OpenCV: detect a face, measure the skin colour, estimate a tone label.

Pipeline
  1. Haar-cascade face detection (largest face).
  2. Sample the cheek/forehead region and keep skin-coloured pixels (YCrCb range).
  3. Convert to CIELAB and run k-means (k=3); the biggest cluster is the dominant skin colour.
  4. Individual Typology Angle (ITA) from that colour -> one of 9 tone labels.

Approximate: brightness and camera colour balance affect the result.
"""
import cv2
import numpy as np

from .skin_hints import hints_from_face

TONES = ["Porcelain", "Fair", "Light", "Medium", "Olive", "Tan", "Deep", "Dark", "Ebony"]
_ITA_FLOOR = [60, 50, 41, 28, 19, 10, -10, -30, -90]  # ITA (degrees) above which each tone applies

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class NoFaceError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image")
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        s = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def rgb_hex_to_lab(hex_code: str) -> np.ndarray:
    """'#c68642' -> CIELAB (L 0-100, a, b)."""
    h = hex_code.lstrip("#")
    bgr = np.array([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]], dtype=np.float32) / 255
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]


def lab_to_hex(lab) -> str:
    bgr = cv2.cvtColor(np.array([[lab]], dtype=np.float32), cv2.COLOR_LAB2BGR)[0, 0]
    r, g, b = (int(round(float(np.clip(v, 0, 1)) * 255)) for v in bgr[::-1])
    return f"#{r:02x}{g:02x}{b:02x}"


def ita_to_tone(ita: float) -> str:
    for tone, floor in zip(TONES, _ITA_FLOOR, strict=True):
        if ita >= floor:
            return tone
    return TONES[-1]


def tone_index(tone: str | None) -> float:
    return float(TONES.index(tone)) if tone in TONES else float("nan")


def detect_face(img: np.ndarray):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        raise NoFaceError("No face found. Use a front-facing, well-lit photo.")
    return max(faces, key=lambda f: f[2] * f[3])


def dominant_skin_lab(img: np.ndarray, face) -> tuple[np.ndarray, int]:
    x, y, w, h = face
    roi = img[y + int(.35 * h): y + int(.8 * h), x + int(.2 * w): x + int(.8 * w)]
    mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb), (0, 133, 77), (255, 173, 127)) > 0
    pixels = roi[mask] if mask.sum() >= 50 else roi.reshape(-1, 3)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    k = min(3, len(lab))
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    biggest = np.bincount(labels.ravel()).argmax()
    return centers[biggest], int(mask.sum())


def estimate_skin_tone(img: np.ndarray) -> dict:
    face = detect_face(img)
    lab, n = dominant_skin_lab(img, face)
    ita = float(np.degrees(np.arctan2(lab[0] - 50, max(float(lab[2]), 1e-3))))
    return {"tone": ita_to_tone(ita), "ita": round(ita, 1), "skin_hex": lab_to_hex(lab),
            "lab": [round(float(v), 1) for v in lab], "face": [int(v) for v in face], "skin_pixels": n,
            "hints": hints_from_face(img, face)}
