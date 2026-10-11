"""Skin-tone estimation from a face photo with OpenCV.

Pipeline: Haar-cascade face detection -> sample the cheek/forehead
region -> keep skin-coloured pixels (YCrCb range) -> median CIELAB colour -> Individual
Typology Angle (ITA) -> dataset tone label. Approximate: brightness and camera colour
balance affect the result (no white-balance correction is attempted, because scene-wide
heuristics such as gray-world fail on photos with dark or coloured backgrounds), so treat it as a suggestion the user can ignore.
"""
import cv2
import numpy as np

# Dataset tone labels ordered light -> dark, with the ITA (degrees) above which each applies.
TONES = ["Porcelain", "Fair", "Light", "Medium", "Olive", "Tan", "Deep", "Dark", "Ebony"]
_ITA_FLOOR = [60, 50, 41, 28, 19, 10, -10, -30, -90]

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class NoFaceError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image")
    h, w = img.shape[:2]
    if max(h, w) > 1200:  # keep detection fast
        s = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def ita_to_tone(ita: float) -> str:
    for tone, floor in zip(TONES, _ITA_FLOOR):
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
    return max(faces, key=lambda f: f[2] * f[3])  # largest face


def estimate_skin_tone(img: np.ndarray) -> dict:
    x, y, w, h = detect_face(img)
    roi = img[y + int(.35 * h): y + int(.8 * h), x + int(.2 * w): x + int(.8 * w)]
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127)) > 0
    pixels = roi[mask] if mask.sum() >= 50 else roi.reshape(-1, 3)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    L = float(np.median(lab[:, 0])) * 100 / 255      # OpenCV scales L to 0-255
    b = float(np.median(lab[:, 2])) - 128            # and offsets b by 128
    ita = float(np.degrees(np.arctan2(L - 50, max(b, 1e-3))))
    return {"tone": ita_to_tone(ita), "ita": round(ita, 1), "face": [int(v) for v in (x, y, w, h)],
            "skin_pixels": int(mask.sum())}
