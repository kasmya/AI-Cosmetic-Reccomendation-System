"""Rough skin-condition hints from a face photo (OpenCV). LOW CONFIDENCE, not a diagnosis.

All three measures are relative to the person's own skin in the same photo, so they do not depend on skin tone:
  redness    share of cheek pixels noticeably redder (Lab a*) than that person's median skin
  shine      share of forehead/nose pixels that are near-white and unsaturated (specular highlights)
  blemishes  count of small spots redder than their surroundings (difference of Gaussians on a*)
Lighting, make-up, filters and image quality all change these numbers. There is no labelled data behind the
thresholds: they were chosen by hand and checked only on synthetic patches. Use as a weak nudge.
"""
import cv2
import numpy as np

FACE_SIZE = 200  # faces are resized to a fixed square so pixel thresholds mean the same thing for every photo


def _level(score: float) -> str:
    return "low" if score < 0.33 else "moderate" if score < 0.66 else "high"


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Skin-coloured pixels, with a deliberately generous red limit so reddened skin is still counted as skin."""
    return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb), (0, 133, 70), (255, 190, 135)) > 0


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def redness_score(patches: list[np.ndarray]) -> dict:
    a = np.concatenate([_lab(p)[..., 1][_skin_mask(p)] for p in patches])
    if len(a) < 100:
        return {"score": 0.0, "level": "low"}
    frac = float((a > np.median(a) + 8).mean())
    score = float(np.clip(frac / 0.2, 0, 1))
    return {"score": round(score, 2), "level": _level(score)}


def shine_score(tzone: np.ndarray) -> dict:
    """Specular highlights: near-white, unsaturated pixels that are much brighter than the skin around them.
    Judging against the local surroundings (not absolute brightness) keeps bright but matte photos from scoring high."""
    hsv = cv2.cvtColor(tzone, cv2.COLOR_BGR2HSV).astype(np.float32)
    v, sat = hsv[..., 2], hsv[..., 1]
    peak = (v - cv2.GaussianBlur(v, (0, 0), 10)) > 0.08 * 255
    highlight = peak & (v > 0.90 * 255) & (sat < 0.20 * 255)
    score = float(np.clip(highlight.mean() / 0.04, 0, 1))
    return {"score": round(score, 2), "level": _level(score)}


def blemish_score(patches: list[np.ndarray]) -> dict:
    count = 0
    for p in patches:
        a = _lab(p)[..., 1]
        diff = cv2.GaussianBlur(a, (0, 0), 1.0) - cv2.GaussianBlur(a, (0, 0), 7.0)
        spots = ((diff > 6) & _skin_mask(p)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(spots)
        count += int(sum(6 <= stats[i, cv2.CC_STAT_AREA] <= 120 for i in range(1, n)))
    score = float(np.clip(count / 10, 0, 1))
    return {"score": round(score, 2), "level": _level(score), "count": count}


def hints_from_face(img: np.ndarray, face) -> dict:
    x, y, w, h = face
    f = cv2.resize(img[y:y + h, x:x + w], (FACE_SIZE, FACE_SIZE))
    cheeks = [f[90:140, 35:80], f[90:140, 120:165]]            # below the eyes, above the mouth, off the nose
    tzone = np.vstack([f[20:60, 60:140], np.pad(f[70:130, 85:115], ((0, 0), (25, 25), (0, 0)), mode="edge")])
    return {"redness": redness_score(cheeks), "shine": shine_score(tzone), "blemishes": blemish_score(cheeks),
            "confidence": "low",
            "note": "Rough hints from one photo; lighting and make-up change them. Not a diagnosis."}
