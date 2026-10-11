"""Virtual try-on with OpenCV: recolour skin (foundation), lips and cheeks (blush) of a detected face.

Everything is done in CIELAB so shading and texture survive: foundation shifts the skin colour by a constant
(target shade minus the person's measured skin); lips keep their own lightness variation and take the target
chroma; blush is a soft radial blend on the cheeks. There is no facial-landmark model, so regions come from face-box
geometry plus colour: lips are the reddish blob in the mouth area, cheeks are fixed ellipses. Works best on
front-facing, evenly lit photos and is skipped (reported in `applied`) when a region cannot be found.
Eyeshadow is not supported (it needs eyelid landmarks).
"""
import cv2
import numpy as np

from .vision import detect_face, rgb_hex_to_lab


def _to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def _to_bgr(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255, 0, 255).astype(np.uint8)


def _face_masks(img: np.ndarray, face, lab: np.ndarray):
    """(skin mask 0-1, face ellipse mask bool) over the whole image."""
    x, y, w, h = face
    ell = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(ell, (x + w // 2, y + h // 2), (int(0.46 * w), int(0.56 * h)), 0, 0, 360, 1, -1)
    skin = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), (0, 133, 70), (255, 190, 135)) > 0
    return skin & (ell > 0), ell > 0


def lip_mask(img: np.ndarray, face, lab: np.ndarray, skin: np.ndarray) -> np.ndarray | None:
    """Reddish blob in the mouth area (a* clearly above the skin's), or None if not found."""
    x, y, w, h = face
    roi = np.zeros(img.shape[:2], bool)
    roi[y + int(0.66 * h): y + int(0.92 * h), x + int(0.25 * w): x + int(0.75 * w)] = True
    if skin.sum() < 50:
        return None
    ref_a = float(np.median(lab[..., 1][skin]))
    cand = (roi & (lab[..., 1] > ref_a + 6)).astype(np.uint8)
    k = max(3, int(0.02 * w) | 1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 0.004 * w * h:
        return None
    return labels == best


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma), 0, 1)


def apply_look(img: np.ndarray, foundation_hex: str | None = None, lips_hex: str | None = None,
               blush_hex: str | None = None, strength: float = 0.8) -> tuple[np.ndarray, list[str], tuple]:
    """Returns (image, list of what was applied, face box). Raises NoFaceError via detect_face."""
    face = detect_face(img)
    x, y, w, h = face
    lab = _to_lab(img)
    out = lab.copy()
    skin, ellipse = _face_masks(img, face, lab)
    lips = lip_mask(img, face, lab, skin)
    applied: list[str] = []
    sigma = max(1.0, 0.012 * w)

    if foundation_hex and skin.sum() >= 50:
        base = skin.copy()
        if lips is not None:
            base &= ~cv2.dilate(lips.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        m = _feather(base, sigma * 2) * strength
        delta = rgb_hex_to_lab(foundation_hex) - np.median(lab[base].reshape(-1, 3), axis=0)
        out += m[..., None] * delta
        applied.append("foundation")

    if lips_hex and lips is not None:
        m = _feather(cv2.dilate(lips.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool), sigma) * strength
        target = rgb_hex_to_lab(lips_hex)
        out[..., 1] += m * (target[1] - lab[..., 1]) * 0.9
        out[..., 2] += m * (target[2] - lab[..., 2]) * 0.9
        out[..., 0] += m * (target[0] - float(np.median(lab[..., 0][lips]))) * 0.5
        applied.append("lips")

    if blush_hex:
        target = rgb_hex_to_lab(blush_hex)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
        alpha = np.zeros(img.shape[:2], np.float32)
        for cx in (x + 0.27 * w, x + 0.73 * w):
            r2 = ((xx - cx) / (0.14 * w)) ** 2 + ((yy - (y + 0.60 * h)) / (0.09 * h)) ** 2
            alpha = np.maximum(alpha, np.exp(-1.5 * r2))
        alpha *= _feather(skin | ellipse, sigma) * 0.55 * strength
        alpha[alpha < 0.02] = 0          # no faint halo reaching other features
        if lips is not None:
            alpha *= 1 - cv2.dilate(lips.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
        out[..., 1] += alpha * (target[1] - lab[..., 1]) * 0.8
        out[..., 2] += alpha * (target[2] - lab[..., 2]) * 0.8
        out[..., 0] += alpha * (target[0] - lab[..., 0]) * 0.25
        applied.append("blush")
    # Write back only the pixels that changed: the Lab round trip alone would nudge every pixel by +-1.
    changed = np.abs(out - lab).max(axis=2) > 0.01
    result = img.copy()
    result[changed] = _to_bgr(out)[changed]
    return result, applied, tuple(int(v) for v in face)
