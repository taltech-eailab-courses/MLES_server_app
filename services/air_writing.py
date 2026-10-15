"""Air-writing: one request per character -> 2D character image -> (last request) word image -> OCR.

Request (POST /api/v1/air_writing, JSON), same shape as /ocr_chunked:
    {"request_id": "0a1b2c3d", "packet_index": 0, "total_packets": 5, "x": "x0,y0,z0,x1,y1,z1,..."}
  request_id     unique string of one word (device session)
  packet_index   0-based index of this character
  total_packets  number of characters of the word
  x              3D trajectory of the character in metres, flattened x,y,z triples, comma separated,
                 world frame: Z up, X/Y horizontal; the writer looks along +Y (reader on the -Y side)

For every request the trajectory is turned into a 2D image of the character (PCA plane, orientation
resolved from the frame above) and stored as .tmp/buffers/{request_id}/{packet_index}.png.
When packet_index == total_packets - 1 arrives, the stored character images are combined into one word
image (same height, bottoms aligned, gap between them, never connected) and the OCR model is called.
Each character is one continuous stroke (no pen-up information); equal height + common baseline suit
uppercase letters only.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from services import temp_file_manager

OCR_HEIGHT, OCR_WIDTH = 60, 180            # input of models.text_recognition.ocr_predict
CHAR_HEIGHT_PX = 120                       # character images are drawn large, the word is scaled down once
LINE_PX = 7
PAD_PX = 12
GAP_RATIO = 0.3                            # gap between characters / character height
MARGIN_PX = 6                              # margin inside the 60x180 OCR image
MIN_EXTENT_M, MAX_EXTENT_M = 0.02, 10.0
MIN_POINTS = 8
MAX_POINTS = 2000
MAX_CHARS = 5
MAX_X_CHARS = 300_000                      # size limit of the x string
COLLINEAR_RATIO = 0.05
HORIZONTAL_COS = 0.8
UP = np.array([0.0, 0.0, 1.0])             # "up" of the writing in the world frame
VIEWER = np.array([0.0, -1.0, 0.0])        # from the drawing towards the reader
KEEP_SECONDS = 3600
INVERT = os.environ.get("AW_OCR_INVERT", "0") == "1"   # white strokes on black instead
OCR_UPSCALE = int(os.environ.get("AW_OCR_UPSCALE", "2"))

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")     # becomes a directory name: no path characters
_lock = threading.Lock()


class AirWritingError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra


# ---------------------------------------------------------------------------- parsing / projection
def parse_trajectory(x: str) -> np.ndarray:
    if len(x) > MAX_X_CHARS:
        raise AirWritingError("x_too_large", "x is too long", 413)
    try:
        values = np.array([float(v) for v in x.split(",") if v.strip() != ""], dtype=np.float64)
    except ValueError:
        raise AirWritingError("bad_x", "x must be comma separated numbers") from None
    if len(values) % 3 != 0:
        raise AirWritingError("bad_x", f"x has {len(values)} values; expected x,y,z triples")
    pts = values.reshape(-1, 3)
    if not MIN_POINTS <= len(pts) <= MAX_POINTS:
        raise AirWritingError("bad_point_count", f"{len(pts)} points; expected {MIN_POINTS}..{MAX_POINTS}")
    if not np.isfinite(pts).all():
        raise AirWritingError("bad_x", "x contains NaN/inf")
    return pts


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise AirWritingError("degenerate_plane", "cannot determine the drawing plane", 422)
    return v / n


def project_to_2d(position, plane):
    if plane == "pca":
        # Best-fitting plane through the reconstructed trajectory.
        centered = position - position.mean(axis=0)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        return centered @ basis[:2].T

    # Each axis letter may be preceded by "-" to negate that axis, e.g. "-xy" flips x -> -x.
    match = re.fullmatch(r"(-?[xyz])(-?[xyz])", plane)
    if not match:
        raise KeyError(plane)
    axis_index = {"x": 0, "y": 1, "z": 2}
    cols = [axis_index[token[-1]] for token in match.groups()]
    flips = [-1.0 if token[0] == "-" else 1.0 for token in match.groups()]
    return position[:, cols] * np.array(flips)


# ---------------------------------------------------------------------------- 2D character image
def character_image(points3d: np.ndarray) -> np.ndarray:
    """Tight grayscale image of one character: black stroke on white, fixed height CHAR_HEIGHT_PX
    (aspect ratio preserved), the baseline at the bottom edge."""
    import cv2

    extent = float(np.linalg.norm(np.ptp(points3d, axis=0)))
    if extent < MIN_EXTENT_M:
        raise AirWritingError("character_too_small", f"extent {extent * 100:.1f} cm: no gesture detected, "
                              "draw a larger character", 422)
    # if extent > MAX_EXTENT_M:
    #     raise AirWritingError("character_too_large", f"extent {extent:.1f} m is implausible (tracking drift); "
    #                           "hold still at the end and repeat", 422)
    p = project_to_2d(points3d, "-xy")
    lo = p.min(axis=0)
    size = p.max(axis=0) - lo
    if size[1] <= 1e-9:
        raise AirWritingError("no_vertical_extent", "the character has no vertical extent", 422)
    scale = CHAR_HEIGHT_PX / size[1]
    width = int(np.ceil(size[0] * scale))
    img = np.full((CHAR_HEIGHT_PX + 2 * PAD_PX, width + 2 * PAD_PX), 255, dtype=np.uint8)
    px = (p[:, 0] - lo[0]) * scale + PAD_PX
    py = PAD_PX + CHAR_HEIGHT_PX - (p[:, 1] - lo[1]) * scale      # image rows grow downwards
    poly = np.round(np.column_stack([px, py]) * 16).astype(np.int32)
    cv2.polylines(img, [poly.reshape(-1, 1, 2)], False, 0, thickness=LINE_PX, lineType=cv2.LINE_AA, shift=4)
    return img


# ---------------------------------------------------------------------------- word image
def combine_images(images: List[np.ndarray]) -> np.ndarray:
    """Character images (same height) -> one (60, 180) word image, characters left to right, separated
    by a white gap (never connected), scaled uniformly to fit the OCR image."""
    import cv2

    gap = np.full((images[0].shape[0], int(GAP_RATIO * CHAR_HEIGHT_PX)), 255, dtype=np.uint8)
    parts = []
    for i, im in enumerate(images):
        if i:
            parts.append(gap)
        parts.append(im)
    word = np.hstack(parts)
    h, w = word.shape
    scale = min((OCR_WIDTH - 2 * MARGIN_PX) / w, (OCR_HEIGHT - 2 * MARGIN_PX) / h)
    small = cv2.resize(word, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((OCR_HEIGHT, OCR_WIDTH), 255, dtype=np.uint8)
    y0, x0 = (OCR_HEIGHT - small.shape[0]) // 2, (OCR_WIDTH - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return 255 - canvas if INVERT else canvas


# ---------------------------------------------------------------------------- OCR
def _default_predict() -> Callable[[np.ndarray], str]:
    """models.text_recognition.ocr_predict (pytesseract, the existing OCR endpoint). If the models
    package cannot be imported (torch/onnx missing, e.g. a light test environment) the same call is used directly."""
    try:
        from models.text_recognition import ocr_predict
        return ocr_predict
    except ImportError:
        import pytesseract
        return lambda img: pytesseract.image_to_string(img, config=r"--oem 3 --psm 6").replace("\n", "")


def run_ocr(image: np.ndarray, predict: Optional[Callable[[np.ndarray], str]] = None) -> str:
    """Tesseract needs larger glyphs than 60 px: the 60x180 image is enlarged and given a border inside this
    function; the 60x180 array contract of the OCR endpoint is unchanged."""
    import cv2

    big = cv2.resize(image, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC) if OCR_UPSCALE > 1 else image
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0 if INVERT else 255)
    text = (predict or _default_predict())(big)
    return re.sub(r'[^\x20-\x7E]|["\\]', "", text.replace("\n", "")).strip()[:24]


# ---------------------------------------------------------------------------- request handling
def _dir(request_id: str) -> str:
    return os.path.join(temp_file_manager.BASE_DIR, request_id)


def _cleanup_old() -> None:
    base = temp_file_manager.BASE_DIR
    if not os.path.isdir(base):
        return
    now = time.time()
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if os.path.isdir(path) and now - os.path.getmtime(path) > KEEP_SECONDS:
            shutil.rmtree(path, ignore_errors=True)


def handle_packet(request_id: str, packet_index: int, total_packets: int, x: str,
                  predict: Optional[Callable[[np.ndarray], str]] = None) -> str:
    """Processes one character. Returns a status text, or the recognised word for the last packet.
    Re-sending a packet overwrites its image (idempotent); re-sending the last one returns the cached text."""
    import cv2

    if not _REQUEST_ID.match(request_id):
        raise AirWritingError("bad_request_id", "request_id must be 1-64 characters of A-Z a-z 0-9 _ -")
    if not 1 <= total_packets <= MAX_CHARS:
        raise AirWritingError("bad_total_packets", f"total_packets must be 1..{MAX_CHARS}")
    if not 0 <= packet_index < total_packets:
        raise AirWritingError("bad_packet_index", f"packet_index must be 0..{total_packets - 1}")

    img = character_image(parse_trajectory(x))            # validates and builds the 2D image

    with _lock:
        _cleanup_old()
        d = _dir(request_id)
        os.makedirs(d, exist_ok=True)
        meta = os.path.join(d, "total.txt")
        if os.path.exists(meta):
            if int(open(meta).read()) != total_packets:
                raise AirWritingError("total_packets_mismatch", "total_packets differs from the first packet", 409)
        else:
            with open(meta, "w") as f:
                f.write(str(total_packets))
        result = os.path.join(d, "result.txt")
        if os.path.exists(result) and packet_index == total_packets - 1:
            return open(result).read()                     # retry of the last packet: cached answer
        tmp, final = os.path.join(d, f"{packet_index}.tmp.png"), os.path.join(d, f"{packet_index}.png")
        cv2.imwrite(tmp, img)
        os.replace(tmp, final)

        if packet_index < total_packets - 1:
            return f"Packet {packet_index + 1}/{total_packets} received."

        missing = [i for i in range(total_packets) if not os.path.exists(os.path.join(d, f"{i}.png"))]
        if missing:
            raise AirWritingError("missing_packets", f"packets {missing} have not been received", 409, missing=missing)
        images = [cv2.imread(os.path.join(d, f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(total_packets)]
        word = combine_images(images)
        cv2.imwrite(os.path.join(d, "word.png"), word)    # what the OCR model sees (debugging)
        text = run_ocr(word, predict)
        with open(result, "w") as f:
            f.write(text)
        return text
