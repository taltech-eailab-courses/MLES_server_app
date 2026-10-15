"""Tests of the air-writing endpoint logic (services/air_writing.py); no hardware, no running server.

    pip install pytest numpy opencv-python-headless pytesseract httpx
    pytest tests/test_air_writing.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BUFFER_DIR", tempfile.mkdtemp(prefix="aw_buffers_"))

from services import air_writing as aw  # noqa: E402
from services import temp_file_manager  # noqa: E402

GLYPHS = {  # single-stroke glyphs in a unit box (x right, y up)
    "L": [(0, 1), (0, 0), (.6, 0)],
    "T": [(0, 1), (.7, 1), (.35, 1), (.35, 0)],
    "A": [(0, 0), (.35, 1), (.7, 0), (.55, .35), (.15, .35)],
    "O": [(.4 + .4 * np.sin(t), .5 + .5 * np.cos(t)) for t in np.linspace(0, 2 * np.pi, 40)],
    "I": [(0, 1), (0, 0)],
}


def densify(poly, n=90):
    poly = np.asarray(poly, dtype=float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(u, s, poly[:, 0]), np.interp(u, s, poly[:, 1])])


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.deg2rad(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def glyph_3d(letter, size=0.15, yaw=0.0, pitch=0.0, offset=(0.3, 0.2, 0.1)):
    """Glyph in the vertical X-Z plane (reader on the -Y side), optionally yawed/pitched."""
    g = densify(GLYPHS[letter]) * size
    p = np.column_stack([g[:, 0], np.zeros(len(g)), g[:, 1]])
    return p @ (rot((0, 0, 1), yaw) @ rot((1, 0, 0), pitch)).T + np.asarray(offset)



def to_x(points):
    """Request field `x`: flattened x,y,z triples, comma separated (what the device sends)."""
    return ",".join(f"{v:.5f}" for v in np.asarray(points).ravel())


@pytest.fixture(autouse=True)
def buffers(tmp_path, monkeypatch):
    monkeypatch.setattr(temp_file_manager, "BASE_DIR", str(tmp_path))
    return tmp_path


def traj(letter, **kw):
    return glyph_3d(letter, **kw)


def similar(a, b, tol=1e-4):
    return np.abs((a - a.mean(axis=0)) - (b - b.mean(axis=0))).max() < tol


def fake_ocr(text):
    return lambda image: text


# ------------------------------------------------------------------ x parsing
def test_parse_trajectory_round_trip():
    pts = traj("L")
    assert np.allclose(aw.parse_trajectory(to_x(pts)), pts, atol=1e-5)


@pytest.mark.parametrize("x,code", [
    ("1,2,3,4", "bad_x"), ("a,b,c", "bad_x"), ("1,2,3," * 3, "bad_point_count"), ("", "bad_point_count"),
    (",".join(["nan"] * 30), "bad_x"), ("1," * 300_000, "x_too_large"),
])
def test_parse_trajectory_rejects_bad_x(x, code):
    with pytest.raises(aw.AirWritingError) as e:
        aw.parse_trajectory(x)
    assert e.value.code == code


# ------------------------------------------------------------------ projection / orientation
@pytest.mark.parametrize("letter", ["L", "T", "A", "O"])
@pytest.mark.parametrize("yaw,pitch", [(0, 0), (35, 0), (-40, 0), (0, 25), (-30, -20)])
def test_projection_recovers_upright_unmirrored_glyph(letter, yaw, pitch):
    """The glyph is drawn in a plane that is yawed/pitched relative to the world frame; PCA plane +
    up/viewer reference must give back the original, upright, non-mirrored glyph."""
    size = 0.15
    expected = densify(GLYPHS[letter]) * size
    got = project_ok = aw.project_to_plane(traj(letter, size=size, yaw=yaw, pitch=pitch))
    assert similar(got, expected, 2e-4), f"{letter} yaw={yaw} pitch={pitch}"
    del project_ok


def test_projection_not_mirrored_for_asymmetric_glyph():
    """'L' mirrored would have its foot to the left."""
    p = aw.project_to_plane(traj("L"))
    assert p[0, 1] > p[-1, 1] + 0.1       # drawn top -> bottom
    assert p[-1, 0] > p[0, 0] + 0.05      # the foot ends on the right


def test_projection_horizontal_plane():
    """Writing on a table: glyph tops point away from the writer (+Y), reader looks from above."""
    g = densify(GLYPHS["L"]) * 0.15
    pts = np.column_stack([g[:, 0], g[:, 1], np.zeros(len(g))]) + (0.2, 0.2, 0.0)
    assert similar(aw.project_to_plane(pts), g, 2e-4)


def test_projection_collinear_stroke():
    t = traj("I", yaw=20)
    p = aw.project_to_plane(t)
    assert abs(np.ptp(p[:, 0])) < 1e-6 and np.ptp(p[:, 1]) == pytest.approx(0.15, abs=1e-4)
    assert p[0, 1] > p[-1, 1]  # starts at the top


# ------------------------------------------------------------------ 2D images
def test_character_image_shape_and_baseline():
    img = aw.character_image(traj("T"))
    assert img.dtype == np.uint8 and img.shape[0] == aw.CHAR_HEIGHT_PX + 2 * aw.PAD_PX
    ink_rows = np.flatnonzero((img < 128).any(axis=1))
    assert ink_rows.min() <= aw.PAD_PX + aw.LINE_PX and ink_rows.max() >= aw.PAD_PX + aw.CHAR_HEIGHT_PX - aw.LINE_PX
    assert img[0, 0] == 255


def test_character_image_rejects_tiny_and_huge():
    with pytest.raises(aw.AirWritingError) as e:
        aw.character_image(traj("L", size=0.002))
    assert e.value.code == "character_too_small"
    with pytest.raises(aw.AirWritingError) as e:
        aw.character_image(traj("L", size=20.0))
    assert e.value.code == "character_too_large"


def test_combine_images_order_gaps_and_size():
    imgs = [aw.character_image(traj(c)) for c in "LLL"]
    word = aw.combine_images(imgs)
    assert word.shape == (60, 180) and word.dtype == np.uint8 and word[0, 0] == 255
    cols = (word < 128).any(axis=0)
    runs = np.flatnonzero(np.diff(np.concatenate([[0], cols.astype(int), [0]])))
    assert len(runs) // 2 == 3, "three separate characters expected (never connected)"
    a = aw.combine_images([aw.character_image(traj("L")), aw.character_image(traj("T"))])
    b = aw.combine_images([aw.character_image(traj("T")), aw.character_image(traj("L"))])
    assert not np.array_equal(a, b)
    five = aw.combine_images([aw.character_image(traj("O"))] * 5)  # widest case still fits the OCR image
    assert five.shape == (60, 180) and (five < 128).any(axis=0)[aw.MARGIN_PX - 1] == False  # noqa: E712


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_real_ocr_on_clean_glyphs():
    pytest.importorskip("pytesseract")
    import pytesseract
    word = aw.combine_images([aw.character_image(traj(c)) for c in "ALL"])
    text = aw.run_ocr(word, lambda a: pytesseract.image_to_string(a, config="--oem 3 --psm 6"))
    assert text == "ALL"


# ------------------------------------------------------------------ request handling
def send(request_id, index, total, letter="L", predict=None, **kw):
    return aw.handle_packet(request_id, index, total, to_x(traj(letter, **kw)), predict)


def test_full_word_creates_temp_files_and_returns_ocr(buffers):
    for i, c in enumerate("ALT"):
        out = send("req_1", i, 3, c, predict=fake_ocr("ALT"))
        if i < 2:
            assert out == f"Packet {i + 1}/3 received."
            assert (buffers / "req_1" / f"{i}.png").exists()   # intermediate file per request_id
    assert out == "ALT"
    assert (buffers / "req_1" / "word.png").exists()


def test_retry_of_a_packet_is_idempotent_and_last_answer_is_cached(buffers):
    calls = []
    ocr = lambda image: calls.append(1) or "LOL"  # noqa: E731
    send("r", 0, 2, "L", ocr)
    send("r", 0, 2, "L", ocr)                      # the response was lost: same packet again
    assert sorted(p.name for p in (buffers / "r").glob("*.png")) == ["0.png"]
    assert send("r", 1, 2, "O", ocr) == "LOL"
    assert send("r", 1, 2, "O", ocr) == "LOL" and len(calls) == 1   # cached, OCR not run twice


def test_last_packet_before_the_others_reports_missing(buffers):
    with pytest.raises(aw.AirWritingError) as e:
        send("m", 2, 3, "T", fake_ocr("x"))
    assert e.value.code == "missing_packets" and e.value.extra["missing"] == [0, 1] and e.value.status == 409
    send("m", 0, 3, "A", fake_ocr("ALT"))
    send("m", 1, 3, "L", fake_ocr("ALT"))
    assert send("m", 2, 3, "T", fake_ocr("ALT")) == "ALT"   # works once the rest arrived (any order)


@pytest.mark.parametrize("rid,idx,total,code", [
    ("../etc", 0, 1, "bad_request_id"), ("a/b", 0, 1, "bad_request_id"), ("", 0, 1, "bad_request_id"),
    ("ok", 0, 6, "bad_total_packets"), ("ok", 0, 0, "bad_total_packets"),
    ("ok", 3, 3, "bad_packet_index"), ("ok", -1, 3, "bad_packet_index"),
])
def test_request_validation(rid, idx, total, code):
    with pytest.raises(aw.AirWritingError) as e:
        send(rid, idx, total)
    assert e.value.code == code


def test_total_packets_must_stay_the_same(buffers):
    send("t", 0, 2, "L", fake_ocr("x"))
    with pytest.raises(aw.AirWritingError) as e:
        send("t", 1, 3, "L", fake_ocr("x"))
    assert e.value.code == "total_packets_mismatch"


def test_ocr_failure_keeps_images_for_a_retry(buffers):
    def boom(image):
        raise RuntimeError("tesseract missing")
    send("f", 0, 2, "L", boom)
    with pytest.raises(RuntimeError):
        send("f", 1, 2, "T", boom)
    assert send("f", 1, 2, "T", fake_ocr("LT")) == "LT"


def test_old_buffers_are_removed(buffers, monkeypatch):
    send("old", 0, 2, "L", fake_ocr("x"))
    os.utime(buffers / "old", (0, 0))
    send("new", 0, 2, "L", fake_ocr("x"))
    assert not (buffers / "old").exists() and (buffers / "new").exists()


def test_ocr_text_is_json_safe():
    assert aw.run_ocr(np.full((60, 180), 255, np.uint8), lambda a: 'HE"L\\LO\n\x0c') == "HELLO"
