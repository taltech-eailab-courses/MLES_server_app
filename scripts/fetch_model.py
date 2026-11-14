"""Download the robot-planner GGUF model at container start, if it is not there yet.

Environment variables:
  ROBOT_PLANNER_URL     HTTPS URL of the model (public S3 object or presigned URL).
                        If unset, nothing is downloaded and the app starts without the model.
  ROBOT_PLANNER_GGUF    Where to store it (default /app/models/robot-q4_k_m.gguf).
  ROBOT_PLANNER_SHA256  Optional checksum. If set, an existing file with a different checksum
                        is replaced, and a download that does not match is rejected.

Never fails the container: on any error it prints a warning and the app starts anyway
(the /api/v1/robot_plan endpoint then returns 503). Works with Python 3.8+ and no extra packages.
"""
import hashlib
import os
import sys
import time
import urllib.request

URL = os.environ.get("ROBOT_PLANNER_URL", "").strip()
DEST = os.environ.get("ROBOT_PLANNER_GGUF", "/app/models/robot-q4_k_m.gguf")
SHA256 = os.environ.get("ROBOT_PLANNER_SHA256", "").strip().lower()


def log(msg):
    print(f"[fetch_model] {msg}", flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_url(url):
    return url.split("?", 1)[0]  # never print presigned-URL signatures


def download(url, dest):
    tmp = dest + ".part"
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, next_report = 0, 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done >= next_report:
                log(f"{done / 1e6:.0f} / {total / 1e6:.0f} MB")
                next_report += max(total // 10, 1)
    if total and done != total:
        os.remove(tmp)
        raise IOError(f"incomplete download: {done} of {total} bytes")
    log(f"downloaded {done / 1e6:.1f} MB in {time.time() - t0:.0f} s")
    return tmp


def main():
    if os.path.isfile(DEST):
        if not SHA256:
            log(f"model present: {DEST} (no checksum set, keeping it)")
            return
        if sha256_of(DEST) == SHA256:
            log(f"model present and checksum OK: {DEST}")
            return
        log("model present but checksum differs -> downloading the new version")
    if not URL:
        log(f"ROBOT_PLANNER_URL not set and {DEST} missing -> starting without the robot planner")
        return

    os.makedirs(os.path.dirname(DEST) or ".", exist_ok=True)
    log(f"downloading {safe_url(URL)} -> {DEST}")
    for attempt in range(1, 4):
        try:
            tmp = download(URL, DEST)
            if SHA256:
                got = sha256_of(tmp)
                if got != SHA256:
                    os.remove(tmp)
                    raise IOError(f"checksum mismatch: expected {SHA256}, got {got}")
            os.replace(tmp, DEST)  # atomic: the app never sees a half-written file
            log("model ready")
            return
        except Exception as e:  # network errors, HTTP 403/404, checksum
            log(f"attempt {attempt} failed: {e}")
            time.sleep(3 * attempt)
    log("giving up -> starting without the robot planner")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never block the app from starting
        log(f"unexpected error: {e}")
    sys.exit(0)
