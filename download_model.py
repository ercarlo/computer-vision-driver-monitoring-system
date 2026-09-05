

import urllib.request
import os
import sys

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILE = "face_landmarker.task"


def download():
    if os.path.isfile(MODEL_FILE):
        print(f"[INFO] Model already present: {MODEL_FILE}  (skipping download)")
        return

    print(f"[INFO] Downloading MediaPipe face landmarker model…")
    print(f"       URL : {MODEL_URL}")
    print(f"       Dest: {os.path.abspath(MODEL_FILE)}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "#" * int(pct / 2)
            sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE, _progress)
        print(f"\n[OK] Saved to {MODEL_FILE}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    download()
