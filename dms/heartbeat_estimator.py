"""
Remote photoplethysmography (rPPG) heart-rate estimator.
 
Method
------
1. Extract the mean green-channel value from a forehead ROI each frame.
2. Accumulate a rolling signal buffer of HEART_RATE_WINDOW seconds.
3. Once the buffer is full, detrend + bandpass-filter the signal and
   find the dominant frequency via FFT.
4. Convert to BPM and return.
 
Expected BPM range: MIN_BPM – MAX_BPM (45 – 180).
"""
 
import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
 

FOREHEAD_Y_OFFSET = 0.30   # fraction of face height above nose to go up
FOREHEAD_H_FRAC   = 0.15   # fraction of face height to use as ROI height
FOREHEAD_W_FRAC   = 0.40   # fraction of face width  to use as ROI width
 
 
class HeartbeatEstimator:
    """
    estimator = HeartbeatEstimator(window=12.0, min_bpm=45, max_bpm=180)
 
    each frame:
        estimator.update(bgr_frame, face_data)
 
    when needed:
        bpm = estimator.bpm   # None until buffer is full
    """
 
    def __init__(self,
                 window:   float = 12.0,
                 min_bpm:  int   = 45,
                 max_bpm:  int   = 180):
        self.window  = window
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
 
        # Buffer stores (timestamp, green_mean) pairs
        self._buffer: deque[tuple[float, float]] = deque()
        self._bpm:    float | None = None
 
 
    @property
    def bpm(self) -> float | None:
        return self._bpm
 
    @property
    def ready(self) -> bool:
        if len(self._buffer) < 2:
            return False
        return self._buffer[-1][0] - self._buffer[0][0] >= self.window * 0.95
 
    def update(self, bgr_frame: np.ndarray, face_data: dict | None) -> None:
        """
        Extract one green-channel sample from the forehead ROI and,
        if the buffer is full, recompute the BPM estimate.
        """
        if face_data is None:
            return   # no face → skip this frame
 
        roi = self._extract_forehead_roi(bgr_frame, face_data)
        if roi is None or roi.size == 0:
            return
 
        # Green channel mean (index 1 in BGR)
        green_mean = float(np.mean(roi[:, :, 1]))
        self._buffer.append((time.time(), green_mean))
 
        # Prune samples older than the window
        self._prune()
 
        # Recompute BPM once we have enough data
        if self.ready:
            self._bpm = self._estimate_bpm()
 
    def draw_roi(self, bgr_frame: np.ndarray, face_data: dict) -> None:
        """Draw the forehead ROI rectangle on the frame"""
        import cv2
        rect = self._forehead_rect(bgr_frame, face_data)
        if rect:
            x, y, w, h = rect
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
 
    
 
    def _prune(self) -> None:
        if not self._buffer:
            return
        cutoff = self._buffer[-1][0] - self.window * 1.10
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()
 
    def _extract_forehead_roi(self,
                               bgr_frame: np.ndarray,
                               face_data: dict) -> np.ndarray | None:
        rect = self._forehead_rect(bgr_frame, face_data)
        if rect is None:
            return None
        x, y, w, h = rect
        return bgr_frame[y:y + h, x:x + w]
 
    def _forehead_rect(self,
                       bgr_frame: np.ndarray,
                       face_data: dict) -> tuple[int, int, int, int] | None:
        """
        Compute the forehead ROI as (x, y, w, h) in pixel coordinates.
 
        use the nose tip and the outer eye corners to estimate
        face centre, width, and where the forehead sits.
        """
        fh, fw = bgr_frame.shape[:2]
 
        nose_x, nose_y   = face_data["nose"]
        face_width       = face_data["face_width_px"]
 
        # Approximate face height from nose to top: use eye Y as reference
        left_eye_top  = min(p[1] for p in face_data["left_eye"])
        right_eye_top = min(p[1] for p in face_data["right_eye"])
        eye_y         = (left_eye_top + right_eye_top) // 2
 
        # Forehead centre: above the eye line by a fixed fraction
        face_height_est = abs(nose_y - eye_y) * 3   # rough estimate
        roi_h = max(int(face_height_est * FOREHEAD_H_FRAC), 10)
        roi_w = max(int(face_width      * FOREHEAD_W_FRAC), 10)
 
        # Centre the ROI horizontally on the nose X
        cx = nose_x
        cy = eye_y - int(face_height_est * 0.10)   # just above eye line
 
        x = cx - roi_w // 2
        y = cy - roi_h
 
        # Clamp to frame bounds
        x = max(0, min(x, fw - roi_w))
        y = max(0, min(y, fh - roi_h))
 
        if roi_w <= 0 or roi_h <= 0:
            return None
        return (x, y, roi_w, roi_h)
 
    def _estimate_bpm(self) -> float | None:
        """
        Run FFT on the buffered green-channel signal and return the
        dominant frequency in BPM, or None if estimation fails.
        """
        timestamps = np.array([t for t, _ in self._buffer])
        signal     = np.array([g for _, g in self._buffer])
 
        if len(signal) < 10:
            return None
 
        # ── Compute average sample rate 
        fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0] + 1e-6)
        if fps < 1:
            return None
 
        # ── Detrend (remove slow drift) 
        signal = signal - np.mean(signal)
 
        # ── Bandpass filter: 0.75 Hz – 3.0 Hz  (45 – 180 BPM)
        low  = self.min_bpm / 60.0   # Hz
        high = self.max_bpm / 60.0   # Hz
        nyq  = fps / 2.0
 
        # Guard: filter only makes sense if cutoffs are inside Nyquist
        if high >= nyq:
            high = nyq * 0.95
        if low >= high:
            return None
 
        try:
            b, a   = butter(4, [low / nyq, high / nyq], btype="band")
            filtered = filtfilt(b, a, signal)
        except Exception:
            return None
 
        # ── FFT
        n      = len(filtered)
        fft    = np.abs(np.fft.rfft(filtered * np.hanning(n)))
        freqs  = np.fft.rfftfreq(n, d=1.0 / fps)
 
        # Restrict to the valid BPM band
        mask   = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return None
 
        peak_freq = freqs[mask][np.argmax(fft[mask])]
        bpm       = peak_freq * 60.0
 
        # Sanity check
        if not (self.min_bpm <= bpm <= self.max_bpm):
            return None
 
        return round(bpm, 1)
 