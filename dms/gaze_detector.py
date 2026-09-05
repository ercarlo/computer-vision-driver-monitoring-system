"""
Estimates whether the driver is looking forward or away using the
horizontal position of the nose tip relative to a calibrated reference.
 

During calibration (first N seconds) the driver looks straight ahead.
We record the mean normalised X position of the nose tip as the
"forward" reference. 
"""
 
import time
from collections import deque
 
 
class GazeDetector:
 
    def __init__(self,
                 nose_threshold: float = 0.08,
                 calibration_time: float = 3.0):
        self.nose_threshold   = nose_threshold
        self.calibration_time = calibration_time
 
        # Internal state
        self._calibrating      = True
        self._cal_samples: list[float] = []   # nose X values during calibration
        self._reference_x: float | None = None  # mean nose X when looking forward
        self._start_time: float | None  = None  # set on first update() call
 
 
    @property
    def is_calibrating(self) -> bool:
        return self._calibrating
 
    @property
    def calibration_seconds_left(self) -> float:
        if self._start_time is None:
            return self.calibration_time
        elapsed = time.time() - self._start_time
        return max(0.0, self.calibration_time - elapsed)
 
    @property
    def reference_x(self) -> float | None:
        return self._reference_x
 
    def update(self, nose_norm_x: float | None) -> str:
        """
        Returns
        str — one of:
            "calibrating"  : still collecting the forward reference
            "no_face"      : nose_norm_x is None (no face detected)
            "forward"      : driver is looking at the road
            "away"         : driver is looking away
        """
        now = time.time()
 
        # Initialise timer on first call
        if self._start_time is None:
            self._start_time = now
 
        # If face is not visible, report no_face but don't advance state machine
        if nose_norm_x is None:
            return "no_face"
 
        #  Calibration phase 
        if self._calibrating:
            self._cal_samples.append(nose_norm_x)
            elapsed = now - self._start_time
 
            if elapsed >= self.calibration_time:
                # Compute mean of collected samples as the forward reference
                self._reference_x = sum(self._cal_samples) / len(self._cal_samples)
                self._calibrating  = False
 
 
            return "calibrating"
 
        # Normal operation 
        deviation = abs(nose_norm_x - self._reference_x)
 
        if deviation > self.nose_threshold:
            return "away"
        return "forward"
 
    def reset_calibration(self):
        """Restart the calibration phase (useful for testing)."""
        self._calibrating  = True
        self._cal_samples  = []
        self._reference_x  = None
        self._start_time   = None
 