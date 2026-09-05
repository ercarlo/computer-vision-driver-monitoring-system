"""

Detects eye closure using the Eye Aspect Ratio (EAR).


--------------------------------------
p1..p6,p1/p4 are the horizontal corners
and p2/p3/p5/p6 are the vertical pairs):

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

When the eye is open  → EAR ≈ 0.25-0.30
When the eye is closed → EAR ≈ 0.0

    "open"       — eyes are open (or no face)
    "closed"     — both eyes below EAR threshold
    "microsleep" — eyes closed for >= MICROSLEEP_TIME seconds
    "sleep"      — eyes closed for >= SLEEP_TIME seconds
"""

import time
import math


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean distance between two (x, y) pixel points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ear(eye_pts: list[tuple[int, int]]) -> float:
   
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical_1  = _dist(p2, p6)
    vertical_2  = _dist(p3, p5)
    horizontal  = _dist(p1, p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)


class EyeClosureDetector:
   
    def __init__(self,
                 ear_threshold:       float = 0.20,
                 microsleep_time:     float = 4.0,
                 sleep_time:          float = 7.0,
                 eye_open_recovery:   float = 2.0):
      
      
        self.ear_threshold     = ear_threshold
        self.microsleep_time   = microsleep_time
        self.sleep_time        = sleep_time
        self.eye_open_recovery = eye_open_recovery

        # ── Internal timers
        self._closed_since:  float | None = None   # when both eyes closed
        self._open_since:    float | None = None   # when both eyes re-opened
        self._sleep_active:  bool         = False
        self._micro_active:  bool         = False

    

    def update(self,
               left_eye:  list[tuple[int, int]] | None,
               right_eye: list[tuple[int, int]] | None) -> str:
        """
        
        left_eye  : 6 (x, y) pixel points for the left eye, or None.
        right_eye : 6 (x, y) pixel points for the right eye, or None.

        Returns
        -------
        str — "open" | "closed" | "microsleep" | "sleep"
        """
        now = time.time()

        # If landmarks are missing, treat as eyes open (no penalty)
        if left_eye is None or right_eye is None:
            return self._handle_open(now)

        left_ear  = _ear(left_eye)
        right_ear = _ear(right_eye)
        both_closed = (left_ear < self.ear_threshold and
                       right_ear < self.ear_threshold)

        if both_closed:
            return self._handle_closed(now)
        else:
            return self._handle_open(now)

    def current_ear(self,
                    left_eye:  list[tuple[int, int]],
                    right_eye: list[tuple[int, int]]) -> float:
        """Return the mean EAR of both eyes """
        return (_ear(left_eye) + _ear(right_eye)) / 2.0

    #  Internal state machine

    def _handle_closed(self, now: float) -> str:
        # Start closed timer if not already running
        if self._closed_since is None:
            self._closed_since = now
        self._open_since = None   # reset recovery timer

        closed_duration = now - self._closed_since

        # Escalate to sleep / microsleep based on duration
        if closed_duration >= self.sleep_time:
            self._sleep_active = True
            self._micro_active = True   # sleep implies microsleep
        elif closed_duration >= self.microsleep_time:
            self._micro_active = True

        if self._sleep_active:
            return "sleep"
        if self._micro_active:
            return "microsleep"
        return "closed"

    def _handle_open(self, now: float) -> str:
        self._closed_since = None   # reset closed timer

        # If no alert is active, nothing to recover from
        if not self._sleep_active and not self._micro_active:
            self._open_since = None
            return "open"

        # Start (or continue) recovery timer
        if self._open_since is None:
            self._open_since = now

        open_duration = now - self._open_since

        if open_duration >= self.eye_open_recovery:
            # Recovery complete — clear both states
            self._sleep_active = False
            self._micro_active = False
            self._open_since   = None
            return "open"

        # Still recovering — keep showing the active alert
        if self._sleep_active:
            return "sleep"
        return "microsleep"
