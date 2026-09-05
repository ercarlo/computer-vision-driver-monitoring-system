"""

Implements the two owl-distraction states from the assignment:
 
  • Distracted (long)  — driver looks away for ≥ LONG_DISTRACTION_TIME seconds
                         continuously.
 
  • Distracted (short) — accumulated away-time inside a rolling
                         SHORT_DISTRACTION_WINDOW reaches
                         SHORT_DISTRACTION_ACCUM seconds.
 
Both states clear only when the driver looks forward for at least
RECOVERY_FORWARD_TIME seconds continuously.
 
"""
 
import time
from collections import deque
 
 
class DistractionLogic:
    """
    State machine for long and short owl distraction detection.
    """
 
    def __init__(self,
                 long_distraction_time:    float = 5.0,
                 short_distraction_window: float = 30.0,
                 short_distraction_accum:  float = 10.0,
                 recovery_forward_time:    float = 2.0):
        
        self.long_distraction_time    = long_distraction_time
        self.short_distraction_window = short_distraction_window
        self.short_distraction_accum  = short_distraction_accum
        self.recovery_forward_time    = recovery_forward_time
 
        #  Long distraction timers 
        self._away_since:    float | None = None   # timestamp when away started
        self._long_active:   bool         = False   # Distracted(long) is firing
 
        #   Short distraction — rolling window     
        
        self._window: deque[tuple[float, bool]] = deque()
        self._short_active: bool = False
 
        #   Recovery timer (shared by both states)   
        self._forward_since: float | None = None   # timestamp when fwd started
 
          
 
    def update(self, gaze_state: str) -> str:
        """
        gaze_state : "forward", "away", or "no_face".
                     "no_face" is treated conservatively as "forward"
                     (we don't penalise for losing tracking momentarily).
 
        Returns
        -------
        str — "Focused on the road", "Distracted (long)", or "Distracted (short)"
        """
        now      = time.time()
        is_away  = (gaze_state == "away")
        is_fwd   = not is_away   # no_face counts as forward here
 
        #   1. Record entry in the rolling window   
        self._window.append((now, is_away))
        self._prune_window(now)
 
        #   2. Long distraction timer         

        if is_away:
            if self._away_since is None:
                self._away_since = now          # start counting
            away_duration = now - self._away_since
            if away_duration >= self.long_distraction_time:
                self._long_active = True
        else:
            self._away_since = None             # reset on any forward frame
 
        #   3. Short distraction — accumulated time in window         

        accum_away = self._accumulated_away_time(now)
        if accum_away >= self.short_distraction_accum:
            self._short_active = True
 
        #   4. Recovery — clear active states after looking forward long enough
        if is_fwd:
            if self._forward_since is None:
                self._forward_since = now
            fwd_duration = now - self._forward_since
 
            # Act only once, right when the threshold is crossed
            if fwd_duration >= self.recovery_forward_time:
                if self._long_active or self._short_active:
                    self._long_active  = False
                    self._short_active = False
                # Keep _forward_since frozen so this block does not re-trigger
        else:
            # Driver looked away — reset the recovery timer
            self._forward_since = None
 
        #   5. Return current state (long has priority over short)       
        if self._long_active:
            return "Distracted (long)"
        if self._short_active:
            return "Distracted (short)"
        return "Focused on the road"
 
      
 
    def continuous_away_seconds(self) -> float:
        """Seconds the driver has been continuously looking away (0 if forward)."""
        if self._away_since is None:
            return 0.0
        return time.time() - self._away_since
 
    def accumulated_away_in_window(self) -> float:
        """Total away-seconds inside the current rolling window."""
        return self._accumulated_away_time(time.time())
 
    #   Internal helpers    
 
    def _prune_window(self, now: float) -> None:
        """Remove entries older than SHORT_DISTRACTION_WINDOW seconds."""
        cutoff = now - self.short_distraction_window
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
 
    def _accumulated_away_time(self, now: float) -> float:
        """
        Approximate the total time spent 'away' inside the rolling window.
 
        """
        entries = list(self._window)
        if len(entries) < 2:
            return 0.0
 
        total = 0.0
        for i in range(1, len(entries)):
            t_prev, away_prev = entries[i - 1]
            t_curr, _         = entries[i]
            if away_prev:
                total += t_curr - t_prev
 
        # Also count from last sample to now if still away
        if entries[-1][1]:
            total += now - entries[-1][0]
 
        return total