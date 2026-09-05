"""

Draws all HUD elements onto a frame:
  - Driver state  (bottom-right)
  - Heart rate    (bottom-left)
  - Calibration banner (centre-top)
  - "Face not detected" warning


"""

import cv2
import numpy as np

# ── Colour palette (BGR)
COLOUR_FOCUSED      = (60,  200,  60)   # green
COLOUR_DISTRACTED   = (30,  165, 255)   # orange
COLOUR_SLEEP        = (50,   50, 220)   # red
COLOUR_CALIBRATING  = (220, 200,  50)   # cyan-ish
COLOUR_NO_FACE      = (50,   50, 220)   # red
COLOUR_BPM          = (200, 200, 200)   # light grey
COLOUR_SHADOW       = (20,   20,  20)   # near-black for text shadow

# Map each driver state string to its display colour
STATE_COLOURS = {
    "Focused on the road": COLOUR_FOCUSED,
    "Distracted (long)":   COLOUR_DISTRACTED,
    "Distracted (short)":  COLOUR_DISTRACTED,
    "Microsleep":          COLOUR_SLEEP,
    "Sleep":               COLOUR_SLEEP,
    "Calibrating...":      COLOUR_CALIBRATING,
    "Face not detected":   COLOUR_NO_FACE,
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75
THICKNESS  = 2
MARGIN     = 16   # pixels from frame edge


def _put_text_shadowed(frame: np.ndarray,
                       text: str,
                       origin: tuple[int, int],
                       colour: tuple[int, int, int],
                       scale: float = FONT_SCALE,
                       thickness: int = THICKNESS) -> None:
    x, y = origin
    # Shadow (offset by 2 px)
    cv2.putText(frame, text, (x + 2, y + 2),
                FONT, scale, COLOUR_SHADOW, thickness + 1, cv2.LINE_AA)
    # Foreground
    cv2.putText(frame, text, (x, y),
                FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_state(frame: np.ndarray, state: str) -> None:
   
    
    h, w = frame.shape[:2]
    colour = STATE_COLOURS.get(state, COLOUR_FOCUSED)
    label  = f"State: {state}"

    # Measure text size so we can right-align it
    (text_w, text_h), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, THICKNESS)

    x = w - text_w - MARGIN
    y = h - MARGIN - baseline

    _put_text_shadowed(frame, label, (x, y), colour)


def draw_bpm(frame: np.ndarray, bpm: float | None) -> None:
    h = frame.shape[0]

    if bpm is None:
        label = "Heart rate: estimating..."
    else:
        label = f"Heart rate: {bpm:.0f} BPM"

    x = MARGIN
    _, text_h = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)[:2]
    y = h - MARGIN - text_h // 2

    _put_text_shadowed(frame, label, (x, y), COLOUR_BPM)


def draw_calibration_banner(frame: np.ndarray, seconds_left: float) -> None:
   
    h, w = frame.shape[:2]
    line1 = "Calibrating... look forward"
    line2 = f"({seconds_left:.1f}s remaining)"

    for i, line in enumerate([line1, line2]):
        scale = 0.80 if i == 0 else 0.60
        (tw, th), _ = cv2.getTextSize(line, FONT, scale, THICKNESS)
        x = (w - tw) // 2
        y = 40 + i * 36
        _put_text_shadowed(frame, line, (x, y), COLOUR_CALIBRATING, scale)


def draw_no_face(frame: np.ndarray) -> None:
   
    h, w = frame.shape[:2]
    label = "Face not detected"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.9, THICKNESS + 1)
    x = (w - tw) // 2
    y = h // 2
    _put_text_shadowed(frame, label, (x, y), COLOUR_NO_FACE, 0.9, THICKNESS + 1)


def draw_overlay(frame: np.ndarray,
                 state: str,
                 bpm: float | None,
                 calibrating: bool,
                 calibration_seconds_left: float,
                 face_detected: bool) -> None:
    """
    frame                    : BGR frame (modified in-place).
    state                    : Current driver state string.
    bpm                      : BPM value or None.
    calibrating              : True while in the initial calibration phase.
    calibration_seconds_left : Countdown shown during calibration.
    face_detected            : False → show "Face not detected" warning.
    """
    if not face_detected:
        draw_no_face(frame)
        # Still show BPM placeholder and a "no face" state label
        draw_bpm(frame, None)
        draw_state(frame, "Face not detected")
        return

    if calibrating:
        draw_calibration_banner(frame, calibration_seconds_left)
        # During calibration no state/BPM is meaningful yet
        draw_bpm(frame, None)
        draw_state(frame, "Calibrating...")
        return

    # Normal operation
    draw_state(frame, state)
    draw_bpm(frame, bpm)
