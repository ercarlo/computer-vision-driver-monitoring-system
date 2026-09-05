
 
import os
import sys
import cv2
 
from dms.face_tracker         import FaceTracker
from dms.gaze_detector        import GazeDetector
from dms.distraction_logic    import DistractionLogic
from dms.eye_closure_detector import EyeClosureDetector
from dms.heartbeat_estimator  import HeartbeatEstimator
from dms.video_overlay        import draw_overlay
 
# ── Configuration 
 
CALIBRATION_TIME          = 3.0
LONG_DISTRACTION_TIME     = 5.0
SHORT_DISTRACTION_WINDOW  = 30.0
SHORT_DISTRACTION_ACCUM   = 10.0
RECOVERY_FORWARD_TIME     = 2.0
MICROSLEEP_TIME           = 4.0
SLEEP_TIME                = 7.0
EYE_OPEN_RECOVERY_TIME    = 2.0
NOSE_THRESHOLD            = 0.08
EYE_AR_THRESHOLD          = 0.20
HEART_RATE_WINDOW         = 8.0
MIN_BPM                   = 45
MAX_BPM                   = 180
 
OUTPUT_DIR   = "output"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "dms_output.avi")
CAMERA_INDEX = 0
TARGET_FPS   = 30
 
# ── Helpers
 
def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
 
def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {index}.", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap
 
 
def create_video_writer(cap: cv2.VideoCapture, path: str) -> cv2.VideoWriter:
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not writer.isOpened():
        print(f"[WARNING] Could not open VideoWriter for {path}.", file=sys.stderr)
    return writer
 
 
def resolve_state(eye_state: str, distraction_state: str) -> str:

    if eye_state == "sleep":
        return "Sleep"
    if eye_state == "microsleep":
        return "Microsleep"
    return distraction_state
 
 
# ── Main loop ─────────────────────────────────────────────────────────────────
 
def run() -> None:
    ensure_output_dir()
 
    print("[INFO] Opening camera...")
    cap = open_camera(CAMERA_INDEX)
 
    print("[INFO] Initialising MediaPipe Face Mesh...")
    tracker = FaceTracker(max_faces=1)
 
    print("[INFO] Initialising gaze detector...")
    gaze = GazeDetector(
        nose_threshold=NOSE_THRESHOLD,
        calibration_time=CALIBRATION_TIME,
    )
 
    print("[INFO] Initialising distraction logic...")
    distraction = DistractionLogic(
        long_distraction_time=LONG_DISTRACTION_TIME,
        short_distraction_window=SHORT_DISTRACTION_WINDOW,
        short_distraction_accum=SHORT_DISTRACTION_ACCUM,
        recovery_forward_time=RECOVERY_FORWARD_TIME,
    )
 
    print("[INFO] Initialising eye closure detector...")
    eye_detector = EyeClosureDetector(
        ear_threshold=EYE_AR_THRESHOLD,
        microsleep_time=MICROSLEEP_TIME,
        sleep_time=SLEEP_TIME,
        eye_open_recovery=EYE_OPEN_RECOVERY_TIME,
    )
 
    print("[INFO] Initialising heartbeat estimator...")
    heartbeat = HeartbeatEstimator(
        window=HEART_RATE_WINDOW,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM,
    )
 
    print(f"[INFO] Opening video writer -> {OUTPUT_FILE}")
    writer = create_video_writer(cap, OUTPUT_FILE)
 
    driver_state = "Focused on the road"
 
    print("[INFO] Starting Driver Monitoring System. Press 'q' to quit.")
    print(f"[INFO] Calibration: look straight ahead for {CALIBRATION_TIME}s")
 
    while True:
 
        # ── 1. Capture
        ret, bgr_frame = cap.read()
        if not ret or bgr_frame is None:
            continue
        bgr_frame = cv2.flip(bgr_frame, 1)
 
        # ── 2. RGB conversion
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
 
        # ── 3. Face detection
        face_data     = tracker.detect(rgb_frame)
        face_detected = face_data is not None
 
        # ── 4. Gaze estimation
        nose_norm_x = face_data["nose_norm"][0] if face_detected else None
        gaze_state  = gaze.update(nose_norm_x)
 
        calibrating              = gaze.is_calibrating
        calibration_seconds_left = gaze.calibration_seconds_left
 
        # ── 5. Distraction logic
        if not calibrating:
            distraction_state = distraction.update(gaze_state)
        else:
            distraction_state = "Focused on the road"
 
        # ── 6. Eye closure detection 
        if face_detected:
            eye_state = eye_detector.update(
                face_data["left_eye"],
                face_data["right_eye"],
            )
        else:
            eye_state = eye_detector.update(None, None)
 
        # ── 7. Heart-rate estimation
        heartbeat.update(bgr_frame, face_data)
        bpm_estimate = heartbeat.bpm   # None until buffer is full
 
        # ── 8. Priority resolution
        if not calibrating:
            driver_state = resolve_state(eye_state, distraction_state)
 
        # ── 9. Debug landmarks + forehead ROI
        if face_detected:
            tracker.draw_landmarks(bgr_frame, face_data)
            heartbeat.draw_roi(bgr_frame, face_data)
 
        # ── 10. HUD overlay
        draw_overlay(
            frame=bgr_frame,
            state=driver_state,
            bpm=bpm_estimate,
            calibrating=calibrating,
            calibration_seconds_left=calibration_seconds_left,
            face_detected=face_detected,
        )
 
        # ── 11. Save + display 
        if writer.isOpened():
            writer.write(bgr_frame)
        cv2.imshow("Driver Monitoring System", bgr_frame)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] 'q' pressed — shutting down.")
            break
 
    # ── Cleanup 
    cap.release()
    writer.release()
    tracker.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Output video saved to: {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    run()