
 
import os
import numpy as np
import cv2
 
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
 
# ── Landmark indices (identical to classic 478-point Face Mesh) 
NOSE_TIP_IDX   = 4
LEFT_EYE_IDXS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX  = 468
RIGHT_IRIS_IDX = 473
 
 
class FaceTracker:
 
    
    DEFAULT_MODEL = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "face_landmarker.task",
    )
 
    def __init__(self, max_faces: int = 1, model_path: str | None = None):
      
        model_path = model_path or self.DEFAULT_MODEL
 
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"\n[ERROR] Model file not found: {model_path}\n\n"
                "Run this command in your project root to download it:\n\n"
                "  python download_model.py\n"
            )
 
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
 
        options = FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            running_mode=RunningMode.IMAGE,   # synchronous per-frame mode
        )
 
        self._landmarker = FaceLandmarker.create_from_options(options)
 
   
 
    def detect(self, rgb_frame: np.ndarray) -> dict | None:
        """
        Returns
        dict with keys:
            landmarks_norm  – full list of NormalizedLandmark (478 points)
            nose            – (x, y) pixels
            nose_norm       – (x, y) normalised [0-1]
            left_eye        – list of (x, y) pixels (EAR landmarks)
            right_eye       – list of (x, y) pixels (EAR landmarks)
            left_iris       – (x, y) pixels
            right_iris      – (x, y) pixels
            face_width_px   – approx. face width in pixels
        None  if no face is detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result   = self._landmarker.detect(mp_image)
 
        if not result.face_landmarks:
            return None
 
        lm   = result.face_landmarks[0]   # list of NormalizedLandmark
        h, w = rgb_frame.shape[:2]
 
        nose      = self._to_xy(lm[NOSE_TIP_IDX], w, h)
        nose_norm = (lm[NOSE_TIP_IDX].x, lm[NOSE_TIP_IDX].y)
 
        left_eye  = [self._to_xy(lm[i], w, h) for i in LEFT_EYE_IDXS]
        right_eye = [self._to_xy(lm[i], w, h) for i in RIGHT_EYE_IDXS]
 
        left_iris  = self._to_xy(lm[LEFT_IRIS_IDX],  w, h) if len(lm) > LEFT_IRIS_IDX  else (0, 0)
        right_iris = self._to_xy(lm[RIGHT_IRIS_IDX], w, h) if len(lm) > RIGHT_IRIS_IDX else (0, 0)
 
        left_outer    = self._to_xy(lm[LEFT_EYE_IDXS[0]],  w, h)
        right_outer   = self._to_xy(lm[RIGHT_EYE_IDXS[0]], w, h)
        face_width_px = max(abs(right_outer[0] - left_outer[0]), 1)
 
        return {
            "landmarks_norm": lm,
            "nose":           nose,
            "nose_norm":      nose_norm,
            "left_eye":       left_eye,
            "right_eye":      right_eye,
            "left_iris":      left_iris,
            "right_iris":     right_iris,
            "face_width_px":  face_width_px,
        }
 
    def draw_landmarks(self, bgr_frame: np.ndarray, face_data: dict) -> np.ndarray:
        """Draw a minimal debug overlay (nose tip, eye corners, iris centres)."""
        cv2.circle(bgr_frame, face_data["nose"], 4, (0, 255, 255), -1)
        for pt in face_data["left_eye"] + face_data["right_eye"]:
            cv2.circle(bgr_frame, pt, 2, (255, 200, 0), -1)
        if face_data["left_iris"]  != (0, 0):
            cv2.circle(bgr_frame, face_data["left_iris"],  3, (0, 200, 255), -1)
        if face_data["right_iris"] != (0, 0):
            cv2.circle(bgr_frame, face_data["right_iris"], 3, (0, 200, 255), -1)
        return bgr_frame
 
    def release(self):
        """Free MediaPipe resources."""
        self._landmarker.close()
 
    
 
    @staticmethod
    def _to_xy(landmark, w: int, h: int) -> tuple[int, int]:
        return (int(landmark.x * w), int(landmark.y * h))