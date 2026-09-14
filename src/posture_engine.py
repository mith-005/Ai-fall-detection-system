import math
import cv2
import numpy as np
import mediapipe as mp
from src.config import Config


class FramePreprocessor:
    """
    Module 2: Frame Preprocessing & Mathematical Tensor Transformations
    """
    @staticmethod
    def normalize_brightness_clahe(frame):
        """
        1. Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Normalizes uneven lighting and shadows on the Luminance (Y) channel.
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(ycrcb))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels[0] = clahe.apply(channels[0])
        equalized_ycrcb = cv2.merge(channels)
        return cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def letterbox_resize(frame, target_size=(640, 640), fill_color=(114, 114, 114)):
        """
        2. Aspect-Ratio Preserving Letterbox Resizing (YOLO Geometry).
        """
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded_frame = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=fill_color
        )
        return padded_frame, scale, (left, top)

    @staticmethod
    def normalize_pixel_values(frame):
        """
        3. Min-Max Dynamic Range Value Normalization [0.0, 1.0].
        """
        return frame.astype(np.float32) / 255.0


class PostureEngine:
    def __init__(self):
        """Initializes MediaPipe Pose Landmark Detection Solution & Preprocessor."""
        self.preprocessor = FramePreprocessor()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_angle(self, p1, p2):
        """Calculates vertical angle (inclination) between two landmark points relative to ground."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        return angle

    def process_frame(self, frame):
        """
        Analyzes frame for human pose keypoints with explicit Module 2 preprocessing.
        Returns: is_fall_posture (bool), annotated_frame, metadata (dict)
        """
        # =========================================================================
        # MODULE 2: EXPLICIT PREPROCESSING
        # =========================================================================
        # 1. Brightness / Contrast Normalization via CLAHE (Actively applied)
        enhanced_frame = self.preprocessor.normalize_brightness_clahe(frame)

        # 2. Color Space Conversion for MediaPipe (BGR -> RGB)
        h, w, _ = enhanced_frame.shape
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

        # =========================================================================
        # MODULE 4: POSE ESTIMATION (MediaPipe)
        # =========================================================================
        results = self.pose.process(rgb_frame)

        is_fall_posture = False
        metadata = {"aspect_ratio": 0.0, "torso_angle": 0.0, "landmarks_detected": False}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract key landmarks (Normalized 0.0 - 1.0 -> Pixels)
            left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                             int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                              int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            left_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                        int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
            right_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h))

            # Compute anatomical midpoints
            mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
            mid_hip = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

            # =====================================================================
            # MODULE 5: KINEMATIC FALL ANALYSIS
            # =====================================================================
            # Torso angle relative to horizontal plane
            torso_angle = self.calculate_angle(mid_hip, mid_shoulder)

            # Compute bounding box from all visible landmarks
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]
            min_x, max_x = max(0, min(x_coords)), min(w, max(x_coords))
            min_y, max_y = max(0, min(y_coords)), min(h, max(y_coords))

            box_width = max_x - min_x
            box_height = max_y - min_y
            aspect_ratio = box_width / float(box_height) if box_height > 0 else 0.0

            # Kinematic Fall Thresholds:
            # 1. Torso inclined parallel to the ground (< 32°)
            # 2. Or wide horizontal bounding box (AR > 1.35) combined with significant tilt (< 45°)
            if torso_angle < Config.FALL_ANGLE_THRESH:
                is_fall_posture = True
            elif aspect_ratio > Config.FALL_ASPECT_RATIO_THRESH and torso_angle < 45.0:
                is_fall_posture = True

            metadata.update({
                "aspect_ratio": round(aspect_ratio, 2),
                "torso_angle": round(torso_angle, 2),
                "bbox": (min_x, min_y, max_x, max_y),
                "landmarks_detected": True
            })

            # Draw skeleton and status box on the enhanced output frame
            self.mp_drawing.draw_landmarks(
                enhanced_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            cv2.rectangle(
                enhanced_frame, (min_x, min_y), (max_x, max_y), 
                (0, 0, 255) if is_fall_posture else (0, 255, 0), 2
            )

        return is_fall_posture, enhanced_frame, metadata