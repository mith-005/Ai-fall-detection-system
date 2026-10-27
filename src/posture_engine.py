import math
import cv2
import numpy as np
import mediapipe as mp
from src.config import Config


class FramePreprocessor:
    """
    Module 2: Frame Preprocessing & Mathematical Transformations
    """
    def __init__(self):
        # CLAHE for contrast enhancement under variable lighting
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def normalize_brightness_clahe(self, frame):
        """Applies CLAHE on the Luminance (Y) channel."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


class PostureEngine:
    def __init__(self):
        """Initializes Frame Preprocessor and MediaPipe Pose Solution."""
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
        """Calculates torso inclination angle relative to the horizontal floor plane."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(abs(dy), max(1e-5, abs(dx))))

    def process_frame(self, frame, yolo_box=None):
        """
        Two-stage Cascaded Processing:
          Stage 1 (YOLO): Upstream spatial candidate bounding.
          Stage 2 (MediaPipe): Biomechanical landmark extraction & kinematic angle calculation.
        """
        # 1. Luminance normalization
        enhanced_frame = self.preprocessor.normalize_brightness_clahe(frame)
        h, w = enhanced_frame.shape[:2]

        # 2. Convert to RGB for MediaPipe inference
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        is_fall_posture = False
        metadata = {
            "aspect_ratio": 0.0,
            "torso_angle": 90.0,
            "landmarks_detected": False
        }

        # 3. Biomechanical Landmark Tracking & Centroid Verification
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract pixel bounds of the tracked skeleton
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]

            mp_x1, mp_x2 = max(0, min(x_coords)), min(w, max(x_coords))
            mp_y1, mp_y2 = max(0, min(y_coords)), min(h, max(y_coords))
            person_center_x = (mp_x1 + mp_x2) // 2
            person_center_y = (mp_y1 + mp_y2) // 2

            # Isolate primary torso joints
            l_sh = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            r_sh = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            l_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
            r_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h))

            mid_shoulder = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
            mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

            torso_angle = self.calculate_angle(mid_hip, mid_shoulder)

            # 4. Spatial Alignment: Validate if candidate YOLO box encloses the person
            use_yolo_coords = False
            if yolo_box is not None:
                yx1, yy1, yx2, yy2 = yolo_box[0], yolo_box[1], yolo_box[2], yolo_box[3]
                # Margin tolerance of 50px
                if (yx1 - 50 <= person_center_x <= yx2 + 50) and (yy1 - 50 <= person_center_y <= yy2 + 50):
                    use_yolo_coords = True
                    x1, y1, x2, y2 = yx1, yy1, yx2, yy2
                    conf = yolo_box[4] if len(yolo_box) > 4 else 0.85
                    label = yolo_box[6] if len(yolo_box) > 6 else "Person"

            # If YOLO locked onto background clutter, bind the box tightly to the subject
            if not use_yolo_coords:
                pad_x = int((mp_x2 - mp_x1) * 0.12)
                pad_y = int((mp_y2 - mp_y1) * 0.08)
                x1 = max(0, mp_x1 - pad_x)
                y1 = max(0, mp_y1 - pad_y)
                x2 = min(w, mp_x2 + pad_x)
                y2 = min(h, mp_y2 + pad_y)
                conf = 0.90
                label = "Person"

            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)
            aspect_ratio = box_width / float(box_height)

            # 5. Dual-Geometry Fall Decision Rules
            # Rule 1: Torso collapsed to ground (< 32°) AND wide aspect ratio (AR > 1.35)
            if torso_angle < Config.FALL_ANGLE_THRESH and aspect_ratio > Config.FALL_ASPECT_RATIO_THRESH:
                is_fall_posture = True
            # Rule 2: Semi-recumbent diagonal fall check
            elif aspect_ratio > Config.FALL_ASPECT_RATIO_THRESH and torso_angle < 45.0:
                is_fall_posture = True

            metadata.update({
                "aspect_ratio": round(aspect_ratio, 2),
                "torso_angle": round(torso_angle, 2),
                "bbox": (x1, y1, x2, y2),
                "landmarks_detected": True
            })

            # 6. Render Overlays
            # Draw MediaPipe skeleton
            self.mp_drawing.draw_landmarks(
                enhanced_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            # Draw YOLO spatial bounding envelope (Red = Fall, Green = Normal)
            box_color = (0, 0, 255) if is_fall_posture else (0, 255, 0)
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), box_color, 2)

            # High-visibility badge
            tag_text = f"YOLO: {label} {conf:.2f} | AR: {aspect_ratio:.2f}"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tag_y1 = max(0, y1 - th - 10)
            cv2.rectangle(enhanced_frame, (x1, tag_y1), (x1 + tw + 10, y1), box_color, -1)
            cv2.putText(
                enhanced_frame,
                tag_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return is_fall_posture, enhanced_frame, metadata