import sys
import time
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from config import Config
from posture_engine import PostureEngine

def test_video_detection(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0  # Fallback standard FPS
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    confirmation_frames_needed = int(Config.CONFIRMATION_TIME_SEC * fps)

    engine = PostureEngine()
    
    consecutive_fall_frames = 0
    fall_confirmed = False
    frame_idx = 0
    max_aspect_ratio = 0.0
    min_torso_angle = 90.0

    print(f"\nEvaluating: {video_path.name}")
    print(f"FPS: {fps:.1f} | Total Frames: {total_frames} | Required Fall Duration: {Config.CONFIRMATION_TIME_SEC}s ({confirmation_frames_needed} frames)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        is_fall_posture, _, meta = engine.process_frame(frame)
        
        ar = meta.get("aspect_ratio", 0.0)
        angle = meta.get("torso_angle", 90.0)
        
        if ar > max_aspect_ratio:
            max_aspect_ratio = ar
        if angle < min_torso_angle:
            min_torso_angle = angle

        if is_fall_posture:
            consecutive_fall_frames += 1
            if consecutive_fall_frames >= confirmation_frames_needed and not fall_confirmed:
                fall_confirmed = True
                timestamp_in_video = frame_idx / fps
                print(f" -> [FALL CONFIRMED] at Video Timestamp: {timestamp_in_video:.2f}s (Frame {frame_idx}/{total_frames})")
                print(f"    Posture Metrics -> Torso Angle: {angle:.1f}° | Aspect Ratio: {ar:.2f}")
        else:
            consecutive_fall_frames = 0

    cap.release()
    
    print(f"Peak Aspect Ratio: {max_aspect_ratio:.2f} (Threshold: > {Config.FALL_ASPECT_RATIO_THRESH})")
    print(f"Lowest Torso Angle: {min_torso_angle:.1f}° (Threshold: < {Config.FALL_ANGLE_THRESH}°)")
    
    return fall_confirmed

# Run on all files in data/uploads
uploads_dir = BASE_DIR / "data" / "uploads"
video_files = list(uploads_dir.glob("*.mp4")) + list(uploads_dir.glob("*.avi"))

print(f"--- Running Benchmark on {len(video_files)} Uploaded Video(s) ---")
for vid in video_files:
    result = test_video_detection(vid)
    status_text = "FALL DETECTED" if result else "NORMAL / NO FALL"
    print(f"Verdict for {vid.name}: {status_text}\n" + "-"*50)