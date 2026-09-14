import cv2
import time
import datetime
from src.config import Config
from src.yolo_detector import YOLODetector
from src.posture_engine import PostureEngine
from src.temporal_filter import TemporalFilter
from src.db_handler import DatabaseHandler
from src.alert_dispatcher import AlertDispatcher

def initialize_camera():
    """Attempts multiple backends and indices to establish a working camera stream."""
    backends = [
        ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
        ("Default (CAP_ANY)", cv2.CAP_ANY)
    ]
    
    for name, backend in backends:
        for index in [0, 1, 2]:
            print(f"[INFO] Testing Camera Index {index} with {name}...")
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Allow hardware sensor warm-up time
                time.sleep(1.0)
                
                # Attempt to read up to 10 test frames
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[SUCCESS] Connected to Camera Index {index} using {name} ({frame.shape[1]}x{frame.shape[0]})")
                        return cap
                cap.release()
    return None

def main():
    print("\n" + "="*50)
    print("      AI HUMAN FALL DETECTION SYSTEM INITIALIZING      ")
    print("="*50 + "\n")

    # Step 1: Open Camera Feed
    cap = initialize_camera()
    
    # Fallback: Check if user wants to run a sample video file instead
    if cap is None:
        video_file = Config.BASE_DIR / "data" / "sample_fall_video.mp4"
        if video_file.exists():
            print(f"[INFO] Webcam unavailable. Falling back to video file: {video_file}")
            cap = cv2.VideoCapture(str(video_file))
        else:
            print("\n[ERROR] Unable to access webcam or sample video file!")
            print("Troubleshooting steps:")
            print(" 1. Ensure the camera slider switch above your laptop screen is OPEN.")
            print(" 2. Press Fn + F6 / Fn + F9 to toggle webcam power.")
            print(" 3. To test immediately with a video, save any .mp4 to 'data/sample_fall_video.mp4'.")
            input("\nPress Enter to exit...")
            return

    # Step 2: Initialize Core Engines
    print("[INFO] Loading YOLOv8 Person Detector (GPU)...")
    yolo = YOLODetector(model_path="yolov8s.pt", conf_thresh=0.5)

    print("[INFO] Loading MediaPipe Pose Estimator...")
    pose_engine = PostureEngine()

    filter_engine = TemporalFilter(confirmation_seconds=Config.CONFIRMATION_TIME_SEC)
    db = DatabaseHandler()
    dispatcher = AlertDispatcher()

    alert_logged = False
    print("\n[SYSTEM READY] Running detection stream. Press 'q' on the video window to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Blank frame encountered or stream ended.")
            break

        # Pipeline Execution
        person_boxes = yolo.detect_persons(frame)
        is_fall, annotated_frame, meta = pose_engine.process_frame(frame)
        confirmed_fall, elapsed = filter_engine.update(is_fall)

        # UI Overlay Logic
        if confirmed_fall:
            status_text = f"CRITICAL: FALL DETECTED ({elapsed:.1f}s)"
            color = (0, 0, 255)

            if not alert_logged:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = Config.SNAPSHOT_DIR / f"fall_{timestamp_str}.jpg"
                cv2.imwrite(str(snap_path), frame)

                db.log_event(meta["aspect_ratio"], meta["torso_angle"], snap_path)
                print(f"[ALERT DISPATCHED] Event recorded at: {snap_path}")

                alert_msg = f"CRITICAL ALERT: Fall Detected!\nTimestamp: {timestamp_str}\nAspect Ratio: {meta['aspect_ratio']}\nTorso Angle: {meta['torso_angle']} deg"
                dispatcher.send_telegram_alert(alert_msg, image_path=snap_path)
                dispatcher.send_email_alert("URGENT: Fall Event Detected", alert_msg, image_path=snap_path)
                alert_logged = True

        elif is_fall:
            status_text = f"WARNING: Fall Pose ({elapsed:.1f}s)"
            color = (0, 165, 255)
        else:
            status_text = "Status: Normal Monitoring"
            color = (0, 255, 0)
            alert_logged = False

        # Visual Dashboard Overlay
        cv2.putText(annotated_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(annotated_frame, f"Persons: {len(person_boxes)} | Aspect Ratio: {meta['aspect_ratio']} | Torso Angle: {meta['torso_angle']} deg", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("AI Fall Detection System (YOLOv8 + MediaPipe)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting detection loop...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System shut down cleanly.")

if __name__ == "__main__":
    main()