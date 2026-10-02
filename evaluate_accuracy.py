import os
import sys
import csv
import sqlite3
import warnings
from pathlib import Path
import cv2

# 1. Suppress TensorFlow Lite / C++ diagnostic logs and Protobuf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from config import Config
from posture_engine import PostureEngine

UPLOADS_DIR = BASE_DIR / "data" / "uploads"
GT_CSV_PATH = BASE_DIR / "data" / "ground_truth.csv"
OUTPUT_REPORT = BASE_DIR / "data" / "final_accuracy_report.csv"
DB_PATH = BASE_DIR / "data" / "fall_events.db"

def init_db():
    """Initializes the local SQLite database table if not present."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fall_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            aspect_ratio REAL,
            torso_angle REAL,
            snapshot_path TEXT,
            source_video TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event_db(ar, angle, video_name):
    """Commits verified emergency fall incidents to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO fall_events (aspect_ratio, torso_angle, snapshot_path, source_video)
        VALUES (?, ?, ?, ?)
    """, (round(float(ar), 2), round(float(angle), 2), f"snapshots/{video_name}.jpg", video_name))
    conn.commit()
    conn.close()

def run_evaluation():
    if not GT_CSV_PATH.exists():
        print(f"\n[ERROR] Ground truth file not found at: {GT_CSV_PATH}")
        print("Please create 'data/ground_truth.csv' with columns: video_name, actual_label, event_type\n")
        return

    init_db()
    engine = PostureEngine()

    # Load Ground Truth mappings from CSV
    gt_data = {}
    with open(GT_CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("video_name"):
                continue
            gt_data[row["video_name"].strip()] = {
                "label": int(row["actual_label"].strip()),
                "type": row.get("event_type", "Standard Activity").strip()
            }

    if not gt_data:
        print("[ERROR] No valid rows found in data/ground_truth.csv.")
        return

    tp, tn, fp, fn = 0, 0, 0, 0
    total_frames_processed = 0
    results = []

    print("\n" + "=" * 90)
    print("           REAL-TIME HUMAN FALL DETECTION: ACCURACY EVALUATION SUITE")
    print(f"Confirmation Window: {Config.CONFIRMATION_TIME_SEC}s | Angle Thresh: < {Config.FALL_ANGLE_THRESH}° | AR Thresh: > {Config.FALL_ASPECT_RATIO_THRESH}")
    print("=" * 90)

    for video_name, info in gt_data.items():
        vid_path = UPLOADS_DIR / video_name
        if not vid_path.exists():
            print(f"[MISSING FILE] Video not found in data/uploads: {video_name}")
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[CORRUPT FILE] Unable to decode video: {video_name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if (fps <= 0 or fps > 120) else fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        required_consecutive_frames = int(Config.CONFIRMATION_TIME_SEC * fps)

        consecutive_fall = 0
        fall_confirmed = False
        min_angle = 90.0
        max_ar = 0.0
        confirmed_timestamp = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames_processed += 1
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            is_fall_posture, _, meta = engine.process_frame(frame)

            angle = meta.get("torso_angle", 90.0)
            ar = meta.get("aspect_ratio", 0.0)

            if angle < min_angle:
                min_angle = angle
            if ar > max_ar:
                max_ar = ar

            if is_fall_posture:
                consecutive_fall += 1
                if consecutive_fall >= required_consecutive_frames and not fall_confirmed:
                    fall_confirmed = True
                    confirmed_timestamp = round(current_frame_idx / fps, 2)
                    log_event_db(max_ar, min_angle, video_name)
            else:
                consecutive_fall = 0

        cap.release()

        actual = info["label"]
        predicted = 1 if fall_confirmed else 0

        if actual == 1 and predicted == 1:
            tp += 1
            verdict = "TP (Confirmed Fall)"
        elif actual == 0 and predicted == 0:
            tn += 1
            verdict = "TN (Correct Rejection)"
        elif actual == 0 and predicted == 1:
            fp += 1
            verdict = "FP (False Alarm)"
        else:
            fn += 1
            verdict = "FN (Missed Fall)"

        print(f"File: {video_name:<28} | GT: {actual} | Pred: {predicted} | Min θ: {min_angle:5.1f}° | Max AR: {max_ar:4.2f} | {verdict}")

        results.append({
            "Video File": video_name,
            "Event Type": info["type"],
            "Total Frames": total_frames,
            "Min Torso Angle (deg)": round(min_angle, 2),
            "Max Aspect Ratio": round(max_ar, 2),
            "Confirmed Time (s)": confirmed_timestamp if confirmed_timestamp else "None",
            "Ground Truth": actual,
            "Predicted": predicted,
            "Outcome": verdict
        })

    # Metric calculations
    total_videos = tp + tn + fp + fn
    accuracy = ((tp + tn) / total_videos) * 100 if total_videos > 0 else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
    f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 55)
    print("         SYSTEM PERFORMANCE EVALUATION MATRIX")
    print("=" * 55)
    print(f"Total Videos Evaluated : {total_videos}")
    print(f"Total Frames Processed : {total_frames_processed}")
    print(f"True Positives (TP)    : {tp}")
    print(f"True Negatives (TN)    : {tn}")
    print(f"False Positives (FP)   : {fp}")
    print(f"False Negatives (FN)   : {fn}")
    print("-" * 55)
    print(f"Accuracy               : {accuracy:.2f}%")
    print(f"Sensitivity / Recall   : {recall:.2f}%")
    print(f"Specificity            : {specificity:.2f}%")
    print(f"Precision              : {precision:.2f}%")
    print(f"F1-Score               : {f1:.2f}%")
    print("=" * 55)

    # Export full audit table and summary into master CSV
    with open(OUTPUT_REPORT, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Video File", "Event Type", "Total Frames", "Min Torso Angle (deg)", 
            "Max Aspect Ratio", "Confirmed Time (s)", "Ground Truth", "Predicted", "Outcome"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

        f.write("\n--- SUMMARY METRICS ---\n")
        f.write(f"Total Videos Evaluated,{total_videos}\n")
        f.write(f"True Positives (TP),{tp}\n")
        f.write(f"True Negatives (TN),{tn}\n")
        f.write(f"False Positives (FP),{fp}\n")
        f.write(f"False Negatives (FN),{fn}\n")
        f.write(f"Accuracy (%),{accuracy:.2f}\n")
        f.write(f"Precision (%),{precision:.2f}\n")
        f.write(f"Sensitivity / Recall (%),{recall:.2f}\n")
        f.write(f"Specificity (%),{specificity:.2f}\n")
        f.write(f"F1-Score (%),{f1:.2f}\n")

    print(f"\n[OUTPUT] Master Evaluation Report saved to: {OUTPUT_REPORT}")
    print(f"[OUTPUT] Audit entries stored in:         {DB_PATH}\n")

if __name__ == "__main__":
    run_evaluation()