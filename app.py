import os
import sys
import time
import sqlite3
import cv2
from pathlib import Path
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# 1. Resolve Absolute Project Directories
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

# Load environment configuration
load_dotenv(dotenv_path=BASE_DIR / ".env")

# 2. Resilient Template and Static Path Discovery
candidate_template_dirs = [
    BASE_DIR / "templates",
    BASE_DIR / "template",
    BASE_DIR / "src" / "templates",
    BASE_DIR
]

resolved_template_dir = None
for folder in candidate_template_dirs:
    if (folder / "index.html").is_file():
        resolved_template_dir = folder
        break

if not resolved_template_dir:
    resolved_template_dir = BASE_DIR / "templates"
    print(f"\n⚠️ [WARNING] 'index.html' not found yet. Defaulting to: {resolved_template_dir}")
else:
    print(f"✅ [UI INIT] Found templates at: {resolved_template_dir}")

resolved_static_dir = BASE_DIR / "static"
if not resolved_static_dir.exists():
    resolved_static_dir = BASE_DIR / "src" / "static"

app = Flask(
    __name__,
    template_folder=str(resolved_template_dir),
    static_folder=str(resolved_static_dir)
)

# 3. Import Project Modules
from config import Config
from posture_engine import PostureEngine
from temporal_filter import TemporalFilter
from alert_dispatcher import AlertDispatcher
from db_handler import DatabaseHandler

# Safety & Upload Constraints
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300 MB limit
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
Config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)

# Global Video Feed State: 0 for Webcam, string path for uploaded video
CURRENT_VIDEO_SOURCE = 0

# Initialize Subsystems
ai_engine = PostureEngine()
dispatcher = AlertDispatcher()
db_handler = DatabaseHandler()


def allowed_file(filename):
    """Validates permitted video container formats."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_video_frames(source):
    """Encodes and yields multipart MJPEG stream from active source."""
    is_webcam = isinstance(source, int) or str(source).isdigit()

    if is_webcam:
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(str(source))

    temporal_filter = TemporalFilter(confirmation_seconds=Config.CONFIRMATION_TIME_SEC)
    alert_already_dispatched = False

    while True:
        if not cap.isOpened():
            break

        success, frame = cap.read()

        # Handle stream completion / loop video files
        if not success or frame is None:
            if not is_webcam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # 1. Biomechanical Posture & Kinematics Processing
        try:
            is_fall_posture, processed_frame, metadata = ai_engine.process_frame(frame)
        except Exception as e:
            print(f"[ENGINE ERROR] Frame processing exception: {e}")
            processed_frame = frame
            is_fall_posture = False
            metadata = {}

        # 2. Temporal Confirmation (3.0s Threshold)
        is_confirmed_fall, elapsed = temporal_filter.update(is_fall_posture)

        # 3. Alert Dispatch & SQLite Audit Logging
        if is_confirmed_fall:
            if not alert_already_dispatched:
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                snapshot_filename = f"FALL_{int(time.time())}.jpg"
                snapshot_path = str(Config.SNAPSHOT_DIR / snapshot_filename)

                # Persist evidence snapshot
                cv2.imwrite(snapshot_path, processed_frame)

                # Log to SQLite database
                db_handler.log_event(
                    aspect_ratio=metadata.get("aspect_ratio", 0.0),
                    torso_angle=metadata.get("torso_angle", 0.0),
                    snapshot_path=snapshot_path
                )

                # Dispatch cascading notifications
                dispatcher.dispatch_fallback_sequence(
                    snapshot_path=snapshot_path,
                    timestamp=timestamp_str
                )

                alert_already_dispatched = True
        else:
            if not is_fall_posture:
                alert_already_dispatched = False

        # Encode frame buffer to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Multipart MJPEG stream boundary
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """Renders the dashboard UI."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG streaming endpoint for client browser display."""
    global CURRENT_VIDEO_SOURCE
    return Response(
        generate_video_frames(CURRENT_VIDEO_SOURCE),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handles file uploads and re-routes active feed."""
    global CURRENT_VIDEO_SOURCE
    try:
        if 'video_file' not in request.files:
            return redirect(url_for('index'))

        file = request.files['video_file']
        if file.filename == '':
            return redirect(url_for('index'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = str(UPLOAD_DIR / filename)
            file.save(save_path)
            print(f"✅ [UPLOAD SUCCESS] Active video changed to: {save_path}")

            CURRENT_VIDEO_SOURCE = save_path
            return redirect(url_for('index'))
        else:
            print(f"❌ [UPLOAD ERROR] Unsupported file extension: {file.filename}")
            return redirect(url_for('index'))

    except Exception as e:
        print(f"❌ [UPLOAD FAILED] Exception: {e}")
        return redirect(url_for('index'))


@app.route('/reset_webcam')
def reset_webcam():
    """Reverts input feed to default hardware webcam (Device 0)."""
    global CURRENT_VIDEO_SOURCE
    CURRENT_VIDEO_SOURCE = 0
    return redirect(url_for('index'))


@app.route('/api/events')
def get_events():
    """Returns recent fall records for live frontend polling."""
    try:
        db_path = str(Config.DB_PATH)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ensure table schema exists to eliminate query crashes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                aspect_ratio REAL,
                torso_angle REAL,
                snapshot_path TEXT
            )
        """)
        conn.commit()

        cursor.execute("SELECT id, timestamp, aspect_ratio, torso_angle FROM fall_events ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        conn.close()

        events = [
            {
                "id": r[0],
                "time": r[1],
                "aspect_ratio": r[2],
                "torso_angle": r[3],
                "status": "DISPATCHED"
            }
            for r in rows
        ]
        return jsonify(events)
    except Exception as e:
        print(f"[DB ERROR] Event query failed: {e}")
        return jsonify([])


if __name__ == '__main__':
    print("\n--- AI Fall Detection Web Station Active ---")
    print("Dashboard URL: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)