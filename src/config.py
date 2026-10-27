import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Config:
    BASE_DIR = BASE_DIR
    DB_PATH = BASE_DIR / "database" / "fall_events.db"
    SNAPSHOT_DIR = BASE_DIR / "data" / "snapshots"
    LOG_DIR = BASE_DIR / "data" / "logs"
    YOLO_MODEL_PATH = BASE_DIR / "runs" / "detect" / "caucafall_yolov8s" / "weights" / "best.pt"
    # Thresholds
    FALL_ASPECT_RATIO_THRESH = 1.35
    FALL_ANGLE_THRESH = 32.0
    CONFIRMATION_TIME_SEC = 3.0

    # Telegram Credentials
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # SMTP Email Credentials
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
    SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
    RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")

    # ntfy Push Alert Topic (SMS Alternative)
    NTFY_TOPIC = os.getenv("NTFY_TOPIC", "fall_detection_alert_default")

Config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
Config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)