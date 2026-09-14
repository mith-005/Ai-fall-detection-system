import sqlite3
import datetime
from src.config import Config

class DatabaseHandler:
    def __init__(self, db_path=Config.DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        """Creates the event log database table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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

    def log_event(self, aspect_ratio, torso_angle, snapshot_path):
        """Records confirmed fall event details into local database."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fall_events (timestamp, aspect_ratio, torso_angle, snapshot_path)
                VALUES (?, ?, ?, ?)
            """, (timestamp, aspect_ratio, torso_angle, str(snapshot_path)))
            conn.commit()