import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from config import Config

class AlertDispatcher:
    def __init__(self):
        self.config = Config

    def send_email_alert(self, snapshot_path, timestamp):
        """1st Priority: Formal SMTP Email with Snapshot Attachment."""
        sender = str(self.config.SENDER_EMAIL).strip('"\'')
        pwd = str(self.config.SENDER_PASSWORD).strip('"\'')
        recipient = str(self.config.RECIPIENT_EMAIL).strip('"\'')

        if not sender or not pwd or not recipient:
            print("[EMAIL ERROR] Missing email credentials in config/.env.")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = recipient
            msg["Subject"] = f"🚨 URGENT: Fall Detected at {timestamp}"
            body = f"<p>A fall was confirmed at <strong>{timestamp}</strong>. Snapshot is attached below.</p>"
            msg.attach(MIMEText(body, "html"))

            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    img_data = MIMEImage(f.read(), name=os.path.basename(snapshot_path))
                    msg.attach(img_data)

            server = smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT, timeout=10)
            server.starttls()
            server.login(sender, pwd)
            server.sendmail(sender, recipient, msg.as_string())
            server.quit()
            print("[EMAIL LOG] Email delivered successfully.")
            return True
        except Exception as e:
            print(f"[EMAIL ERROR] {e}")
            return False

    def send_ntfy_push(self, snapshot_path, timestamp):
        """2nd Priority: Instant Mobile Buzzer with Attached Screenshot."""
        topic = str(self.config.NTFY_TOPIC).strip('"\'')
        if not topic:
            print("[NTFY ERROR] Missing NTFY_TOPIC in config/.env.")
            return False
        try:
            url = f"https://ntfy.sh/{topic}"
            headers = {
                "Title": "EMERGENCY: Fall Detected!",
                "Priority": "5",
                "Tags": "warning,rotating_light,sos",
                "Message": f"Fall confirmed at {timestamp}. Screenshot attached."
            }

            if snapshot_path and os.path.exists(snapshot_path):
                headers["Filename"] = os.path.basename(snapshot_path)
                with open(snapshot_path, "rb") as img:
                    res = requests.post(url, data=img, headers=headers, timeout=8)
            else:
                message_body = f"Fall confirmed at {timestamp}.".encode("utf-8")
                res = requests.post(url, data=message_body, headers=headers, timeout=8)

            print(f"[NTFY LOG] Status Code: {res.status_code}")
            return res.status_code == 200
        except Exception as e:
            print(f"[NTFY ERROR] {e}")
            return False

    def send_telegram_alert(self, snapshot_path, timestamp):
        """3rd Priority: Instant Photo Snapshot in Telegram."""
        token = str(self.config.TELEGRAM_BOT_TOKEN).strip('"\'')
        chat_id = str(self.config.TELEGRAM_CHAT_ID).strip('"\'')
        
        if not token or not chat_id:
            print("[TELEGRAM ERROR] Missing bot token or chat ID.")
            return False

        caption = f"🚨 *EMERGENCY: FALL DETECTED!*\n📅 *Time:* `{timestamp}`\n⚠️ *Status:* Confirmed Ground Fall (> 3.0s)"

        try:
            if snapshot_path and os.path.exists(snapshot_path):
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                with open(snapshot_path, "rb") as img:
                    res = requests.post(
                        url,
                        data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"},
                        files={"photo": img},
                        timeout=10
                    )
            else:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                res = requests.post(
                    url,
                    json={"chat_id": chat_id, "text": caption, "parse_mode": "Markdown"},
                    timeout=10
                )
            print(f"[TELEGRAM LOG] Status Code: {res.status_code}")
            return res.status_code == 200
        except Exception as e:
            print(f"[TELEGRAM ERROR] {e}")
            return False

    def dispatch_fallback_sequence(self, snapshot_path, timestamp):
        """Executes strictly ONE channel at a time with cascading fallback (Email -> ntfy -> Telegram)."""
        print(f"\n[ALERT DISPATCH] Starting sequential alert dispatch for {timestamp}...")

        # 1. Try Email first
        print("[ALERT TRY] 1. Attempting Primary Email dispatch...")
        if self.send_email_alert(snapshot_path, timestamp):
            print("[ALERT SUCCESS] Delivered via Email. Stopping fallback sequence.")
            return True

        # 2. Fall back to ntfy only if Email failed
        print("[ALERT FALLBACK] Email failed. 2. Attempting ntfy mobile buzzer...")
        if self.send_ntfy_push(snapshot_path, timestamp):
            print("[ALERT SUCCESS] Delivered via ntfy. Stopping fallback sequence.")
            return True

        # 3. Fall back to Telegram only if both Email and ntfy failed
        print("[ALERT FALLBACK] ntfy failed. 3. Attempting Telegram...")
        if self.send_telegram_alert(snapshot_path, timestamp):
            print("[ALERT SUCCESS] Delivered via Telegram.")
            return True

        print("[ALERT CRITICAL] All alert channels failed!\n")
        return False