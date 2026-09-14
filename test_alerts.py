import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.alert_dispatcher import AlertDispatcher
from src.config import Config

def run_multi_channel_test():
    print("=" * 50)
    print("🚀 INITIATING MULTI-CHANNEL ALERT SYSTEM TEST")
    print("=" * 50)

    # 1. Create a dummy test snapshot image
    test_snapshot_path = Config.SNAPSHOT_DIR / "test_snapshot.jpg"
    
    # Generate a sample frame with text
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        test_image, 
        "EMERGENCY FALL TEST", 
        (120, 240), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (0, 0, 255), 
        2, 
        cv2.LINE_AA
    )
    cv2.imwrite(str(test_snapshot_path), test_image)
    print(f"📸 Test snapshot generated: {test_snapshot_path}")

    # 2. Dispatch across all 3 channels
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dispatcher = AlertDispatcher()

    print("\n⏳ Dispatching alerts concurrently (Telegram, Email, ntfy)...")
    dispatcher.dispatch_all(str(test_snapshot_path), timestamp)
    print("✅ All alert threads triggered. Check your phone and email inbox!")
    print("=" * 50)

if __name__ == "__main__":
    run_multi_channel_test()