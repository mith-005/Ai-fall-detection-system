import cv2
import sys

def check_available_cameras(max_tested=5):
    """Scans camera indices from 0 to max_tested and returns working indices."""
    working_cameras = []
    print("[INFO] Scanning for available camera devices...")
    for index in range(max_tested):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow backend for Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                working_cameras.append(index)
                print(f"  --> Camera index {index}: WORKING (Resolution: {frame.shape[1]}x{frame.shape[0]})")
            cap.release()
    return working_cameras

def run_camera_stream(camera_index):
    """Opens selected camera index and displays real-time window."""
    # Using cv2.CAP_DSHOW provides faster & reliable webcam access on Windows 11
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {camera_index}.")
        return

    print(f"[SUCCESS] Streaming from camera index {camera_index}. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame from camera.")
            break

        cv2.imshow(f"Camera Index {camera_index} Test", frame)

        # Press 'q' to break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = check_available_cameras()
    
    if not cameras:
        print("\n[ALERT] No working webcam detected!")
        print("Check the following Windows settings:")
        print(" 1. Go to Windows Settings -> Privacy & security -> Camera.")
        print(" 2. Ensure 'Camera access' is turned ON.")
        print(" 3. Ensure 'Let apps access your camera' and 'Let desktop apps access your camera' are BOTH turned ON.")
        print(" 4. If your laptop (Dell G15) has a physical privacy slider or Fn camera key (e.g. Fn + F6/F10), toggle it ON.")
        sys.exit(1)

    print(f"\n[INFO] Opening primary working camera index: {cameras[0]}")
    run_camera_stream(cameras[0])