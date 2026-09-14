import cv2

# Try opening camera with DirectShow backend for Windows 11
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("[ERROR] Webcam could not be opened.")
    print("Fixes to try:")
    print("1. Slide open the physical privacy switch above your laptop screen.")
    print("2. Press Fn + F6 (or Fn + F9) to enable webcam hardware.")
    print("3. Open Windows Settings -> Privacy & security -> Camera -> Turn ON 'Let desktop apps access your camera'.")
else:
    print("[SUCCESS] Webcam opened! Press 'q' on the video window to close.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed.")
            break
        cv2.imshow("Webcam Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()