import cv2

input_file = "data/uploads/fall_stairs.mp4"      # Path to your current 5-second trimmed clip
output_file = "data/uploads/fall_stairs_ready.mp4"

cap = cv2.VideoCapture(input_file)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

last_frame = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    last_frame = frame

# Hold the final lying-down frame for an additional 4.0 seconds (approx. 120 frames)
if last_frame is not None:
    hold_frames = int(fps * 4.0)
    for _ in range(hold_frames):
        out.write(last_frame)

cap.release()
out.release()
print("Extended clip saved successfully as:", output_file)