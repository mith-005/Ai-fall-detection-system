import cv2
import torch
from ultralytics import YOLO
from src.config import Config

class YOLODetector:
    def __init__(self, model_path=None, conf_thresh=0.5):
        """
        Initializes YOLOv8 model for fall/person detection using custom weights.
        Utilizes CUDA GPU if available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load custom trained model from Config if not provided
        self.model_path = str(model_path or Config.YOLO_MODEL_PATH)
        print(f"[INFO] Initializing YOLO on device: {self.device.upper()}")
        print(f"[INFO] Loading model weights: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        self.conf_thresh = conf_thresh

    def detect_persons(self, frame):
        """
        Runs YOLOv8 detection on the input frame.
        Returns list of bounding boxes: [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        results = self.model(frame, verbose=False, device=self.device)[0]
        detected_boxes = []

        for box in results.boxes:
            conf = float(box.conf[0])

            if conf >= self.conf_thresh:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = self.model.names.get(cls_id, str(cls_id))
                
                # Append coordinates, confidence, class ID, and string label
                detected_boxes.append((x1, y1, x2, y2, conf, cls_id, label))

        return detected_boxes