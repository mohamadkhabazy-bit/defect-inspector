from ultralytics import YOLO
import numpy as np
import cv2
from datetime import datetime

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

CLASS_COLORS = {
    "crazing":         (255, 100,  50),
    "inclusion":       ( 50, 200, 255),
    "patches":         ( 50, 255, 150),
    "pitted_surface":  (255,  50, 200),
    "rolled-in_scale": (255, 220,  50),
    "scratches":       (100,  50, 255),
}

class DefectDetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        self.log = []

    def predict(self, image: np.ndarray):
        results = self.model.predict(image, conf=self.conf, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2

            det = {
                "class":      cls_name,
                "confidence": round(conf, 3),
                "bbox":       (x1, y1, x2, y2),
                "center":     (cx, cy),
                "time":       datetime.now().strftime("%H:%M:%S"),
            }
            detections.append(det)
            self.log.append(det)

        annotated = self._draw(image.copy(), detections)
        return annotated, detections

    def _draw(self, image, detections):
        for det in detections:
            cls  = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(cls, (200, 200, 200))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(image, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return image

    def reset(self):
        self.log = []