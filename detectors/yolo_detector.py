class YOLODetector:
    def __init__(self, model_path="yolov12n.pt"):
        self.model = YOLO(model_path)
        self.model.classes = [2, 3, 5, 7]

    def detect(self, frame):
        results = self.model(frame)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls,

                })
        return detections
