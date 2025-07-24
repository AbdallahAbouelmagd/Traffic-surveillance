from ultralytics import YOLO


class YOLOByteTrackDetector:
    def __init__(self,
                 model_path="yolov12n.pt",
                 tracker_cfg_path="bytetrack.yaml"):

        self.model = YOLO(model_path)
        self.tracker_cfg = tracker_cfg_path
        self.model.model.classes = [2, 3, 5, 7]

    def detect(self, frame):
        results = self.model.track(source=frame, persist=True, tracker=self.tracker_cfg, verbose=False)

        result = results[0] if isinstance(results, list) else results

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                track_id = int(box.id[0].item()) if box.id is not None and len(box.id) > 0 else None
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": cls,
                    "track_id": track_id
                })
        return detections
