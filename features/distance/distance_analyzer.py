from collections import defaultdict

class DistanceAnalyzer:
    def __init__(self, correction_factor=1.0):
        self.lane_boxes = defaultdict(list)
        self.lane_dist_history = defaultdict(list)

        self.vehicle_length_map = {
            "car": 4.5,
            "truck": 10.0,
            "motorcycle": 2,
            "bus": 12.0,
            "van": 5.0
        }

        self.calibration_values = []
        self.max_calibration_frames = 20
        self.pixel_to_meter = None
        self.correction_factor = correction_factor

    def reset(self):
        self.lane_boxes.clear()

    def calibrate(self, vehicle_type, bbox):
        if self.pixel_to_meter is not None:
            return

        if vehicle_type in self.vehicle_length_map:
            height_px = bbox[3] - bbox[1]

            if height_px > 0:
                estimated_length_m = self.vehicle_length_map[vehicle_type]
                self.calibration_values.append(estimated_length_m / height_px)

            if len(self.calibration_values) >= self.max_calibration_frames:
                raw_pixel_to_meter = sum(self.calibration_values) / len(self.calibration_values)
                self.pixel_to_meter = raw_pixel_to_meter * self.correction_factor
                print(f"[INFO] Dynamische Kalibrierung abgeschlossen.")
                print(f"       1 Pixel = {self.pixel_to_meter:.6f} m (inkl. Korrekturfaktor)")

    def add_vehicle(self, lane_id, bbox, vehicle_type="car"):
        if lane_id is not None:
            self.lane_boxes[lane_id].append((bbox, vehicle_type))

    def compute_distances(self):
        average_distances = {}

        for lane_id, boxes_with_types in self.lane_boxes.items():
            distances = []

            if self.pixel_to_meter is None and len(boxes_with_types) > 0:
                for box, vtype in boxes_with_types:
                    self.calibrate(vtype, box)
                    if self.pixel_to_meter is not None:
                        break

            if self.pixel_to_meter is None:
                continue

            boxes = [box for box, _ in boxes_with_types]
            if len(boxes) < 2:
                continue

            boxes_sorted = sorted(boxes, key=lambda b: b[3])

            for i in range(len(boxes_sorted) - 1):
                lower_y2 = boxes_sorted[i][3]
                upper_y1 = boxes_sorted[i + 1][1]
                pixel_distance = upper_y1 - lower_y2

                if pixel_distance > 0:
                    meter_distance = pixel_distance * self.pixel_to_meter
                    distances.append(meter_distance)

            if distances:
                avg_m = sum(distances) / len(distances)
                average_distances[lane_id] = avg_m
                self.lane_dist_history[lane_id].append(avg_m)

        return average_distances

    def get_average_distance_per_lane(self):
        final_results = {}
        for lane_id, values in self.lane_dist_history.items():
            if values and self.pixel_to_meter is not None:
                avg_m = sum(values) / len(values)
                final_results[lane_id] = avg_m
        return final_results