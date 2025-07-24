import os
import numpy as np
import cv2
MATRIX_FILE = "speed/data/perspectiv_matrix.npy"


class SpeedAnalyzer:
    def __init__(self, fps, reference_car_length_m=4.5,
                 upscale_factor=1, confidence_threshold=0.3,
                 max_time_gap=1.0, calibration_frames=30):
        self.fps = fps
        self.reference_car_length_m = reference_car_length_m
        self.pixel_to_meter = None
        self.car_tracks = {}
        self.max_time_gap = max_time_gap
        self.confidence_threshold = confidence_threshold
        self.upscale_factor = upscale_factor

        self.calibration_frames = calibration_frames
        self.calibration_data = []
        self.frame_count = 0

        self.speed_history = []

        if os.path.exists(MATRIX_FILE):
            try:
                matrix = np.load(MATRIX_FILE)
                if matrix.shape == (3, 3) and np.isfinite(matrix).all():
                    self.matrix = matrix
                    print("[INFO] Perspektivmatrix erfolgreich geladen.")
                else:
                    self.matrix = None
                    print("[FEHLER] Ungültige Perspektivmatrix.")
            except Exception as e:
                self.matrix = None
                print(f"[FEHLER] Matrix konnte nicht geladen werden: {e}")
        else:
            self.matrix = None
            print("[WARNUNG] Keine Perspektivmatrix gefunden. Kalibrierung nötig.")

    def warp_point(self, point):
        if self.matrix is None:
            return point
        src = np.array([[point]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.matrix)
        return tuple(dst[0][0])

    def compute_speed(self, track_id, bbox, current_time, confidence=1.0):

        if confidence < self.confidence_threshold:
            return None

        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        current_warped = self.warp_point(center)


        if self.pixel_to_meter is None:
            p1 = self.warp_point((x1, y1))
            p2 = self.warp_point((x2, y1))
            pixel_length = np.linalg.norm(np.array(p2) - np.array(p1))
            if pixel_length > 1e-3:
                self.calibration_data.append(pixel_length)

            self.frame_count += 1
            if self.frame_count >= self.calibration_frames and len(self.calibration_data) > 0:
                median_length = np.median(self.calibration_data)
                self.pixel_to_meter = (self.reference_car_length_m / median_length) / self.upscale_factor
                print(f"[INFO] Kalibriert: 1 Pixel = {self.pixel_to_meter:.6f} m (Median: {median_length:.2f} px)")
            return None


        if track_id not in self.car_tracks:
            self.car_tracks[track_id] = (current_warped, current_time)
            return None

        prev_pos, prev_time = self.car_tracks[track_id]
        time_diff = current_time - prev_time

        if time_diff <= 0 or time_diff > self.max_time_gap:
            self.car_tracks[track_id] = (current_warped, current_time)
            return None

        dist_pix = np.linalg.norm(np.array(current_warped) - np.array(prev_pos))
        speed_m_s = (dist_pix * self.pixel_to_meter) / time_diff
        speed_kmh = speed_m_s * 3.6


        self.speed_history.append(speed_kmh)

        self.car_tracks[track_id] = (current_warped, current_time)
        return speed_kmh

    def get_average_speed(self):
        if len(self.speed_history) == 0:
            return None
        return sum(self.speed_history) / len(self.speed_history)
