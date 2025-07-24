import os
import pickle
import cv2
import numpy as np
import warnings
from collections import defaultdict
from speed.speed_analyzer import SpeedAnalyzer
warnings.filterwarnings("ignore", category=FutureWarning)
from detectors.Bytetracker import YOLOByteTrackDetector
from features.lane_switching.auto_manual_lane import get_lane_polygons
from features.distance.distance_analyzer import DistanceAnalyzer
from features.lane_switching.lane_utils import generate_colors

use_super_resolution = False
VIDEO_PATH = "videos/road_video2.mp4"
LANE_FILE = "data/lanes.pkl"
CONGESTION_THRESHOLD_M = 5.0

def main():
    vehicle_tracks_per_lane = defaultdict(set)

    if not os.path.exists(LANE_FILE):
        lane_polygons = get_lane_polygons(VIDEO_PATH, LANE_FILE)
    else:
        with open(LANE_FILE, "rb") as f:
            lane_polygons = pickle.load(f)

    upscale_factor = 2
    if use_super_resolution:
        scaled_polygons = [np.array(poly) * upscale_factor for poly in lane_polygons]
        scaled_polygons = [poly.astype(np.int32) for poly in scaled_polygons]
    else:
        scaled_polygons = [np.array(poly).astype(np.int32) for poly in lane_polygons]

    detector = YOLOByteTrackDetector("yolov12n.pt")
    correction_factor = 5.1
    distance_analyzer = DistanceAnalyzer(correction_factor=correction_factor)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    speed_analyzer = SpeedAnalyzer(fps)
    lane_colors = generate_colors(len(scaled_polygons))

    sr = None
    if use_super_resolution:
        sr = cv2.dnn_superres.DnnSuperResImpl.create()
        sr.readModel("FSRCNN_x2.pb")
        sr.setModel("fsrcnn", 2)

    previous_lane_ids = {}
    vehicle_class_counts = defaultdict(int)
    unique_track_ids = set()
    total_detections = 0
    vehicle_count_per_lane = defaultdict(int)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        current_time = frame_idx / fps

        frame = cv2.resize(frame, (960, 540))

        if use_super_resolution and sr is not None:
            frame = sr.upsample(frame)

        distance_analyzer.reset()
        results = detector.detect(frame)

        for det in results:
            x1, y1, x2, y2 = map(int, det['bbox'])
            track_id = det.get('track_id', None)
            conf = det['confidence']
            cls = det['class_id']
            cx, cy = (x1 + x2) // 2, y2

            vehicle_type = "unknown"
            if hasattr(detector, "model") and hasattr(detector.model, "names"):
                vehicle_type = detector.model.names.get(cls, "unknown")
            if vehicle_type in ["bus", "truck"]:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            else:
                cx, cy = (x1 + x2) // 2, y2

            lane_id = None
            bbox_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), ((x1 + x2) // 2, (y1 + y2) // 2)]

            detected_inside_any_lane = False
            for poly in scaled_polygons:
                poly_np = poly.reshape((-1, 1, 2))
                if any(cv2.pointPolygonTest(poly_np, pt, False) >= 0 for pt in bbox_points):
                    detected_inside_any_lane = True
                    break

            if detected_inside_any_lane:
                cx, cy = (x1 + x2) // 2, y2

                for idx, poly in enumerate(scaled_polygons):
                    poly_np = poly.reshape((-1, 1, 2))
                    if cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0:
                        lane_id = idx
                        break
            else:
                lane_id = None

            if track_id is not None:
                unique_track_ids.add(track_id)
            total_detections += 1
            vehicle_class_counts[vehicle_type] += 1

            if lane_id is not None and track_id is not None:
                prev_lane = previous_lane_ids.get(track_id)
                if prev_lane is not None and prev_lane != lane_id:
                    print(f"Vehicle with ID {track_id} switched from lane {prev_lane + 1} to lane {lane_id + 1}")
                previous_lane_ids[track_id] = lane_id
                distance_analyzer.add_vehicle(lane_id, (x1, y1, x2, y2))
                vehicle_count_per_lane[lane_id] += 1
                vehicle_tracks_per_lane[lane_id].add(track_id)

                speed = speed_analyzer.compute_speed(track_id, (x1, y1, x2, y2), current_time, confidence=conf)

                label = f"{vehicle_type} ID:{track_id} Lane:{lane_id + 1}"

                if speed is not None:
                    avg_speed = speed_analyzer.get_average_speed()
                    label += f" Speed:{speed:.1f} km/h"
                    if avg_speed is not None:
                        label += f" Avg:{avg_speed:.1f} km/h"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        overlay = frame.copy()
        for idx, poly in enumerate(scaled_polygons):
            poly_reshaped = poly.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [poly_reshaped], lane_colors[idx])
            cv2.polylines(frame, [poly_reshaped], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        avg_distances = distance_analyzer.compute_distances()
        y_pos = 60
        for lane_id, avg in avg_distances.items():
            text = f"Lane {lane_id + 1} Avg Dist: {avg:.1f} m"
            color = (0, 0, 255) if avg < CONGESTION_THRESHOLD_M else (0, 255, 0)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25

        cv2.imshow("Traffic Analysis with Distances", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if speed_analyzer.pixel_to_meter:
        print("\nAverage distances per lane (in meters):")
        final_distances = distance_analyzer.get_average_distance_per_lane()
        for lane_id, dist_m in final_distances.items():
            print(f" - Lane {lane_id + 1}: {dist_m:.2f} m")

        print("\nCongestion Analysis:")
        for lane_id, dist_m in final_distances.items():
            if dist_m < CONGESTION_THRESHOLD_M:
                print(f"  Lane {lane_id + 1}: CONGESTION detected (avg {dist_m:.2f} m)")
            else:
                print(f"  Lane {lane_id + 1}: No congestion (avg {dist_m:.2f} m)")
    else:
        print("[WARNING] No valid pixel-to-meter calibration available.")

    print("\nOverall Statistics:")
    print(f"Unique track IDs (tracked vehicles): {len(unique_track_ids)}")
    print("Analysis complete.")
    print("\nUnique tracked vehicles per lane:")
    for lane_id, track_ids in vehicle_tracks_per_lane.items():
        print(f" - Lane {lane_id + 1}: {len(track_ids)} unique vehicles (tracked)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()