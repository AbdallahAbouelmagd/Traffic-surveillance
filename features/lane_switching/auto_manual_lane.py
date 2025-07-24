import cv2
import numpy as np
import pickle
import os

from .lane_utils import (
    compute_bezier_curve,
    create_lane_space,
    color_filter,
    region_of_interest,
    filter_lines_by_angle,
    filter_lines_by_clustering,
    point_line_distance
)


#VIDEO_PATH = "../../videos/road_video2.mp4"
RESIZE_DIM = (960, 540)

selected_lines = []
clicked_points = []
skip_connection_indices = set()
new_group_mode = False
dragging_index = None


def preprocess_frame(frame):
    filtered = color_filter(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 10, 100)

    height, width = edges.shape
    roi_vertices = np.array([[ (int(0.02 * width), height),
                               (int(0.35 * width), int(0.6 * height)),
                               (int(0.9 * width), int(0.6 * height)),
                               (int(0.95 * width), height) ]], dtype=np.int32)

    roi_edges = region_of_interest(edges, roi_vertices)
    return filtered, edges, roi_edges

def mouse_callback(event, x, y, flags, param):
    global selected_lines, new_group_mode, skip_connection_indices
    frame, lines = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if lines is None:
            return
        for idx, line in enumerate(lines):
            for x1, y1, x2, y2 in line:
                if point_line_distance((x1, y1), (x2, y2), (x, y)) < 10:
                    existing = [i for i, (i_line, _) in enumerate(selected_lines) if i_line == idx]
                    if existing:
                        index_removed = existing[0]
                        del selected_lines[index_removed]
                        skip_connection_indices = {i - 1 if i > index_removed else i for i in skip_connection_indices}
                        print(f"Deselected line {idx}")
                    else:
                        selected_lines.append((idx, [x1, y1, x2, y2]))
                        print(f"Selected line {idx}")
                        if new_group_mode:
                            skip_connection_indices.add(len(selected_lines) - 1)
                            new_group_mode = False
                            print("New group mode OFF")
                    return

def find_nearest_point(x, y, threshold=10):
    for i, (px, py) in enumerate(clicked_points):
        if np.hypot(px - x, py - y) < threshold:
            return i
    return None

def click_event(event, x, y, flags, param):
    global dragging_index, clicked_points, new_group_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging_index = find_nearest_point(x, y)
        if dragging_index is None:
            clicked_points.append((x, y))
            if new_group_mode and len(clicked_points) % 3 == 0:
                skip_connection_indices.add(len(clicked_points) // 3 - 1)
                new_group_mode = False
                print("New group mode OFF")
        redraw_canvas(param)
    elif event == cv2.EVENT_MOUSEMOVE and dragging_index is not None:
        clicked_points[dragging_index] = (x, y)
        redraw_canvas(param)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_index = None

def redraw_canvas(frame):
    canvas = frame.copy()
    for i, point in enumerate(clicked_points):
        color = (0, 0, 255) if (i % 3 == 1) else (0, 255, 0)
        cv2.circle(canvas, point, 5, color, -1)

    if len(clicked_points) >= 3:
        for i in range(0, len(clicked_points) - 2, 3):
            bezier = compute_bezier_curve(*clicked_points[i:i+3])
            for j in range(1, len(bezier)):
                cv2.line(canvas, tuple(bezier[j-1]), tuple(bezier[j]), (255, 255, 0), 2)

    cv2.imshow("Define Lanes", canvas)

def get_lane_polygons(video_path, lane_file_path):
    global VIDEO_PATH
    VIDEO_PATH = video_path
    main(lane_file_path)

    if not os.path.exists(lane_file_path):
        raise FileNotFoundError(f"{lane_file_path} was not created.")

    with open(lane_file_path, "rb") as f:
        lane_polygons = pickle.load(f)

    return lane_polygons

def main(lane_file_path):
    global selected_lines, clicked_points, new_group_mode

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    frame = cv2.resize(frame, RESIZE_DIM)
    _, _, roi_edges = preprocess_frame(frame)
    raw_lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=100)

    angle_filtered = filter_lines_by_angle(raw_lines, min_angle_deg=28)
    clustered_lines = filter_lines_by_clustering(angle_filtered, eps=30, min_samples=1)

    lines = clustered_lines

    cv2.namedWindow("Select Lines")
    cv2.setMouseCallback("Select Lines", mouse_callback, (frame, lines))
    print("Click lines to select/deselect. Press 'n' to toggle new group, 's' to save, 'q' to quit.")

    while True:
        disp = frame.copy()
        for idx, line in enumerate(lines):
            for x1, y1, x2, y2 in line:
                if any(li == idx for li, _ in selected_lines):
                    coords = [coords for li, coords in selected_lines if li == idx][0]
                    cv2.line(disp, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 3)
                else:
                    cv2.line(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(disp, f"New Group Mode: {'ON' if new_group_mode else 'OFF'}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if new_group_mode else (0, 255, 0), 2)
        cv2.imshow("Select Lines", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            new_group_mode = not new_group_mode
            print(f"New group mode: {'ON' if new_group_mode else 'OFF'}")
        elif key == ord('s'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    for _, coords in selected_lines:
        p1 = (coords[0], coords[1])
        p2 = (coords[2], coords[3])
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        clicked_points.extend([p1, mid, p2])

    ret, first_frame = cv2.VideoCapture(VIDEO_PATH).read()
    first_frame = cv2.resize(first_frame, RESIZE_DIM)
    cv2.namedWindow("Define Lanes")
    cv2.setMouseCallback("Define Lanes", click_event, first_frame)
    redraw_canvas(first_frame)

    print("Add curves. Press 'n' to toggle new group, 'q' when done.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            new_group_mode = not new_group_mode
            print(f"New group mode: {'ON' if new_group_mode else 'OFF'}")

    cv2.destroyAllWindows()

    lane_polygons = []
    num_curves = len(clicked_points) // 3
    i = 0
    while i < num_curves - 1:
        if (i + 1) not in skip_connection_indices:
            bez1 = compute_bezier_curve(*clicked_points[i * 3: i * 3 + 3])
            bez2 = compute_bezier_curve(*clicked_points[(i + 1) * 3: (i + 1) * 3 + 3])
            poly = create_lane_space(bez1, bez2)
            lane_polygons.append(poly)

        if (i + 1) in skip_connection_indices:
            bez1 = compute_bezier_curve(*clicked_points[(i + 1) * 3: (i + 1) * 3 + 3])
            bez2 = compute_bezier_curve(*clicked_points[(i + 2) * 3: (i + 2) * 3 + 3])
            poly = create_lane_space(bez1, bez2)
            lane_polygons.append(poly)
            i += 2
        else:
            i += 1

    os.makedirs(os.path.dirname(lane_file_path), exist_ok=True)
    with open(lane_file_path, "wb") as f:
        pickle.dump(lane_polygons, f)

    print(f"Saved lane polygons to {lane_file_path}")

if __name__ == "__main__":
    main()
