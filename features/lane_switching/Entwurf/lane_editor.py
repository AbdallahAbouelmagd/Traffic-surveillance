import cv2
import pickle
from features.lane_switching.Entwurf.utils import compute_bezier_curve, create_lane_space

clicked_points = []
dragging_index = None
frame_for_clicks = None

skip_connection_indices = set()
new_lane_group = False

def redraw_canvas(first_frame):
    global frame_for_clicks
    frame_for_clicks = first_frame.copy()
    for i, point in enumerate(clicked_points):
        color = (0, 0, 255) if (i % 3 == 1) else (0, 255, 0)
        cv2.circle(frame_for_clicks, point, 5, color, -1)
    if len(clicked_points) >= 3:
        for i in range(0, len(clicked_points), 3):
            if i + 2 < len(clicked_points):
                bezier = compute_bezier_curve(*clicked_points[i:i+3])
                for j in range(1, len(bezier)):
                    cv2.line(frame_for_clicks, tuple(bezier[j-1]), tuple(bezier[j]), (255, 255, 0), 2)
    cv2.imshow("Define Lanes", frame_for_clicks)

def click_event(event, x, y, flags, param):
    global dragging_index
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging_index = find_nearest_point(x, y)
        if dragging_index is None:
            clicked_points.append((x, y))
            if new_lane_group and len(clicked_points) % 3 == 0:
                curve_idx = len(clicked_points) // 3 - 1
                skip_connection_indices.add(curve_idx)
        redraw_canvas(param)
    elif event == cv2.EVENT_MOUSEMOVE and dragging_index is not None:
        clicked_points[dragging_index] = (x, y)
        redraw_canvas(param)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_index = None

def find_nearest_point(x, y, threshold=10):
    for i, (px, py) in enumerate(clicked_points):
        if abs(px - x) < threshold and abs(py - y) < threshold:
            return i
    return None

def get_lane_polygons(video_path):
    global clicked_points, new_lane_group
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read video")
    cv2.namedWindow("Define Lanes")
    cv2.setMouseCallback("Define Lanes", click_event, param=first_frame)
    redraw_canvas(first_frame)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            new_lane_group = not new_lane_group
            print(f"Next-only mode: {'ON' if new_lane_group else 'OFF'}")

    cv2.destroyWindow("Define Lanes")
    lane_polygons = []
    num_curves = len(clicked_points) // 3

    for i in range(num_curves - 1):
        connect = True

        if (i + 1) in skip_connection_indices:
            continue

        if i in skip_connection_indices:
            continue

        bez1 = compute_bezier_curve(*clicked_points[i * 3: i * 3 + 3])
        bez2 = compute_bezier_curve(*clicked_points[(i + 1) * 3: (i + 1) * 3 + 3])
        poly = create_lane_space(bez1, bez2)
        lane_polygons.append(poly)

    for i in range(1, num_curves - 1):
        if i in skip_connection_indices:
            bez1 = compute_bezier_curve(*clicked_points[i * 3: i * 3 + 3])
            bez2 = compute_bezier_curve(*clicked_points[(i + 1) * 3: (i + 1) * 3 + 3])
            poly = create_lane_space(bez1, bez2)
            lane_polygons.append(poly)

    with open("data/lanes.pkl", "wb") as f:
        pickle.dump(lane_polygons, f)
    return lane_polygons
