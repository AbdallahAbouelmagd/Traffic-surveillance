import cv2
import numpy as np
import pickle
import os
from sklearn.cluster import DBSCAN


VIDEO_PATH = "../../../videos/cars.mp4"     # the only not great with auto: road_video3.mp4 + road_video.mp4
LANE_FILE_PATH = "../../../data/lanes.pkl"
RESIZE_DIM = (960, 540)

selected_lines = []
clicked_points = []
skip_connection_indices = set()
new_group_mode = False
dragging_index = None


def compute_bezier_curve(pt0, pt1, pt2, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve_x = (1 - t)**2 * pt0[0] + 2 * (1 - t) * t * pt1[0] + t**2 * pt2[0]
    curve_y = (1 - t)**2 * pt0[1] + 2 * (1 - t) * t * pt1[1] + t**2 * pt2[1]
    return np.vstack((curve_x, curve_y)).T.astype(int)


def create_lane_space(curve1, curve2):
    return np.vstack((curve1, curve2[::-1])).astype(np.int32)


def filter_lines_by_angle(lines, min_angle_deg=28):
    if lines is None:
        return []
    filtered = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) > min_angle_deg:
                filtered.append([[x1, y1, x2, y2]])
    return np.array(filtered)


def color_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(img, img, mask=combined_mask), combined_mask


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def preprocess_frame(frame):
    masked, combined_mask = color_filter(frame)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 10, 100)

    height, width = edges.shape
    roi_vertices = np.array([[ (int(0.02 * width), height),
                               (int(0.35 * width), int(0.6 * height)),
                               (int(0.9 * width), int(0.6 * height)),
                               (int(0.95 * width), height) ]], dtype=np.int32)

    roi_edges = region_of_interest(edges, roi_vertices)
    return masked, combined_mask, edges, roi_edges


def point_line_distance(p1, p2, p):
    p1, p2, p = np.array(p1), np.array(p2), np.array(p)
    line_vec = p2 - p1
    p_vec = p - p1
    line_len = np.dot(line_vec, line_vec)
    if line_len == 0:
        return np.linalg.norm(p - p1)
    t = np.clip(np.dot(p_vec, line_vec) / line_len, 0, 1)
    proj = p1 + t * line_vec
    return np.linalg.norm(p - proj)


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
                        new_skips = {i - 1 if i > index_removed else i for i in skip_connection_indices}
                        skip_connection_indices = new_skips
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

def filter_lines_by_proximity(lines, min_distance=20):
    if lines is None or len(lines) == 0:
        return lines

    filtered = []
    for i, line1 in enumerate(lines):
        x1, y1, x2, y2 = line1[0]
        mid1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        too_close = False
        for line2 in filtered:
            x3, y3, x4, y4 = line2[0]
            mid2 = np.array([(x3 + x4) / 2, (y3 + y4) / 2])
            if np.linalg.norm(mid1 - mid2) < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(line1)
    return np.array(filtered)
def filter_lines_by_clustering(lines, eps=30, min_samples=1):
    if lines is None or len(lines) == 0:
        return lines

    features = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        features.append([mid_x, mid_y, angle * 5])
    features = np.array(features)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_

    filtered_lines = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            cluster_lines = lines[labels == cluster_id]
        else:
            cluster_lines = lines[labels == cluster_id]

        max_len = 0
        rep_line = None
        for line in cluster_lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > max_len:
                max_len = length
                rep_line = line
        if rep_line is not None:
            filtered_lines.append(rep_line)

    return np.array(filtered_lines)

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


def main():
    global selected_lines, clicked_points, new_group_mode

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    frame = cv2.resize(frame, RESIZE_DIM)
    masked, mask_img, edges, roi_edges = preprocess_frame(frame)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Mask Only", mask_img)
    cv2.imshow("Color Masked", masked)
    cv2.imshow("Edges", edges)
    cv2.imshow("ROI Edges", roi_edges)

    raw_lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=100)
    raw_hough = frame.copy()
    if raw_lines is not None:
        for line in raw_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(raw_hough, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imshow("Raw Hough Lines", raw_hough)

    angle_filtered = filter_lines_by_angle(raw_lines)
    angle_img = frame.copy()
    if angle_filtered is not None:
        for line in angle_filtered:
            for x1, y1, x2, y2 in line:
                cv2.line(angle_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow("After Angle Filter", angle_img)

    # lines = filter_lines_by_proximity(angle_filtered, min_distance=30)
    lines = filter_lines_by_clustering(angle_filtered, eps=30, min_samples=1)
    # lines = angle_filtered
    proximity_img = frame.copy()
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(proximity_img, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.imshow("After Proximity Filter", proximity_img)
    cv2.imshow("Hough Lines", proximity_img)  # keep this for compatibility with rest of code

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
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if new_group_mode else (0, 255, 0), 2)
        cv2.imshow("Select Lines", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            new_group_mode = not new_group_mode
        elif key == ord('s'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    for _, coords in selected_lines:
        clicked_points.extend([
            (coords[0], coords[1]),
            ((coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2),
            (coords[2], coords[3])
        ])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    frame = cv2.resize(frame, RESIZE_DIM)
    cv2.namedWindow("Define Lanes")
    cv2.setMouseCallback("Define Lanes", click_event, frame)
    redraw_canvas(frame)

    print("Add curves. Press 'n' to toggle new group, 'q' when done.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            new_group_mode = not new_group_mode

    cv2.destroyAllWindows()

    lane_polygons = []
    num_curves = len(clicked_points) // 3
    for i in range(num_curves - 1):
        if (i + 1) in skip_connection_indices or i in skip_connection_indices:
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

    os.makedirs(os.path.dirname(LANE_FILE_PATH), exist_ok=True)
    with open(LANE_FILE_PATH, "wb") as f:
        pickle.dump(lane_polygons, f)

    print(f"Saved lane polygons to {LANE_FILE_PATH}")

    vis = frame.copy()
    for poly in lane_polygons:
        cv2.polylines(vis, [poly], isClosed=True, color=(255, 0, 255), thickness=2)
    cv2.imshow("Final Lanes", vis)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
