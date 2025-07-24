import cv2
import numpy as np

VIDEO_PATH = "../../../videos/road_video4.mp4"
RESIZE_DIM = (960, 540)

selected_lines = []
dragging_point = None
add_mode = False
new_line_pts = []


def filter_lines_by_angle(lines, min_angle_deg=20):
    if lines is None:
        return []
    filtered = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) > min_angle_deg:
                filtered.append(line)
    return np.array(filtered)


def color_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(img, img, mask=combined_mask)
    return filtered


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def preprocess_frame(frame):
    filtered = color_filter(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 150)
    height, width = edges.shape
    roi_vertices = np.array([[
        (int(0.02 * width), height),
        (int(0.35 * width), int(0.6 * height)),
        (int(0.9 * width), int(0.6 * height)),
        (int(0.95 * width), height)
    ]], dtype=np.int32)
    roi_edges = region_of_interest(edges, roi_vertices)
    return filtered, edges, roi_edges


def mouse_callback(event, x, y, flags, param):
    global selected_lines, dragging_point, add_mode, new_line_pts
    frame, lines = param

    if add_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            new_line_pts.append((x, y))
            if len(new_line_pts) == 2:
                selected_lines.append((-1, [new_line_pts[0][0], new_line_pts[0][1],
                                            new_line_pts[1][0], new_line_pts[1][1]]))
                print(f"Added new line: {selected_lines[-1][1]}")
                new_line_pts = []
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, (line_idx, coords) in enumerate(selected_lines):
            for pt_idx, (px, py) in enumerate([(coords[0], coords[1]), (coords[2], coords[3])]):
                if np.hypot(x - px, y - py) < 10:
                    dragging_point = (idx, pt_idx)
                    return

        if lines is None:
            return
        for idx, line in enumerate(lines):
            for x1, y1, x2, y2 in line:
                dist = point_line_distance((x1, y1), (x2, y2), (x, y))
                if dist < 10:
                    exists = [i for i, (li, _) in enumerate(selected_lines) if li == idx]
                    if exists:
                        del selected_lines[exists[0]]
                        print(f"Deselected line {idx}")
                    else:
                        selected_lines.append((idx, [x1, y1, x2, y2]))
                        print(f"Selected line {idx}")
                    return

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            line_idx, pt_idx = dragging_point
            _, coords = selected_lines[line_idx]
            if pt_idx == 0:
                coords[0] = x
                coords[1] = y
            else:
                coords[2] = x
                coords[3] = y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None


def point_line_distance(p1, p2, p):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p = np.array(p)
    line_vec = p2 - p1
    p_vec = p - p1
    line_len = np.dot(line_vec, line_vec)
    if line_len == 0:
        return np.linalg.norm(p - p1)
    t = max(0, min(1, np.dot(p_vec, line_vec) / line_len))
    proj = p1 + t * line_vec
    return np.linalg.norm(p - proj)


def draw_lines(frame, lines, selected_lines, new_line_pts):
    overlay = frame.copy()
    for idx, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            if any(li == idx for li, _ in selected_lines):
                coords = [coords for li, coords in selected_lines if li == idx][0]
                cv2.line(overlay, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 4)
                cv2.circle(overlay, (coords[0], coords[1]), 6, (255, 0, 0), -1)
                cv2.circle(overlay, (coords[2], coords[3]), 6, (255, 0, 0), -1)
            else:
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
    # Draw manually added lines too
    for li, coords in selected_lines:
        if li == -1:
            cv2.line(overlay, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 4)
            cv2.circle(overlay, (coords[0], coords[1]), 6, (255, 0, 0), -1)
            cv2.circle(overlay, (coords[2], coords[3]), 6, (255, 0, 0), -1)
    # Draw new line points during adding
    for pt in new_line_pts:
        cv2.circle(overlay, pt, 5, (0, 255, 255), -1)
    return overlay


def main():
    global selected_lines, add_mode, new_line_pts

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    frame = cv2.resize(frame, RESIZE_DIM)
    _, _, roi_edges = preprocess_frame(frame)
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
    lines = filter_lines_by_angle(lines, min_angle_deg=20)

    cv2.namedWindow("Calibrate Lines")
    cv2.setMouseCallback("Calibrate Lines", mouse_callback, param=(frame, lines))

    print("Instructions:")
    print("- Click lines to select/deselect.")
    print("- Drag endpoints to adjust.")
    print("- Press 'a' to ADD new lines by clicking start & end.")
    print("- Press 's' to save and play.")
    print("- Press 'q' to quit.")

    while True:
        display_frame = draw_lines(frame, lines, selected_lines, new_line_pts)
        mode_text = "MODE: ADD NEW LINE" if add_mode else "MODE: SELECT/DRAG"
        cv2.putText(display_frame, mode_text + " | 'a'=toggle add | 's'=save | 'q'=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibrate Lines", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            add_mode = not add_mode
            new_line_pts = []
            print(f"Add mode: {add_mode}")
        elif key == ord('s'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    adjusted_lines = [coords for _, coords in selected_lines]

    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, RESIZE_DIM)
        output = frame.copy()
        for coords in adjusted_lines:
            cv2.line(output, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 4)
        cv2.imshow("Final Adjusted Lines", output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
