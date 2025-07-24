import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def generate_colors(n):
    cmap = plt.get_cmap("tab20")
    return [tuple(int(255 * c) for c in cmap(i % 20)[:3]) for i in range(n)]

def compute_bezier_curve(pt0, pt1, pt2, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve_x = (1 - t)**2 * pt0[0] + 2 * (1 - t) * t * pt1[0] + t**2 * pt2[0]
    curve_y = (1 - t)**2 * pt0[1] + 2 * (1 - t) * t * pt1[1] + t**2 * pt2[1]
    return np.vstack((curve_x, curve_y)).T.astype(int)

def create_lane_space(curve1, curve2):
    return np.vstack((curve1, curve2[::-1])).astype(np.int32)

def color_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(img, img, mask=combined_mask)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

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
            continue
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
