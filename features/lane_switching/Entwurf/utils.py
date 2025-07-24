import numpy as np

def compute_bezier_curve(pt0, pt1, pt2, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve_x = (1 - t)**2 * pt0[0] + 2 * (1 - t) * t * pt1[0] + t**2 * pt2[0]
    curve_y = (1 - t)**2 * pt0[1] + 2 * (1 - t) * t * pt1[1] + t**2 * pt2[1]
    return np.array([np.round(p).astype(int) for p in zip(curve_x, curve_y)])

def create_lane_space(curve1, curve2):
    polygon = np.vstack((curve1, curve2[::-1]))
    return polygon.astype(np.int32)
