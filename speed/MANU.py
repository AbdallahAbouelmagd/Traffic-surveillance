import numpy as np
import cv2
import os

class PerspectiveTransformerManual:
    def __init__(self, frame=None, default_width=1280, default_height=720):
        if frame is not None:
            self.height, self.width = frame.shape[:2]
        else:
            self.width = default_width
            self.height = default_height
        self.matrix = None


    def compute_perspective_matrix(self, frame, src_pts):
        dst_pts = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width, 0],
            [0, 0]
        ])
        H, _ = cv2.findHomography(src_pts, dst_pts)
        self.matrix = H
        return H

    def warp(self, frame):
        if self.matrix is None:
            raise Exception("Matrix ist nicht gesetzt.")
        return cv2.warpPerspective(frame, self.matrix, (self.width, self.height))

    def save_matrix(self, path):
        if self.matrix is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, self.matrix)
            print(f"[INFO] Matrix gespeichert unter: {path}")
        else:
            print("[FEHLER] Keine Matrix zum Speichern vorhanden.")

def select_four_points(image):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"📌 Punkt {len(points)}: ({x}, {y})")
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Bild", image)

    cv2.imshow("Bild", image)
    cv2.setMouseCallback("Bild", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError("❌ Es wurden nicht genau 4 Punkte ausgewählt.")
    return np.float32(points)