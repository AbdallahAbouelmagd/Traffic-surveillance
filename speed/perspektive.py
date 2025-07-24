import cv2
import torch
import numpy as np
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os
print("Arbeitsverzeichnis:", os.getcwd())
class PerspectiveTransformerAuto:
    def __init__(self, width=1280, height=720, use_cuda=False, debug=False):
        self.width = width
        self.height = height
        self.matrix = None
        self.debug = debug

        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        ).eval()

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect_street(self, frame):
        inputs = self.feature_extractor(images=frame, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            seg = torch.argmax(logits, dim=1)[0].cpu().numpy()

        road_mask = (seg == 0).astype(np.uint8) * 255  # Klasse 0 = Straße
        return road_mask

    def compute_perspective_matrix(self, frame):
        road_mask = self.detect_street(frame)

        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[WARNUNG] Keine Straße erkannt, verwende Default-Region.")
            h, w = road_mask.shape
            src_pts = np.float32([
                [w * 0.3, h],
                [w * 0.7, h],
                [w * 0.7, h * 0.5],
                [w * 0.3, h * 0.5]
            ])
        else:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            src_pts = np.float32([
                [x, y + h],
                [x + w, y + h],
                [x + w, y],
                [x, y]
            ])


        h_orig, w_orig = frame.shape[:2]
        h_mask, w_mask = road_mask.shape[:2]


        scale_x = w_orig / w_mask
        scale_y = h_orig / h_mask


        src_pts_scaled = src_pts.copy()
        src_pts_scaled[:, 0] *= scale_x
        src_pts_scaled[:, 1] *= scale_y



        dst_pts = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width, 0],
            [0, 0]
        ])

        if self.debug:
            debug_img = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            for pt in src_pts:
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
            cv2.imwrite("C:/Users/adhel/debug_road_mask_with_points.jpg", debug_img)
            print("[INFO] Debug-Bild 'debug_road_mask_with_points.jpg' gespeichert.")


        H, _ = cv2.findHomography(src_pts_scaled, dst_pts)
        if H is not None:
            self.matrix = H
            if self.debug:
                print("[INFO] Homographie-Matrix berechnet:")
                print(self.matrix)
        else:
            print("[WARNUNG] Homographie-Matrix konnte nicht berechnet werden.")

        return H, road_mask, src_pts_scaled

    def warp(self, frame):
        if self.matrix is None:
            raise Exception("Homographie-Matrix ist nicht gesetzt.")
        return cv2.warpPerspective(frame, self.matrix, (self.width, self.height))

    def save_matrix(self, path):
        if self.matrix is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, self.matrix)
            print(f"[INFO] Matrix gespeichert unter: {path}")

        else:
            print("[FEHLER] Keine Matrix zum Speichern vorhanden.")