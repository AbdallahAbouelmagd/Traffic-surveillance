import cv2
from speed.perspektive import PerspectiveTransformerAuto
from speed.MANU import PerspectiveTransformerManual, select_four_points
import numpy as np
VIDEO_PATH = "C:/Users/adhel/Desktop/road_video4.mp4"
MATRIX_FILE = "data/perspectiv_matrix.npy"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[FEHLER] Video konnte nicht geladen werden!")
        return

    height, width = frame.shape[:2]

    print("Wähle Kalibrierungsmethode:")
    print("1 = Automatisch (Segformer)")
    print("2 = Manuell (4 Punkte klicken)")
    mode = input("👉 Eingabe (1 oder 2): ").strip()

    if mode == "1":
        transformer = PerspectiveTransformerAuto(width=width, height=height)
        matrix, road_mask, src_pts = transformer.compute_perspective_matrix(frame)
        if matrix is not None:
            warped = transformer.warp(frame)
            transformer.save_matrix(MATRIX_FILE)
            print("✅ Automatische Kalibrierung abgeschlossen.")

            # Debug-Bild erzeugen und speichern
            debug_img = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            for pt in src_pts:
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
            cv2.imwrite("debug_road_mask_with_points.jpg", debug_img)
            print("[INFO] Debug-Bild 'debug_road_mask_with_points.jpg' gespeichert.")
        else:
            print("[WARNUNG] Automatische Kalibrierung fehlgeschlagen.")

    elif mode == "2":
        transformer = PerspectiveTransformerManual(frame)
        src_pts = select_four_points(frame.copy())
        matrix = transformer.compute_perspective_matrix(frame, src_pts)
        warped = transformer.warp(frame)
        transformer.save_matrix(MATRIX_FILE)
        print("✅ Manuelle Kalibrierung abgeschlossen.")
    else:
        print("❌ Ungültige Eingabe!")
        return

    # Speichere das Originalbild und das transformierte Bild
    cv2.imwrite("original_frame.jpg", frame)
    if matrix is not None:
        cv2.imwrite("warped_street.jpg", warped)

if __name__ == "__main__":
    main()
