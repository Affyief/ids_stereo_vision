#!/usr/bin/env python3

"""
Interactive Single Camera Calibration Tool

- Opens a live view from your camera (IDS U3-3680XCP-C, Sony IMX178 sensor).
- Press "c" to capture calibration images (stored to ./calib_captures/).
- When target number of captures is reached, calibrates automatically.
- Displays results overlayed on a calibration image, with full parameter names and explanations.

Camera: IDS U3-3680XCP-C
Sensor: Sony IMX178 CMOS, 1/1.8″
Active pixel size: 2592 x 1944 px
Pixel size: 2.4 μm × 2.4 μm
Active sensor area: 6.22 mm × 4.67 mm
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path

# ---- Camera and sensor details (precise, per datasheet) ----
CAMERA_MODEL = "IDS U3-3680XCP-C"
SENSOR_TYPE = "Sony IMX178 CMOS (1/1.8″)"
PIXEL_SIZE_UM = 2.4  # micrometers, square pixels
IMAGE_WIDTH = 2592
IMAGE_HEIGHT = 1944
SENSOR_WIDTH_MM = IMAGE_WIDTH * PIXEL_SIZE_UM / 1000  # 6.22 mm
SENSOR_HEIGHT_MM = IMAGE_HEIGHT * PIXEL_SIZE_UM / 1000  # 4.67 mm

# ---- Calibration pattern and session configuration ----
SAVE_DIR = "calib_captures"
NUM_IMAGES_NEEDED = 25
CHECKERBOARD_ROWS = 5 # Number of inner corners in rows (chessboard squares - 1)
CHECKERBOARD_COLS = 7 # Number of inner corners in columns (chessboard squares - 1)
SQUARE_SIZE_MM = 15.0 # Actual side length of each calibration checkerboard square

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def capture_images():
    ensure_dir(SAVE_DIR)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        exit(1)
    
    print(f"\nCamera ready. Press 'c' to capture an image when the checkerboard is visible.")
    print(f"Capture at various angles/distances. Need {NUM_IMAGES_NEEDED} images total.")
    print("Press 'q' to quit early.")

    img_count = 0
    while img_count < NUM_IMAGES_NEEDED:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        display = frame.copy()
        status = f"Captured: {img_count}/{NUM_IMAGES_NEEDED}. Press 'c' to capture."
        cv2.putText(display, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Camera Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            filename = os.path.join(SAVE_DIR, f"calib_{img_count+1:02d}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            img_count += 1
            time.sleep(0.3)
        elif key == ord('q'):
            print("Quitting early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished image capture ({img_count} images).")
    return img_count

def find_checkerboard_corners(images_dir, checkerboard_size, square_size_mm, show_corners=False):
    """Returns objpoints (world), imgpoints (pixels), image_size"""
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[1], 0:checkerboard_size[0]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []
    img_points = []
    image_size = None
    images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not images:
        print(f"No calibration images found in {images_dir}")
        return [], [], None

    for fname in images:
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray,
            (checkerboard_size[1], checkerboard_size[0]), # (cols, rows)
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            if show_corners:
                display = img.copy()
                cv2.drawChessboardCorners(display, (checkerboard_size[1], checkerboard_size[0]), corners_refined, True)
                cv2.imshow('Found Corners', display)
                cv2.waitKey(300)
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])
        else:
            print(f"No corners found in {fname}")
    if show_corners:
        cv2.destroyAllWindows()
    print(f"Found corners in {len(obj_points)} out of {len(images)} images.")
    return obj_points, img_points, image_size

def calibrate_camera(obj_points, img_points, image_size):
    print("Calibrating camera using OpenCV ...")
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None
    )
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(obj_points)
    return ret, K, D, mean_error

def display_results_over_image(img_path, K, D, mean_error):
    img = cv2.imread(img_path)
    overlay = img.copy()
    params = [
        ("Focal Length (fx, fy)", f"{K[0,0]:.2f}, {K[1,1]:.2f} px", "How 'zoomed in' camera is (in pixels)"),
        ("Principal Point (cx, cy)", f"{K[0,2]:.2f}, {K[1,2]:.2f} px", "Where optical axis hits image (usually center)"),
        ("Radial Distortion (k1, k2, k3)", f"{D[0][0]:.4f}, {D[0][1]:.4f}, {D[0][4]:.4f}", "Lens barrel/pincushion distortion"),
        ("Tangential Distortion (p1, p2)", f"{D[0][2]:.4f}, {D[0][3]:.4f}", "Tilted lens or misalignment effects"),
        ("Reprojection error", f"{mean_error:.4f} px", "Avg pixel error mapping model to detected points"),
    ]
    x, y = 50, 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, f"Camera: {CAMERA_MODEL} / {SENSOR_TYPE}", (x, y-20), font, 0.7, (255,255,0), 2)
    for name, val, expl in params:
        txt = f"{name}: {val} ({expl})"
        cv2.putText(overlay, txt, (x, y), font, 0.7, (255,255,255), 2)
        y += 30
    cv2.imshow('Calibration Results', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("\n==== SINGLE CAMERA CALIBRATION ====")
    print(f"Camera model: {CAMERA_MODEL}")
    print(f"Sensor: {SENSOR_TYPE}")
    print(f"Sensor active area: {SENSOR_WIDTH_MM:.2f} mm (W) × {SENSOR_HEIGHT_MM:.2f} mm (H)")
    print(f"Checkerboard: {CHECKERBOARD_ROWS} rows × {CHECKERBOARD_COLS} cols, square {SQUARE_SIZE_MM}mm")
    print(f"Image resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print("Images will be saved to:", SAVE_DIR)

    n_captured = capture_images()
    if n_captured < 8:
        print("Too few images, calibration aborted.")
        return

    obj_points, img_points, image_size = find_checkerboard_corners(
        SAVE_DIR,
        (CHECKERBOARD_ROWS, CHECKERBOARD_COLS),
        SQUARE_SIZE_MM,
        show_corners=True
    )
    if len(obj_points) < 8:
        print("ERROR: Checkerboard not found in enough images! Try again.")
        return

    ret, K, D, mean_error = calibrate_camera(obj_points, img_points, image_size)

    print("\n======= CALIBRATION RESULTS =======")
    print("Camera intrinsic matrix (K):")
    print(K)
    print("Distortion coefficients (D):")
    print(D)
    print(f"Average reprojection error: {mean_error:.4f} px")
    print("Full parameter explanations are shown overlayed on image.")

    first_img = sorted([
        f for f in os.listdir(SAVE_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])[0]
    display_results_over_image(os.path.join(SAVE_DIR, first_img), K, D, mean_error)

if __name__ == '__main__':
    main()
