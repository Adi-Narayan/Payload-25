import cv2
import numpy as np
from picamera2 import Picamera2
import time
import json
from pathlib import Path

# --- Calibration settings ---
CHECKERBOARD = (9, 6)  # inner corners per a chessboard row and column
SQUARE_SIZE = 25  # mm

# Prepare object points: (0,0,0), (1,0,0), (2,0,0) ... multiplied by square size
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# --- Initialize Picamera2 ---
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)  # warm up camera

print("Press SPACE to capture calibration frame with checkerboard visible.")
print("Press ESC to finish and run calibration.")

captured_frames = 0

while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Refine corner locations
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        cv2.drawChessboardCorners(frame_bgr, CHECKERBOARD, corners2, ret)
        cv2.putText(frame_bgr, "Checkerboard detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Calibration", frame_bgr)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to exit and calibrate
        print("Calibration starting...")
        break
    elif key == 32 and ret:  # SPACE to capture frame if checkerboard detected
        imgpoints.append(corners2)
        objpoints.append(objp)
        captured_frames += 1
        print(f"Captured frame #{captured_frames}")

picam2.stop()
cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("Not enough frames captured for calibration. Capture at least 5 good frames.")
    exit(1)

# Run calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"Calibration done. RMS error: {ret}")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

# Save results
output = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "rms_error": ret
}
Path("calibration_results.json").write_text(json.dumps(output, indent=4))
print("Calibration results saved to calibration_results.json")
