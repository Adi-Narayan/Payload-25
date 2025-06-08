import cv2
import numpy as np
import json

# Load calibration results
with open("calibration_results.json") as f:
    calib = json.load(f)

camera_matrix = np.array(calib["camera_matrix"])
dist_coeffs = np.array(calib["dist_coeff"])

# Load your test image (replace 'test.jpg' with your image path)
img = cv2.imread(r"/home/payload/Hailo-Application-Code-Examples/runtime/python/pose_estimation/Raw_Camera_Output/raw_frame_2025-06-06_23-26-15-206166.jpg")
if img is None:
    print("Error: Could not load image.")
    exit(1)

h, w = img.shape[:2]

# Get optimal new camera matrix for undistortion
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))

# Undistort
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)

# Crop the image based on roi
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

# Show images side by side
combined = cv2.hconcat([img, cv2.resize(undistorted_img, (img.shape[1], img.shape[0]))])
cv2.imshow("Original (left) vs Undistorted (right)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
