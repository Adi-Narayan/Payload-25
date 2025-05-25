# Vrishabha: Real-Time Cube Pose Estimation System

**Vrishabha** is a real-time pose estimation and tracking system developed by **thrustMIT**. It is designed as the payload for the sounding rocket **Vayuvega**, for the **International Rocket Engineering Competition 2025 (IREC '25)**. The system demonstrates AI-based inference on the edge using the Hailo-8 AI accelerator and a Raspberry Pi 5.

![Mission Patch](https://github.com/user-attachments/assets/3fd351c0-2ae0-47ac-92b4-4fe3c0cd5a11)

## 🧠 Overview

Vayuvega’s primary objective is to reach an altitude of 30,000 feet, in accordance with the IREC (Intercollegiate Rocket Engineering Competition) guidelines, while ensuring a safe and successful recovery of the rocket and payload. During ascent, Vayuvega is expected to achieve a maximum velocity of approximately Mach 1.71, marking a critical performance milestone and demonstrating the structural integrity and aerodynamic efficiency of the launch vehicle under transonic and supersonic conditions. The successful accomplishment of these objectives is crucial for achieving the project’s goals of competitiveness and operational safety. The secondary objective of Vayuvega’s mission is to test the functioning of the scientific payload Vrishabha, which aims to demonstrate real-time 3D tracking, pose estimation, and point cloud generation under high vibrations and G-forces. A camera module mounted on a servo tracks a cube attached to a spring within the CanSat. The system operates within the limited field of view, validating the performance of AI-driven
perception in extreme dynamic conditions.

At the core of the experiment is a cube mounted on a spring, simulating a target undergoing unpredictable movement. A camera, mounted on a dual-shaft 35 kg·cm torque servo motor, tracks the cube using colour-based detection within a limited and dynamically shifting field of view. The servo tilts the camera to follow the cube’s movement, enabling near real-time tracking despite motion blur, frame drops, and rotational skew.

A pose estimation model based on YOLOv8 and trained on a custom dataset predicts the positions of the cube’s vertices, thereby determining its orientation. The camera’s raw frames are processed using the Depth-Anything-V2 architecture to generate point clouds, providing detailed 3D spatial information of the object. All computation processes run on a Raspberry Pi 5, coupled with the Raspberry Pi AI HAT+ (Hailo-8), acting as an AI accelerator.

The mission aims to prove the reliability of AI-based visual inference in fast-changing, resource-constrained environments. Its success supports future developments in space robotics, autonomous spacecraft operations, and defence, where intelligent systems must perform under minimal sensor feedback and high uncertainty.

## 🚀 Features

- Real-time 3D pose estimation of a red cube using YOLOv8 keypoints.
- Servo-based camera tracking with adaptive tilt control.
- MPU9250 trigger detection to activate the system on significant acceleration changes.
- Multi-process architecture separating camera capture, pose estimation, servo control, and data logging.
- Visual outputs including bounding boxes, annotated cube keypoints, and visible face rendering.
- Data logging for frame metadata and IMU readings.

## 🔧 Architecture Summary

The project includes the following modules:

- `launch.py`: Manages the process lifecycle including camera feed, pose inference, servo updates, and MPU trigger.
- `FIGURE.py`: Core module for pose estimation and annotated visualization.
- `diskqueue.py`: Persistent queue to offload memory usage during high frame capture.
- `pose_frames/`, `servo_frames/`, `Raw_Camera_Output/`: Directories where output frames are stored.
- `pose_estimation_log.txt`, `mpu_logs/`: Pose and IMU logs for analysis.

## 🧪 Experimental Setup

- **Target**: Red-colored cube mounted on a spring.
- **Camera**: Raspberry Pi Camera or Picamera2 compatible cameras.
- **AI Inference**: Hailo-8 on Raspberry Pi 5.
- **MPU**: MPU9250 for motion-triggered activation.
- **System Outputs**:
  - Annotated images with bounding boxes and keypoints.
  - Text logs of pose results.
  - CSV logs for accelerometer data.

## 📸 Sample Outputs

| Pose Estimation |
|-----------------|
 ![pose_frame](https://github.com/user-attachments/assets/1946a489-69f5-4a85-8ee1-c11eea98c84a) 

| Tracked Output |
|----------------|
 ![servo_frame](https://github.com/user-attachments/assets/2358ae15-7c9d-419b-8aa0-53683a1dd4c7) 

| Raw Frame and Estimated Depth |
|-------------------------------|
 ![raw_frame & Output](https://github.com/user-attachments/assets/e9b340b1-2bee-4004-a3de-10c1799529a2) 

## Wiring Diagram

 ![Wiring Diagram](https://github.com/user-attachments/assets/aec363d2-a506-4a8e-a4e3-a069ddc714b6)

## 3D Render

 ![CAD view](https://github.com/user-attachments/assets/fa540125-c2a5-44df-ac28-6f1469461d34) 

## Physical Setup

 ![Fully prepped test setup](https://github.com/user-attachments/assets/7656c69d-550d-4cc4-92f2-6b88b856798d) 

## Cansat Placement

 ![Nose Cone](https://github.com/user-attachments/assets/de2847de-08e3-4a44-bbd4-8b6a4d300d8b) 

## 📂 Directory Structure

```
.
├── launch.py
├── FIGURE.py
├── diskqueue.py
├── pose_frames/
├── Raw_Camera_Output/
├── mpu_logs/
├── servo_frames/
├── dump/
└── pose_estimation_log.txt
```

## 📌 Project Context

Vrishabha leverages the Hailo-8 AI accelerator to perform efficient 3D pose estimation of a cube-shaped object using a Raspberry Pi camera. The system detects and tracks a cube by identifying its 8 keypoints and constructing its 3D structure, accounting for occlusions by visualizing only fully visible faces.

Integrated with an MPU9250 inertial measurement unit, Vrishabha triggers data collection upon detecting significant acceleration changes, logging pose data and accelerometer readings. A servo-controlled camera dynamically adjusts to track red-colored objects, ensuring precise alignment in high-speed flight scenarios. The system captures, processes, and saves images with annotated bounding boxes, keypoints, and cube faces, delivering robust performance in dynamic environments.

## ⚙️ System Requirements - Hardware and Software

- Raspberry Pi 5
- Raspberry Pi AI HAT+ (Hailo-8)
- Picamera compatible cameras
- MPU9250
- PCA9685
- I2C hub
- Servo motor with camera mount (custom)
- Lights (depends on environment)
- Buzzer
- Python 3.9+
- Dependencies:
  - `opencv-python`, `numpy`, `Pillow`, `loguru`, `smbus`, `psutil`, `adafruit_servokit`, `picamera2`, `libcamera`, `hailo_platform`, `requests`

## 🔒 Disclaimer

This project is research-grade and optimized for a custom payload setup. It is **not intended for generic use** or deployment without hardware-specific adaptation.

