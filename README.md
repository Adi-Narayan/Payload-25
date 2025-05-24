# Vrishabha: Real-Time Cube Pose Estimation System

**Vrishabha** is a real-time pose estimation and tracking system developed by **thrustMIT**. It is designed as the payload for the sounding rocket **Vayuvega**, for the **International Rocket Engineering Competition 2025 (IREC '25)**. The system demonstrates AI-based inference on the edge using the Hailo-8 AI accelerator and a Raspberry Pi 5.

## 🧠 Overview

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
- **Camera**: Raspberry Pi HQ Camera or Picamera2.
- **AI Inference**: Hailo-8 on Raspberry Pi 5.
- **MPU**: MPU9250 for motion-triggered activation.
- **System Outputs**:
  - Annotated images with bounding boxes and keypoints.
  - Text logs of pose results.
  - CSV logs for accelerometer data.

## 📸 Sample Outputs

| Pose Estimation |
|-----------------|
| ![pose_frame](images/pose_frame.jpg) |

| Raw_Frame | Keypoint Overlay |
|-------------|------------------|
| ![servo_frame](images/servo_frame.jpg) | ![keypoints](images/keypoints.jpg) |

| Raw Frame and Estimated Depth |
|-------------------------------|
| ![raw_frame & Output](images/raw_frame.jpg) |

## Wiring Diagram

| ![raw_frame](images/raw_frame.jpg) |

## 3D Render

| ![raw_frame](images/raw_frame.jpg) |

## Physical Setup

| ![raw_frame](images/raw_frame.jpg) |

## Cansat Placement

| ![raw_frame](images/raw_frame.jpg) |

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

## ⚙️ System Requirements

- Raspberry Pi 5
- Raspberry Pi AI HAT+ (Hailo-8)
- Picamera2 or compatible camera
- MPU9250
- Python 3.9+
- Dependencies:
  - `opencv-python`, `numpy`, `Pillow`, `loguru`, `smbus`, `psutil`, `adafruit_servokit`, `picamera2`, `libcamera`, `hailo_platform`, `diskqueue`, `requests`

## 🔒 Disclaimer

This project is research-grade and optimized for a custom payload setup. It is **not intended for generic use** or deployment without hardware-specific adaptation.

