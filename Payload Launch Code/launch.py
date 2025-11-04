╔══════════════════════════════════════════════════════════════════════╗
║                          PROJECT INFORMATION                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Project      : IREC 2025 Payload — Pose Estimation & Orientation    ║
║  Organization : thrustMIT                                             ║
║  Authors      :                                                      ║
║      • Tejas Patil                                                   ║
║      • Aryan Gupta                                                   ║
║      • Adi Narayn R S                                                ║
║                                                                      ║
║  Description :                                                       ║
║      ▸ Real-time camera-based pose estimation using Hailo AI         ║
║      ▸ Orientation estimation with Madgwick filter + MPU9250 IMU     ║
║      ▸ Triggered launch detection via IMU (acceleration threshold)   ║
║      ▸ Multi-process architecture for camera, pose, servo & logging  ║
║      ▸ Designed for high reliability in IREC 2025 Rocket Payload     ║
║                                                                      ║
║  License      : Proprietary — Internal use for thrustMIT only        ║
║                           ║
╚══════════════════════════════════════════════════════════════════════╝

import time
import os
import csv
import cv2
import sys
import gc
import argparse
import requests
from multiprocessing import Process, Queue, Event, Array, Lock
import numpy as np
from picamera2 import Picamera2
from pathlib import Path
from loguru import logger
from adafruit_servokit import ServoKit
from hailo_platform import HEF
import queue
import threading
import psutil
from PIL import Image
from libcamera import controls, Transform, CameraManager, StreamRole
import multiprocessing
import smbus
import math
import logging
from diskqueue import DiskQueue
import board
import digitalio
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference
from FIGURE import PoseEstPostProcessing, output_data_type2dict, check_process_errors

def timestamp_filename(prefix: str, extension: str = "jpg") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.{extension}"

logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add("pose.log", level="DEBUG", rotation="10 MB")

# Quaternion operations for Madgwick Filter
def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quaternion_conjugate(q):
    """Return the conjugate of a quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)  # Normalize
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

# Madgwick Filter class
class MadgwickFilter:
    def __init__(self, beta=0.033, dt=0.01):
        self.beta = beta  # Filter gain
        self.dt = dt      # Time step (s)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion (identity)

    def update(self, gyro, accel):
        """Update quaternion using gyroscope and accelerometer data."""
        q = self.q / np.linalg.norm(self.q) 
        gyro = np.array(gyro)  # Angular velocity (rad/s)
        accel = np.array(accel) / np.linalg.norm(accel)  # Normalize acceleration

        # Objective function: error between measured and predicted gravity
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accel[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accel[1],
            2*(0.5 - q[1]*q[1] - q[2]*q[2]) - accel[2]
        ])

        # Jacobian of objective function
        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])

        # Gradient
        gradient = J.T @ f
        gradient = gradient / np.linalg.norm(gradient)

        # Gyroscope quaternion derivative
        q_dot = 0.5 * quaternion_multiply(q, [0, *gyro])

        # Combine gyroscope prediction and accelerometer correction
        q_dot = q_dot - self.beta * gradient

        # Integrate to get new quaternion
        self.q = q + q_dot * self.dt
        self.q = self.q / np.linalg.norm(self.q)  # Normalize
        return self.q

def get_orientation_state(q):
    """Determine rocket orientation from quaternion."""
    R = quaternion_to_rotation_matrix(q)
    z_axis = R[:, 2]  # Rocket's z-axis in global frame
    threshold = 0.7  # Cosine of ~45 degrees

    if z_axis[2] > threshold:
        return "Upright"
    elif z_axis[2] < -threshold:
        return "Inverted"
    else:
        # Horizontal: check which side (x or y dominant)
        xy_plane = z_axis[:2]  # x, y components
        if np.abs(xy_plane[0]) > np.abs(xy_plane[1]):
            return "Side (+X)" if xy_plane[0] > 0 else "Side (-X)"
        else:
            return "Side (+Y)" if xy_plane[1] > 0 else "Side (-Y)"

def initialize_mpu():
    PWR_MGMT_1 = 0x6B
    ACCEL_CONFIG = 0x1C
    GYRO_CONFIG = 0x1B
    MPU_ADDRESS = 0x68

    try:
        bus = smbus.SMBus(1)
        bus.write_byte_data(MPU_ADDRESS, PWR_MGMT_1, 0)
        bus.write_byte_data(MPU_ADDRESS, ACCEL_CONFIG, 0x18)  # ±16g
        bus.write_byte_data(MPU_ADDRESS, GYRO_CONFIG, 0x18)   # ±2000 deg/s
        logger.info("MPU9250 initialized successfully")
        return bus
    except Exception as e:
        logger.error(f"Failed to initialize MPU9250: {e}")
        return None

def read_acceleration(bus):
    MPU_ADDRESS = 0x68
    ACCEL_XOUT_H = 0x3B
    
    try:
        accel_data = bus.read_i2c_block_data(MPU_ADDRESS, ACCEL_XOUT_H, 6)
        
        accel_x = (accel_data[0] << 8) | accel_data[1]
        accel_y = (accel_data[2] << 8) | accel_data[3]
        accel_z = (accel_data[4] << 8) | accel_data[5]
        
        if accel_x > 32767:
            accel_x -= 65536
        if accel_y > 32767:
            accel_y -= 65536
        if accel_z > 32767:
            accel_z -= 65536
            
        accel_x_g = accel_x / 2048.0
        accel_y_g = accel_y / 2048.0
        accel_z_g = accel_z / 2048.0
        
        return (accel_x_g, accel_y_g, accel_z_g)
    except Exception as e:
        logger.error(f"Error reading MPU9250: {e}")
        return None

def read_gyroscope(bus):
    """Read gyroscope data from MPU9250 (in rad/s)."""
    MPU_ADDRESS = 0x68
    GYRO_XOUT_H = 0x43

    try:
        gyro_data = bus.read_i2c_block_data(MPU_ADDRESS, GYRO_XOUT_H, 6)
        
        gyro_x = (gyro_data[0] << 8) | gyro_data[1]
        gyro_y = (gyro_data[2] << 8) | gyro_data[3]
        gyro_z = (gyro_data[4] << 8) | gyro_data[5]
        
        if gyro_x > 32767:
            gyro_x -= 65536
        if gyro_y > 32767:
            gyro_y -= 65536
        if gyro_z > 32767:
            gyro_z -= 65536
            
        # Convert to deg/s (FS_SEL=3, ±2000 deg/s, 16.4 LSB/(deg/s))
        gyro_x_deg = gyro_x / 16.4
        gyro_y_deg = gyro_y / 16.4
        gyro_z_deg = gyro_z / 16.4
        
        # Convert to rad/s
        gyro_x_rad = gyro_x_deg * (np.pi / 180)
        gyro_y_rad = gyro_y_deg * (np.pi / 180)
        gyro_z_rad = gyro_z_deg * (np.pi / 180)
        
        return (gyro_x_rad, gyro_y_rad, gyro_z_rad)
    except Exception as e:
        logger.error(f"Error reading MPU9250 gyroscope: {e}")
        return None

def calibrate_mpu(bus, samples=100, sample_interval=0.01):
    logger.info("Calibrating MPU9250...")
    accel_samples = []
    gyro_samples = []
    
    for _ in range(samples):
        accel = read_acceleration(bus)
        gyro = read_gyroscope(bus)
        if accel is not None and gyro is not None:
            accel_samples.append(accel)
            gyro_samples.append(gyro)
        time.sleep(sample_interval)
    
    if not accel_samples or not gyro_samples:
        logger.error("No valid MPU9250 readings during calibration")
        return None, None
        
    avg_accel = np.mean(accel_samples, axis=0)
    avg_gyro = np.mean(gyro_samples, axis=0)
    
    logger.info(f"MPU9250 calibration complete. Accel baseline: {avg_accel}, Gyro baseline: {avg_gyro}")
    return avg_accel, avg_gyro

def orientation_process(trigger_time, stop_event, baseline_accel, baseline_gyro, log_dir="mpu_logs", interval=0.01, bus_lock=None):
    """Process to estimate and log rocket orientation using Madgwick Filter."""
    os.makedirs(log_dir, exist_ok=True)
    filename = f"orientation_{datetime.fromtimestamp(trigger_time).strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    file_path = os.path.join(log_dir, filename)

    # Initialize MPU9250 inside the process
    bus = initialize_mpu()
    if bus is None:
        logger.error("Orientation process failed to initialize MPU9250")
        return

    filter = MadgwickFilter(beta=0.033, dt=interval)

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Quaternion_W", "Quaternion_X", "Quaternion_Y", "Quaternion_Z", "Orientation"])

        while not stop_event.is_set():
            if time.time() - trigger_time > 600:
                break
            try:
                with bus_lock:
                    accel = read_acceleration(bus)
                    gyro = read_gyroscope(bus)
                if accel and gyro:
                    # Apply calibration
                    accel = np.array(accel) - np.array(baseline_accel)
                    gyro = np.array(gyro) - np.array(baseline_gyro)
                    # Update quaternion
                    q = filter.update(gyro, accel)
                    # Get orientation state
                    state = get_orientation_state(q)
                    # Log data
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    writer.writerow([timestamp, f"{q[0]:.4f}", f"{q[1]:.4f}", f"{q[2]:.4f}", f"{q[3]:.4f}", state])
                    csvfile.flush()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Orientation process error: {e}")
                time.sleep(interval)

def mpu_logger(trigger_time, stop_event, baseline_accel, bus_lock, log_dir="mpu_logs", interval=0.01):
    os.makedirs(log_dir, exist_ok=True)
    filename = f"mpu_readings_{datetime.fromtimestamp(trigger_time).strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    file_path = os.path.join(log_dir, filename)

    # Initialize MPU9250 inside the process
    bus = initialize_mpu()
    if bus is None:
        logger.error("MPU logger failed to initialize MPU9250")
        return

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Accel_X_g", "Accel_Y_g", "Accel_Z_g"])

        while not stop_event.is_set():
            if time.time() - trigger_time > 600: 
                break
            try:
                with bus_lock:
                    accel = read_acceleration(bus)
                if accel:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    writer.writerow([timestamp, f"{accel[0]:.4f}", f"{accel[1]:.4f}", f"{accel[2]:.4f}"])
                    csvfile.flush()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"MPU logger error: {e}")
                time.sleep(interval)

class ServoController:
    def __init__(self):
        self.kit = ServoKit(channels=16, address=0x40)
        self.kit.frequency = 125 
        self.kit.servo[0].set_pulse_width_range(500, 2500)
        self.current_angle = 115
        self.last_update = time.time()

    def update(self, target):
        target = np.clip(target, 80, 130)
        current_time = time.time()
        dt = current_time - self.last_update
        max_speed = 450
        max_movement = max_speed * dt
        diff = target - self.current_angle
        movement = np.clip(diff, -max_movement, max_movement)
        self.current_angle += movement
        self.current_angle = np.clip(self.current_angle, 80, 130)
        self.kit.servo[0].angle = self.current_angle
        self.last_update = current_time

def parse_args():
    parser = argparse.ArgumentParser(description="Payload for IREC 25 by thrustMIT")
    parser.add_argument("-n", "--net", default="/home/payload/Hailo-Application-Code-Examples/runtime/python/pose_estimation/payload.hef",
                        help="Path to HEF model file")
    parser.add_argument("--fps", type=int, default=20, help="Camera FPS")
    parser.add_argument("--resolution", type=str, default="640x640", help="Camera resolution")
    parser.add_argument("--tilt-gain", type=float, default=0.015, help="Servo tilt gain factor")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Pose Estimation score threshold")
    parser.add_argument("--simple-display", action="store_true", 
                       help="Use minimal visualization (no heatmaps/history)")
    return parser.parse_args()

def wait_for_mpu_trigger(bus, baseline_accel, threshold=1.5, samples=2, sample_interval=0.01):
    logger.info(f"Waiting for MPU9250 trigger ({threshold}G change, {samples} samples)...")
    
    while True:
        try:
            accel_samples = []
            for _ in range(samples):
                accel = read_acceleration(bus)
                if accel is not None:
                    accel_samples.append(accel)
                time.sleep(sample_interval)
            
            if not accel_samples:
                logger.warning("No valid MPU9250 readings in sample set")
                continue
                
            avg_accel = np.mean(accel_samples, axis=0)
            delta = np.array(avg_accel) - np.array(baseline_accel)
            magnitude = np.sqrt(np.sum(delta**2))
            
            if magnitude >= threshold:
                logger.info(f"MPU9250 triggered! Acceleration change: {magnitude:.2f}g")
                return True
            time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error checking MPU trigger: {e}")
            time.sleep(0.01)
            continue

def delete_old_frames(raw_output_folder, trigger_event, frame_counter, stop_event):
    while not stop_event.is_set() and not trigger_event.is_set():
        if frame_counter[0] >= 20000:
            try:
                files = sorted(raw_output_folder.glob("raw_frame_*.jpg"), key=lambda x: x.stat().st_ctime)
                if len(files) >= 20000:
                    for file in files[:10000]:
                        try:
                            file.unlink()
                            logger.debug(f"Deleted old frame: {file}")
                        except Exception as e:
                            logger.error(f"Error deleting frame {file}: {e}")
                    frame_counter[0] -= 10000
                    logger.info(f"Deleted 10,000 old frames. New frame count: {frame_counter[0]}")
            except Exception as e:
                logger.error(f"Error during frame deletion: {e}")
        time.sleep(10)

def camera_process(shared_array, servo_queue, pose_queue, pose_spill, stop_event, args, trigger_event):
    cam_width, cam_height = map(int, args.resolution.split('x'))
    raw_output_folder = Path("Raw_Camera_Output")
    raw_output_folder.mkdir(exist_ok=True)
    
    picam2 = Picamera2()
    full_sensor_width, full_sensor_height = 3280, 2464
    
    camera_config = picam2.create_preview_configuration(
        main={"size": (cam_width, cam_height), "format": "RGB888"},
        raw={"size": (full_sensor_width, full_sensor_height)}
    )
    picam2.configure(camera_config)
    
    controls_dict = {
        "FrameRate": args.fps,
        "ScalerCrop": (0, 0, full_sensor_width, full_sensor_height),
        "NoiseReductionMode": 0
    }
    
    try:
        picam2.set_controls(controls_dict)
        logger.info(f"Camera controls set: {controls_dict}")
    except RuntimeError as e:
        logger.warning(f"Failed to set some controls: {e}. Proceeding with defaults.")
    
    picam2.start()
    logger.info(f"Camera started with resolution {cam_width}x{cam_height} at {args.fps} FPS")
    
    shared_np_array = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape(cam_height, cam_width, 3)
    
    frame_interval = 1.0 / args.fps
    frame_count = 0
    start_time = time.time()
    frame_counter = [0] 
    
    deletion_thread = threading.Thread(
        target=delete_old_frames,
        args=(raw_output_folder, trigger_event, frame_counter, stop_event),
        daemon=True
    )
    deletion_thread.start()
    
    try:
        while not stop_event.is_set():
            loop_start = time.time()
    
            frame = picam2.capture_array()
            
            if frame.shape != (cam_height, cam_width, 3):
                frame = frame[0:cam_height, 0:cam_width, :]
            
            np.copyto(shared_np_array, frame)
            
            raw_filename = raw_output_folder / timestamp_filename("raw_frame")
            cv2.imwrite(str(raw_filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_counter[0] += 1
            
            timestamp = time.time()
            if trigger_event.is_set():
                try:
                    pose_queue.put((timestamp, args.fps, frame), timeout=0)
                except queue.Full:
                    pose_spill.put((timestamp, args.fps, frame))
                
                try:
                    while True:
                        _ = servo_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    servo_queue.put((timestamp, args.fps), timeout=0)
                except queue.Full:
                    logger.warning("Servo queue full even after clearing")
            
            frame_count += 1
            
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time * 0.95)
            
            if frame_count % 200 == 0:
                logger.info(f"Camera captured and saved {frame_count} frames so far")
            
    finally:
        picam2.stop()
        pose_queue.put(None)
        deletion_thread.join()
        logger.info(f"Camera process completed. Total frames captured and saved: {frame_count}, Pose queue size: {pose_queue.qsize()}")

def servo_update_thread(controller, target_queue, stop_event):
    while not stop_event.is_set():
        try:
            target = target_queue.get_nowait()
            controller.update(target)
        except queue.Empty:
            pass
        time.sleep(0.008)

def servo_pipeline(shared_array, servo_queue, display_queue, target_queue, stop_event, args, trigger_event, trigger_time):
    cam_width, cam_height = map(int, args.resolution.split('x'))
    servo_controller = ServoController()
    CENTER_Y = cam_height / 2

    servo_frames_folder = Path("servo_frames")
    servo_frames_folder.mkdir(exist_ok=True)

    shared_np_array = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape(cam_height, cam_width, 3)

    servo_thread = threading.Thread(target=servo_update_thread, args=(servo_controller, target_queue, stop_event))
    servo_thread.start()

    frame_count = 0
    start_time = time.time()
    average_fps = 0.0

    non_detection_count = 0
    scanning = False
    scan_direction = 1
    scan_speed = 300

    while not stop_event.is_set():
        if not trigger_event.is_set():
            time.sleep(0.01)
            continue

        if time.time() - trigger_time > 600: 
            break

        try:
            timestamp, fps = servo_queue.get(timeout=0)
            frame = shared_np_array.copy()

            small_frame = cv2.resize(shared_np_array, (320, 240), interpolation=cv2.INTER_NEAREST)

            hsv = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            mask = (s >= 192) & (v >= 63)
            mask = mask.astype(np.uint8) * 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display_frame = frame.copy()
            detection_found = False

            if contours:
                biggest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(biggest_contour) > 90:
                    detection_found = True
                    non_detection_count = 0
                    scanning = False
                    x, y, w, h = cv2.boundingRect(biggest_contour)
                    center_y = (y + h / 2.0) * (cam_height / 240)
                    target_angle = servo_controller.current_angle + (center_y - CENTER_Y) * args.tilt_gain
                    try:
                        target_queue.put(target_angle, timeout=0)
                    except queue.Full:
                        pass
                    cv2.rectangle(display_frame, (int(x * cam_width / 320), int(y * cam_height / 240)),
                                (int((x+w) * cam_width / 320), int((y+h) * cam_height / 240)), (0, 255, 0), 2)
            
            if not detection_found:
                non_detection_count += 1
                if non_detection_count > 4:
                    scanning = True

            if scanning:
                current_time = time.time()
                dt = current_time - servo_controller.last_update
                scan_movement = scan_speed * dt * scan_direction
                target_angle = servo_controller.current_angle + scan_movement

                if target_angle >= 130:
                    target_angle = 130
                    scan_direction = -1
                elif target_angle <= 80:
                    target_angle = 80
                    scan_direction = 1

                try:
                    target_queue.put(target_angle, timeout=0)
                except queue.Full:
                    pass

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 30:
                average_fps = frame_count / elapsed_time
                start_time = time.time()

            cv2.putText(display_frame, f"Camera FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Avg FPS (30s): {average_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Angle: {servo_controller.current_angle:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            servo_filename = servo_frames_folder / timestamp_filename("servo")
            cv2.imwrite(str(servo_filename), display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        except queue.Empty:
            continue

    servo_thread.join()

def frame_saving_worker(save_queue, stop_event, trigger_time, output_folder):
    dump_folder = Path("dump")
    dump_folder.mkdir(exist_ok=True)
    
    while not stop_event.is_set() or not save_queue.empty():
        try:
            item = save_queue.get(timeout=0.1)
            if item is None:
                break
                
            output_image, pose_filename, frame_id = item
            
            if time.time() - trigger_time <= 600:
                filename = timestamp_filename("pose")
                cv2.imwrite(str(output_folder / filename), output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            else:
                filename = timestamp_filename("dump")
                cv2.imwrite(str(dump_folder / filename), output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            continue

def pose_estimation_process(pose_queue, pose_spill, save_queue, stop_event, args, trigger_event, trigger_time):
    cam_width, cam_height = map(int, args.resolution.split('x'))
    output_folder = Path("pose_frames")
    output_folder.mkdir(exist_ok=True)
    max_files = 1000000

    hef = HEF(args.net)
    output_type_dict = output_data_type2dict(hef, 'FLOAT32')
    post_processing = PoseEstPostProcessing(
        max_detections=100, 
        score_threshold=args.score_threshold, 
        nms_iou_thresh=0.7,
        regression_length=15, 
        strides=[8, 16, 32]
    )
    
    inference_input_queue = Queue()
    inference_output_queue = Queue()
    
    save_thread = threading.Thread(
        target=frame_saving_worker,
        args=(save_queue, stop_event, trigger_time, output_folder),
        daemon=True
    )
    save_thread.start()
    
    hailo_inference = HailoAsyncInference(args.net, inference_input_queue, inference_output_queue, 
                                         batch_size=1, output_type=output_type_dict)
    
    height, width, _ = hailo_inference.get_input_shape()
    
    hailo_thread = threading.Thread(target=hailo_inference.run)
    hailo_thread.start()

    frame_counter = 0
    inference_count = 0
    start_time = time.time()
    inference_fps = 0.0
    average_fps = 0.0
    class_num = 1
    saved_frames = 0
    received_frames = 0
    
    pending_frames = []

    logger.info("Starting pose estimation process")

    try:
        while not stop_event.is_set() or pending_frames or not pose_queue.empty():
            if not trigger_event.is_set():
                time.sleep(0.01)
                continue
                
            try:
                while not inference_output_queue.empty() and pending_frames:
                    inference_result = inference_output_queue.get_nowait()
                    if inference_result is None:
                        continue
                    
                    frame_data = pending_frames.pop(0)
                    if frame_data is None:
                        continue
                        
                    timestamp, camera_fps, frame, pil_image, frame_id = frame_data
                    
                    _, raw_detections = inference_result
                    
                    results = post_processing.post_process(raw_detections, height, width, class_num)
                    post_processing.log_pose_estimation_data(results, output_folder, frame_id)
                    
                    if args.simple_display:
                        output_image = post_processing.visualize_pose_estimation_result(
                            results, pil_image,
                            detection_threshold=args.score_threshold, 
                            joint_threshold=args.score_threshold,
                            draw_heatmap=False, 
                            draw_history=False
                        )
                    else:
                        output_image = post_processing.visualize_pose_estimation_result(
                            results, pil_image,
                            detection_threshold=args.score_threshold, 
                            joint_threshold=args.score_threshold
                        )
                    
                    current_time = time.time()
                    inference_count += 1
                    
                    elapsed_time = current_time - start_time
                    if elapsed_time >= 30:
                        average_fps = inference_count / elapsed_time
                        inference_count = 0
                        start_time = current_time
                    
                    inference_fps = inference_count / max(elapsed_time, 0.001)
                    
                    print(f"Inference FPS: {inference_fps:.1f}, Avg FPS (30s): {average_fps:.1f}, Frame ID: {frame_id}, Saved frames: {saved_frames}")
                    
                    cv2.putText(output_image, f"Inference FPS: {inference_fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(output_image, f"Avg FPS (30s): {average_fps:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    pose_filename = output_folder / f"pose_frame_{frame_id:04d}.jpg"
                    try:
                        save_queue.put((output_image.copy(), pose_filename, frame_id), timeout=0.1)
                        saved_frames += 1
                    except queue.Full:
                        logger.warning("Save queue full, dropping frame")
                
                try:
                    try:
                        frame_data = pose_queue.get(timeout=0)
                    except queue.Empty:
                        if not pose_spill.empty():
                            frame_data = pose_spill.get()
                        else:
                            time.sleep(0.01)
                            continue

                    if frame_data is None:
                        logger.info("Received end of camera feed signal")
                        continue
                    
                    received_frames += 1
                    timestamp, camera_fps, frame = frame_data
                    frame_id = frame_counter
                    frame_counter += 1
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    processed_image = post_processing.preprocess(pil_image, width, height)
                    
                    try:
                        inference_input_queue.put([processed_image], timeout=0)
                    except queue.Full:
                        logger.warning("Inference input queue full, dropping frame")
                        continue
                    
                    pending_frames.append((timestamp, camera_fps, frame, pil_image, frame_id))
                    
                    if received_frames % 100 == 0:
                        logger.info(f"Received {received_frames} frames, Pending: {len(pending_frames)}, Save queue: {save_queue.qsize()}")
                except queue.Empty:
                    if stop_event.is_set() and pose_queue.empty() and not pending_frames:
                        break
                    continue
                    
            except Exception as e:
                logger.error(f"Inference error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    finally:
        logger.info(f"Finishing pose estimation with {len(pending_frames)} remaining frames")
        
        while pending_frames:
            try:
                inference_result = inference_output_queue.get(timeout=0)
                if inference_result is None:
                    continue
                
                frame_data = pending_frames.pop(0)
                if frame_data is None:
                    continue
                    
                timestamp, camera_fps, frame, pil_image, frame_id = frame_data
                
                _, raw_detections = inference_result
                results = post_processing.post_process(raw_detections, height, width, class_num)
                
                if args.simple_display:
                    output_image = post_processing.visualize_pose_estimation_result(
                        results, pil_image,
                        detection_threshold=args.score_threshold, 
                        joint_threshold=args.score_threshold,
                        draw_heatmap=False, 
                        draw_history=False
                    )
                else:
                    output_image = post_processing.visualize_pose_estimation_result(
                            results, pil_image,
                            detection_threshold=args.score_threshold, 
                            joint_threshold=args.score_threshold
                    )
                
                pose_filename = output_folder / timestamp_filename("pose_frame")
                try:
                    save_queue.put((output_image.copy(), pose_filename, frame_id), timeout=0.1)
                    saved_frames += 1
                except queue.Full:
                    pass
                
                try:
                    output_queue.put((output_image, frame_id), timeout=0.1)
                except queue.Full:
                    pass
            except queue.Empty:
                if pending_frames:
                    logger.warning(f"Timeout waiting for inference, dropping frame {pending_frames[0][4]}")
                    pending_frames.pop(0)
            except Exception as e:
                logger.error(f"Error processing remaining frames: {e}")
                if pending_frames:
                    pending_frames.pop(0)
        
        inference_input_queue.put(None)
        hailo_thread.join()
        
        save_queue.put(None)
        save_thread.join()
        
        output_queue.put(None)
        logger.info(f"Pose estimation process completed. Total frames received: {received_frames}, Total frames saved: {saved_frames}")

def buzz_launch(buzzer):
    while True:
        buzzer.value = True   
        time.sleep(0.5)
        buzzer.value = False  
        time.sleep(0.5)

def main():
    gc.disable()
    buzzer = digitalio.DigitalInOut(board.D16)
    buzzer.direction = digitalio.Direction.OUTPUT
    buzzer.value = True   
    time.sleep(0.5)
    buzzer.value = False  
    time.sleep(0.5)

    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    stop_event = Event()
    trigger_event = Event()
    servo_queue = Queue()
    pose_queue = Queue(maxsize=500)
    pose_spill = DiskQueue(directory="pose_spill")
    servo_display_queue = Queue(maxsize=40)
    target_queue = Queue(maxsize=50)
    save_queue = Queue(maxsize=50)

    cam_width, cam_height = map(int, args.resolution.split('x'))
    frame_shape = (cam_height, cam_width, 3)
    shared_array = Array('B', int(np.prod(frame_shape)))

    camera_core = [0]
    servo_core = [1]
    pose_cores = [2, 3]
    orientation_core = [3]  # Share with pose estimation

    camera_proc = Process(target=camera_process, args=(shared_array, servo_queue, pose_queue, pose_spill, stop_event, args, trigger_event), name="Camera")
    camera_proc.start()
    psutil.Process(camera_proc.pid).cpu_affinity(camera_core)

    time.sleep(1)
    if camera_proc.is_alive():
        logger.info("Camera process started successfully.")
        buzzer.value = True   
        time.sleep(0.5)
        buzzer.value = False  
        time.sleep(0.5)
    else:
        logger.warning("Camera process failed to start.")
        buzzer.value = True   
        time.sleep(3)
        buzzer.value = False  
        time.sleep(0.5)

    bus = initialize_mpu()
    baseline_accel = None
    baseline_gyro = None
    calibration_attempts = 0
    max_attempts = 10
    mpu_calibrated = False
    trigger_time = None

    if bus is not None:
        while calibration_attempts < max_attempts:
            baseline_accel, baseline_gyro = calibrate_mpu(bus, samples=100, sample_interval=0.01)
            if baseline_accel is not None and baseline_gyro is not None:
                mpu_calibrated = True
                logger.info("MPU9250 calibration succeeded.")
                buzzer.value = True   
                time.sleep(0.5)
                buzzer.value = False  
                time.sleep(0.5)
                break
            calibration_attempts += 1
            logger.warning(f"MPU9250 calibration attempt {calibration_attempts} failed. Retrying...")

    if not mpu_calibrated:
        logger.warning("MPU9250 failed to initialize or calibrate.")
        buzzer.value = True   
        time.sleep(3)
        buzzer.value = False  
        time.sleep(0.5)
    else:
        if wait_for_mpu_trigger(bus, baseline_accel, threshold=1.5, samples=2, sample_interval=0.01):
            trigger_event.set()
            trigger_time = time.time()
            buzz_thread = threading.Thread(target=buzz_launch, args=(buzzer,), daemon=True)
            bus_lock = Lock()
            buzz_thread.start()
            mpu_log_thread = threading.Thread(target=mpu_logger, args=(trigger_time, stop_event, baseline_accel, bus_lock, "mpu_logs", 0.01), daemon=True)
            mpu_log_thread.start()
        else:
            logger.warning("MPU9250 trigger failed.")
            buzzer.value = True   
            time.sleep(3)
            buzzer.value = False  
            time.sleep(0.5)

    processes = [
        Process(target=servo_pipeline, args=(shared_array, servo_queue, servo_display_queue, target_queue, stop_event, args, trigger_event, trigger_time), name="Servo"),
        Process(target=pose_estimation_process, args=(pose_queue, pose_spill, save_queue, stop_event, args, trigger_event, trigger_time), name="Pose"),
        Process(target=orientation_process, args=(trigger_time, stop_event, baseline_accel, baseline_gyro, "mpu_logs", 0.01, bus_lock), name="Orientation")
    ]

    for proc in processes:
        proc.start()
        p = psutil.Process(proc.pid)
        if "Servo" in proc.name:
            p.cpu_affinity(servo_core)
        elif "Pose" in proc.name:
            p.cpu_affinity(pose_cores)
        elif "Orientation" in proc.name:
            p.cpu_affinity(orientation_core)

    try:
        for proc in processes:
            proc.join()
        camera_proc.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, finishing processing remaining frames...")
        stop_event.set()
        for proc in processes:
            proc.join()
        camera_proc.join()
    finally:
        stop_event.set()
        check_process_errors(camera_proc, *processes)
        logger.info("All processes terminated.")
    gc.enable()
    gc.collect()

if __name__ == "__main__":
    main()
