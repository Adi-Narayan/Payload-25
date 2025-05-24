#!/usr/bin/env python3

import time
import os
import cv2
import sys
import argparse
from multiprocessing import Process, Queue, Event, Array
import numpy as np
from picamera2 import Picamera2
from pathlib import Path
from loguru import logger
from adafruit_servokit import ServoKit
from hailo_platform import HEF
import queue
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference
from FIGURE_UTILS import PoseEstPostProcessing, output_data_type2dict, check_process_errors

logger.remove()
logger.add(sys.stderr, level="DEBUG")

class ServoController:
    def __init__(self):
        self.kit = ServoKit(channels=16, address=0x40)
        self.kit.servo[0].set_pulse_width_range(500, 2500)
        self.current_angle = 90
        self.last_update = time.time()
        
    def update(self, target):
        current_time = time.time()
        dt = current_time - self.last_update
        max_speed = 180
        max_movement = max_speed * dt
        diff = target - self.current_angle
        movement = np.clip(diff, -max_movement, max_movement)
        self.current_angle += movement
        self.current_angle = np.clip(self.current_angle, 0, 180)
        self.kit.servo[0].angle = self.current_angle
        self.last_update = current_time

def parse_args():
    parser = argparse.ArgumentParser(description="Dual pipeline with pose estimation display")
    parser.add_argument("-n", "--net", default="/home/thrustmit/Hailo-Application-Code-Examples/runtime/python/pose_estimation/pyramid.hef",
                        help="Path to HEF model file")
    parser.add_argument("--fps", type=int, default=60, help="Camera FPS")
    parser.add_argument("--resolution", type=str, default="640x480", help="Camera resolution")
    parser.add_argument("--tilt-gain", type=float, default=0.007, help="Servo tilt gain factor")
    parser.add_argument("--score-threshold", type=float, default=0.1, help="Pose Estimation score threshold")
    return parser.parse_args()

def camera_process(shared_frame, frame_shape, stop_event, args):
    cam_width, cam_height = map(int, args.resolution.split('x'))
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (cam_width, cam_height), "format": "RGB888"})
    picam2.configure(config)
    picam2.set_controls({"FrameRate": args.fps})
    logger.debug("Starting camera...")
    picam2.start()
    
    try:
        while not stop_event.is_set():
            frame = picam2.capture_array()
            with shared_frame.get_lock():
                shared_array = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8)
                shared_array[:] = frame.flatten()
            time.sleep(1 / args.fps)
    finally:
        picam2.stop()
        logger.debug("Camera stopped.")

def servo_pipeline(shared_frame, frame_shape, servo_queue, stop_event, args):
    cam_width, cam_height = map(int, args.resolution.split('x'))
    servo_controller = ServoController()
    CENTER_Y = cam_height / 2
    last_frame_time = time.time()

    logger.debug("Starting servo pipeline...")
    while not stop_event.is_set():
        with shared_frame.get_lock():
            shared_array = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8)
            frame = shared_array.reshape(frame_shape)

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 192, 63]), np.array([179, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display_frame = frame.copy()
        center_y = None
        if contours:
            biggest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest_contour) > 200:
                x, y, w, h = cv2.boundingRect(biggest_contour)
                center_y = y + (h / 2.0)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if center_y is not None:
            error_y = center_y - CENTER_Y
            target_angle = servo_controller.current_angle + (error_y * args.tilt_gain)
            servo_controller.update(target_angle)

        current_time = time.time()
        fps = 1 / (current_time - last_frame_time)
        last_frame_time = current_time
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Angle: {servo_controller.current_angle:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            servo_queue.put_nowait(display_frame)
        except queue.Full:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

def pose_estimation_process(shared_frame, frame_shape, output_queue, stop_event, args):
    from PIL import Image
    cam_width, cam_height = map(int, args.resolution.split('x'))
    output_folder = Path("Pose_Estimation_Output")
    output_folder.mkdir(exist_ok=True)
    
    hef = HEF(args.net)
    output_type_dict = output_data_type2dict(hef, 'FLOAT32')
    post_processing = PoseEstPostProcessing(
        max_detections=100, score_threshold=args.score_threshold, nms_iou_thresh=0.7,
        regression_length=15, strides=[8, 16, 32]
    )
    hailo_inference = HailoAsyncInference(args.net, Queue(), Queue(), batch_size=1, output_type=output_type_dict)
    height, width, _ = hailo_inference.get_input_shape()
    
    frame_counter = 0
    hailo_thread = threading.Thread(target=hailo_inference.run)
    hailo_thread.start()
    logger.debug("Starting pose estimation process...")
    
    try:
        while not stop_event.is_set():
            with shared_frame.get_lock():
                shared_array = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8)
                frame = shared_array.reshape(frame_shape)
            
            # Save raw frame (convert to BGR for cv2.imwrite)
            raw_filename = output_folder / f"raw_frame_{frame_counter:04d}.jpg"
            cv2.imwrite(str(raw_filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            output_image = frame.copy()
            detection_info = None
            try:
                image = Image.fromarray(frame)
                processed_image = post_processing.preprocess(image, width, height)
                input_batch = [np.array(processed_image)]
                hailo_inference.input_queue.put(input_batch)
                
                result = hailo_inference.output_queue.get(timeout=5)
                if result is not None:
                    _, raw_detections = result
                    results = post_processing.post_process(raw_detections, height, width, 1)
                    
                    # Extract detection info for display
                    bboxes, scores, keypoints, joint_scores = (
                        results['bboxes'], results['scores'], results['keypoints'], results['joint_scores']
                    )
                    logger.debug(f"Scores shape: {scores.shape}, Scores: {scores}")
                    if scores.size > 0 and scores[0].size > 0:
                        max_score_idx = np.argmax(scores[0])
                        logger.debug(f"Max score index: {max_score_idx}, Score: {scores[0][max_score_idx]}")
                        # Lowered confidence threshold to debug
                        if scores[0][max_score_idx] > 0.1:  # Lowered from 0.5 to 0.1
                            box = bboxes[0][max_score_idx]
                            keypoint = keypoints[0][max_score_idx].reshape(5, 2)
                            score = float(scores[0][max_score_idx].item())
                            xmin, ymin, xmax, ymax = [int(x) for x in box]
                            
                            # Calculate detection info
                            green_face_points = np.array([keypoint[idx] for idx in [0, 1, 2]], dtype=np.int32)
                            green_face_center = np.mean(green_face_points, axis=0)
                            apex = keypoint[0]
                            dx = float(apex[0] - green_face_center[0])
                            dy = float(apex[1] - green_face_center[1])
                            angle = float(np.degrees(np.arctan2(dy, dx)).item())
                            
                            detection_info = {
                                'score': score,
                                'width': int(xmax - xmin),
                                'height': int(ymax - ymin),
                                'area': int((xmax - xmin) * (ymax - ymin)),
                                'perimeter': int(2 * ((xmax - xmin) + (ymax - ymin))),
                                'angle': angle
                            }
                            logger.debug(f"Detection info: {detection_info}")
                    
                    # Visualize with a lower detection threshold
                    output_image = post_processing.visualize_pose_estimation_result(
                        results, image, detection_threshold=0.2, joint_threshold=0.1
                    )
                else:
                    logger.warning("No inference result received, using raw frame")
            except queue.Empty:
                logger.warning("Inference timeout, saving raw frame with no annotations")
            except Exception as e:
                logger.error(f"Inference error: {e}")
            
            # Add detection info to the image if available (already handled in visualize_pose_estimation_result)
            
            # Save processed frame (already in BGR from visualize_pose_estimation_result)
            pose_filename = output_folder / f"pose_frame_{frame_counter:04d}.jpg"
            cv2.imwrite(str(pose_filename), output_image)
            
            try:
                output_queue.put_nowait((output_image, frame_counter))
            except queue.Full:
                pass
            
            frame_counter += 1
    
    finally:
        hailo_inference.input_queue.put(None)
        hailo_thread.join()
        output_queue.put(None)
        logger.debug(f"Saved {frame_counter} frames to {output_folder}")

def servo_display(servo_queue, stop_event):
    logger.debug("Starting servo display...")
    cv2.namedWindow("Red Color Tracking", cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        try:
            frame = servo_queue.get(timeout=5)
            # Convert to BGR for display
            cv2.imshow("Red Color Tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
        except queue.Empty:
            continue
    
    cv2.destroyWindow("Red Color Tracking")
    logger.debug("Servo display stopped.")

def pose_estimation_display(output_queue, stop_event):
    logger.debug("Starting pose estimation display...")
    cv2.namedWindow("Pyramid Skeleton", cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        try:
            item = output_queue.get(timeout=5)
            if item is None:
                break
            output_image, frame_counter = item
            # Already in BGR from visualize_pose_estimation_result
            cv2.imshow("Pyramid Skeleton", output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
        except queue.Empty:
            continue
    
    cv2.destroyWindow("Pyramid Skeleton")
    logger.debug("Pose estimation display stopped.")

def main():
    args = parse_args()
    stop_event = Event()
    cam_width, cam_height = map(int, args.resolution.split('x'))
    frame_shape = (cam_height, cam_width, 3)
    shared_frame = Array('B', cam_height * cam_width * 3)
    output_queue = Queue(maxsize=10)
    servo_queue = Queue(maxsize=10)

    processes = [
        Process(target=camera_process, args=(shared_frame, frame_shape, stop_event, args), name="Camera"),
        Process(target=servo_pipeline, args=(shared_frame, frame_shape, servo_queue, stop_event, args), name="Servo"),
        Process(target=servo_display, args=(servo_queue, stop_event), name="ServoDisplay"),
        Process(target=pose_estimation_process, args=(shared_frame, frame_shape, output_queue, stop_event, args), name="Pose"),
        Process(target=pose_estimation_display, args=(output_queue, stop_event), name="PoseDisplay")
    ]

    # Start servo-related processes first
    servo_processes = [p for p in processes if "Servo" in p.name]
    for proc in servo_processes:
        proc.start()

    # Then start pose estimation processes
    pose_processes = [p for p in processes if "Pose" in p.name]
    for proc in pose_processes:
        proc.start()

    # Start camera process last
    camera_proc = next(p for p in processes if p.name == "Camera")
    camera_proc.start()

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        stop_event.set()
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        for proc in processes:
            proc.join()
        check_process_errors(*processes)
        logger.info("All processes terminated.")

if __name__ == "__main__":
    main()