#!/usr/bin/env python3

import time
import os
import cv2
import sys
import argparse
import threading
import multiprocessing as mp
from multiprocessing import Process, Event, Queue
from pathlib import Path
from loguru import logger
from PIL import Image
from hailo_platform import HEF
from picamera2 import Picamera2
from servo_pose_utils import (output_data_type2dict,
                                   check_process_errors, PoseEstPostProcessing)
import numpy as np
from adafruit_servokit import ServoKit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from utils import HailoAsyncInference


# ServoController class to manage servo motor movement
class ServoController:
    def __init__(self):
        self.kit = ServoKit(channels=16, address=0x40)
        self.kit.servo[0].set_pulse_width_range(500, 2500)
        self.current_angle = 90  # Start at neutral position
        self.target_angle = 90
        self.last_update = time.time()
        
    def update(self, target):
        """Smoothly update servo angle based on target."""
        self.target_angle = target
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Smooth movement with maximum speed limit
        max_speed = 180  # degrees per second
        max_movement = max_speed * dt
        
        diff = self.target_angle - self.current_angle
        movement = np.clip(diff, -max_movement, max_movement)
        
        self.current_angle += movement
        self.current_angle = np.clip(self.current_angle, 0, 180)
        
        self.kit.servo[0].angle = self.current_angle
        self.last_update = current_time

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pose estimation with servo control for Raspberry Pi 5 with Hailo accelerator"
    )
    parser.add_argument(
        "-n", "--net",
        default="/home/thrustmit/Hailo-Application-Code-Examples/runtime/python/pose_estimation/pyramid.hef",
        help="Path to the HEF model file"
    )
    parser.add_argument(
        "-cn", "--class_num", type=int, default=1,
        help="Number of classes (default: 1)"
    )
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Process every N-th frame (default: 2)"
    )
    parser.add_argument(
        "--max-queue", type=int, default=90,
        help="Max input queue size (default: 2)"
    )
    parser.add_argument(
        "--simple-display", action="store_true",
        help="Use minimal visualization (no heatmaps/history)"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.1,
        help="Score threshold for detections (default: 0.1)"
    )
    parser.add_argument(
        "--resolution", type=str, default="320x240",
        help="Camera resolution, e.g., 640x480 (default: 640x480)"
    )
    return parser.parse_args()

def preprocess_input(
    input_queue: Queue,
    width: int,
    height: int,
    post_processing: PoseEstPostProcessing,
    stop_event: Event,
    args: argparse.Namespace
) -> None:
    """Capture and preprocess camera frames."""
    picam2 = Picamera2()
    cam_width, cam_height = map(int, args.resolution.split('x'))
    config = picam2.create_still_configuration(main={"size": (cam_width, cam_height)})
    picam2.configure(config)
    picam2.start()
    frame_counter = 0
    try:
        while not stop_event.is_set():
            frame = picam2.capture_array()
            frame_counter += 1

            if frame_counter % args.skip_frames != 0:
                continue

            if input_queue.qsize() >= args.max_queue:
                try:
                    input_queue.get_nowait()
                except:
                    pass

            image = Image.fromarray(frame)
            processed_image = post_processing.preprocess(image, width, height)
            input_queue.put([processed_image])

    finally:
        picam2.stop()
        input_queue.put(None)

def postprocess_output(
    output_queue: Queue,
    display_queue: Queue,
    width: int,
    height: int,
    class_num: int,
    post_processing: PoseEstPostProcessing,
    stop_event: Event,
    args: argparse.Namespace
) -> None:
    """Post-process inference results and extract apex position."""
    while not stop_event.is_set():
        result = output_queue.get()
        if result is None:
            display_queue.put((None, None))  # Signal end
            break

        processed_image, raw_detections = result
        results = post_processing.post_process(raw_detections, height, width, class_num)

        # Visualize results based on display mode
        if args.simple_display:
            output_image = post_processing.visualize_pose_estimation_result(
                results, processed_image,
                draw_heatmap=False, draw_history=False
            )
        else:
            output_image = post_processing.visualize_pose_estimation_result(
                results, processed_image
            )

        # Extract apex y-coordinate from highest-scoring detection
        apex_y = None
        if results['scores'].size > 0:
            max_score_idx = np.argmax(results['scores'][0])
            if results['scores'][0][max_score_idx] > 0.5:  # Confidence threshold
                apex = results['keypoints'][0][max_score_idx][0]  # Keypoint 0 is apex
                apex_y = apex[1]  # y-coordinate

        if display_queue.qsize() >= 2:
            try:
                display_queue.get_nowait()
            except:
                pass
        display_queue.put((output_image, apex_y))

def infer(
    net_path: str,
    class_num: int,
    data_type_dict: dict,
    post_processing: 'PoseEstPostProcessing',
    stop_event: Event,
    args: 'argparse.Namespace'
) -> None:
    """Run inference, save frames, and control servo."""
    # Initialize queues
    input_queue = Queue(maxsize=args.max_queue)
    output_queue = Queue()
    display_queue = Queue(maxsize=2)

    # Set up inference and processing processes
    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue,
        batch_size=1, output_type=data_type_dict
    )
    height, width, _ = hailo_inference.get_input_shape()

    preprocess = Process(
        target=preprocess_input,
        args=(input_queue, width, height, post_processing, stop_event, args)
    )
    postprocess = Process(
        target=postprocess_output,
        args=(output_queue, display_queue, width, height, class_num,
              post_processing, stop_event, args)
    )

    preprocess.start()
    postprocess.start()
    hailo_thread = threading.Thread(target=hailo_inference.run)

    try:
        hailo_thread.start()
        last_frame_time = time.time()

        # Initialize frame saving and servo control
        output_folder = Path("Frames_output")
        output_folder.mkdir(exist_ok=True)
        frame_counter = 0
        servo_controller = ServoController()
        TILT_GAIN = 0.07  # Servo adjustment gain

        # Main processing loop
        while True:
            item = display_queue.get()
            if item[0] is None:  # Check for end signal
                break
            output_image, apex_y = item

            # Save visualized frame
            frame_filename = output_folder / f"frame_{frame_counter:04d}.jpg"
            cv2.imwrite(str(frame_filename), output_image)
            frame_counter += 1

            # Control servo if apex detected
            if apex_y is not None:
                center_y = height / 2  # Model input height
                error_y = apex_y - center_y
                target_angle = servo_controller.current_angle + (error_y * TILT_GAIN)
                servo_controller.update(target_angle)

            # Display frame with FPS
            cv2.imshow('Pose Estimation', output_image)
            current_time = time.time()
            fps = 1 / (current_time - last_frame_time)
            last_frame_time = current_time
            cv2.putText(output_image, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    finally:
        # Cleanup
        stop_event.set()
        cv2.destroyAllWindows()
        print(f"Saved {frame_counter} frames to {output_folder}")
        hailo_thread.join()
        preprocess.join()
        postprocess.join()
        check_process_errors(preprocess, postprocess)

def main() -> None:
    """Main entry point."""
    args = parse_args()
    stop_event = Event()
    output_type_dict = output_data_type2dict(HEF(args.net), 'FLOAT32')
    post_processing = PoseEstPostProcessing(
        max_detections=100,
        score_threshold=args.score_threshold,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )
    infer(args.net, args.class_num, output_type_dict,
          post_processing, stop_event, args)

if __name__ == "__main__":
    main()