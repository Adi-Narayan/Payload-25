"""
1.Run yolov8_pose_estimation.py to get detections.txt in output_detections
2.Run parse_detections.py in output_detections to get results.json
3.Run reconstruct3d.py, which uses results.json and calibration_results.json to generate analysis_results.json and analysis_summary.json in "output" folder
4.Run web_server.py for animation, which uses these 2 output json files. Open debug_vrishabha.html using localhost address provided on prompt on running web_server.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple
import torch

class YOLOv8PoseEstimator:
    """Class to perform pose estimation on video frames using YOLOv8 and output detection data with cube skeleton overlay."""
    
    def __init__(self, model_path: str, output_dir: str, confidence_threshold: float = 0.4, joint_threshold: float = 0.6):
        """Initialize the pose estimator with YOLOv8 model."""
        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.overlaid_dir = self.output_dir / "pose_frames"
        self.overlaid_dir.mkdir(exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.joint_threshold = joint_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Define colors for cube faces (BGR format for OpenCV)
        self.face_colors = [
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (0, 0, 255),   # Blue
            (255, 255, 0), # Yellow
            (255, 0, 255), # Magenta
            (0, 255, 255)  # Cyan
        ]
        
        # Define cube faces based on vertex indices (from FIGURE.py)
        self.faces = [
            [0, 1, 3, 2],  # Back face
            [4, 5, 7, 6],  # Front face
            [0, 4, 5, 1],  # Left face
            [2, 6, 7, 3],  # Right face
            [1, 5, 7, 3],  # Top face
            [0, 4, 6, 2]   # Bottom face
        ]
        
        # Define joint pairs for cube skeleton (from FIGURE.py)
        self.joint_pairs = [
            [1, 3], [5, 1], [7, 3], [2, 3], [1, 0], [0, 4], [2, 6], [5, 7], [4, 6], [4, 5], [0, 2], [6, 7]
        ]
        
    def draw_cube_on_image(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """Draw cube skeleton on the frame based on pose estimation results, mimicking FIGURE.py logic."""
        frame_copy = frame.copy()
        keypoints = detection['keypoints_2d']
        num_keypoints = len(keypoints)
        
        if num_keypoints != 8:
            print(f"Warning: Expected 8 keypoints, but got {num_keypoints}. Skipping visualization.")
            return frame_copy
        
        # Extract keypoint coordinates and confidence
        kpts = [(kp[0], kp[1]) for kp in keypoints]
        kpts_conf = [kp[2] for kp in keypoints]
        
        # Track visible keypoints (those belonging to fully visible faces)
        visible_keypoints = set()
        
        # Draw colored faces if all keypoints are visible
        for i, face in enumerate(self.faces):
            all_visible = all(kpts_conf[idx] >= self.joint_threshold for idx in face)
            if all_visible:
                pts = np.array([kpts[idx] for idx in face], dtype=np.int32)
                cv2.fillPoly(frame_copy, [pts], self.face_colors[i])
                visible_keypoints.update(face)
        
        # Draw visible keypoints and their labels
        for idx, (x, y) in enumerate(kpts):
            if idx in visible_keypoints:
                cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 255), -1)  # Yellow keypoints
                cv2.putText(frame_copy, str(idx), (int(x) + 8, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White labels
        
        # Draw joint lines only if both keypoints are visible
        for joint0, joint1 in self.joint_pairs:
            if joint0 in visible_keypoints and joint1 in visible_keypoints:
                pt1 = (int(kpts[joint0][0]), int(kpts[joint0][1]))
                pt2 = (int(kpts[joint1][0]), int(kpts[joint1][1]))
                cv2.line(frame_copy, pt1, pt2, (255, 0, 255), 2)  # Magenta lines
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, detection['bbox'])
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Red bounding box
        
        # Add detection info
        cube_label = f"Cube: {detection['confidence']:.2f}"
        cv2.putText(frame_copy, cube_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
        
        return frame_copy
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> List[dict]:
        """Process a single frame, return detection data, and save overlaid image."""
        results = self.model(frame, conf=self.confidence_threshold, task='pose')
        detections = []
        
        for result in results:
            if result.boxes is None or result.keypoints is None:
                print(f"Frame {frame_id}: No boxes or keypoints detected")
                continue
                
            for box, keypoints in zip(result.boxes, result.keypoints):
                bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                confidence = float(box.conf.cpu().numpy()[0]) if box.conf is not None else 0.0
                
                kpts = keypoints.xy.cpu().numpy()
                kpts_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else np.ones(len(kpts))
                
                if len(kpts.shape) == 3:
                    kpts = kpts[0] if kpts.shape[0] == 1 else kpts
                elif len(kpts.shape) == 1 or len(kpts) == 0:
                    print(f"Frame {frame_id}: Invalid keypoints, skipping detection")
                    continue
                
                if kpts_conf.size == 0 or len(kpts_conf.shape) > 1:
                    kpts_conf = np.ones(len(kpts))
                
                keypoints_data = []
                for i in range(len(kpts)):
                    x = float(kpts[i][0]) if np.isscalar(kpts[i][0]) else float(kpts[i][0][0]) if len(kpts[i][0]) > 0 else 0.0
                    y = float(kpts[i][1]) if np.isscalar(kpts[i][1]) else float(kpts[i][1][0]) if len(kpts[i][1]) > 0 else 0.0
                    conf = float(kpts_conf[i]) if i < len(kpts_conf) else 1.0
                    keypoints_data.append((x, y, conf))
                
                orientation_2d = None
                if len(kpts) >= 2:
                    dx = float(kpts[1][0] - kpts[0][0])
                    dy = float(kpts[1][1] - kpts[0][1])
                    orientation_2d = float(np.arctan2(dy, dx) * 180 / np.pi)
                
                pose_confidence = float(np.mean(kpts_conf)) if len(kpts_conf) > 0 else 0.0
                
                detection = {
                    'frame_id': frame_id,
                    'confidence': confidence,
                    'bbox': bbox,
                    'keypoints_2d': keypoints_data,
                    'orientation_2d': orientation_2d,
                    'pose_confidence': pose_confidence
                }
                detections.append(detection)
                
                # Draw and save overlaid image
                overlaid_frame = self.draw_cube_on_image(frame, detection)
                output_path = self.overlaid_dir / f"pose_frame_{frame_id:04d}.jpg"
                cv2.imwrite(str(output_path), overlaid_frame)
                print(f"Saved overlaid image for frame {frame_id} to {output_path}")
                
        return detections
    
    def write_detections_to_file(self, detections: List[dict], output_file: str):
        """Write detection data to a text file."""
        with open(self.output_dir / output_file, 'w') as f:
            for detection in detections:
                f.write(f"--- Frame {detection['frame_id']}\n")
                bbox_str = f"({detection['bbox'][0]:.2f}, {detection['bbox'][1]:.2f}, " \
                          f"{detection['bbox'][2]:.2f}, {detection['bbox'][3]:.2f})"
                keypoints_str = "[" + ", ".join(f"({k[0]:.2f},{k[1]:.2f},{k[2]:.2f})" 
                                               for k in detection['keypoints_2d']) + "]"
                orientation = detection['orientation_2d'] if detection['orientation_2d'] is not None else 0.0
                f.write(f"Detection 0: Confidence {detection['confidence']:.4f}, "
                       f"BBox {bbox_str}, Keypoints {keypoints_str}, "
                       f"Orientation {orientation:.1f}, "
                       f"PoseConf {detection['pose_confidence']:.2f}\n")
    
    def process_video(self, video_path: str, output_file: str = "detections.txt"):
        """Process a video file and save detection results."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        all_detections = []
        frame_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = self.process_frame(frame, frame_id)
            all_detections.extend(detections)
            frame_id += 1
            
            if frame_id % 100 == 0:
                print(f"Processed {frame_id} frames")
                
        cap.release()
        self.write_detections_to_file(all_detections, output_file)
        print(f"Detections saved to {self.output_dir / output_file}")
        return all_detections
    
    def process_image_folder(self, image_folder: str, output_file: str = "detections.txt"):
        """Process a folder of images and save detection results."""
        image_folder = Path(image_folder)
        image_extensions = ['.jpg', '.jpeg', '.png']
        all_detections = []
        
        image_files = sorted([f for f in image_folder.glob('*') 
                            if f.suffix.lower() in image_extensions])
        
        for frame_id, image_path in enumerate(image_files):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Could not read image: {image_path}")
                continue
                
            detections = self.process_frame(frame, frame_id)
            all_detections.extend(detections)
            
            if frame_id % 100 == 0:
                print(f"Processed {frame_id + 1} images")
                
        self.write_detections_to_file(all_detections, output_file)
        print(f"Detections saved to {self.output_dir / output_file}")
        return all_detections

def main():
    model_path = r"D:\Home\Desktop\Payload_2025_Research\depth_any\payload2025-1.pt"
    input_path = r"D:\Home\Desktop\Payload_2025_Research\depth_any\aryan\Raw_Camera_Output"
    output_dir = "output_detections"
    output_file = "detections.txt"
    
    estimator = YOLOv8PoseEstimator(model_path, output_dir, confidence_threshold=0.4, joint_threshold=0.6)
    
    if Path(input_path).is_file():
        estimator.process_video(input_path, output_file)
    else:
        estimator.process_image_folder(input_path, output_file)

if __name__ == "__main__":
    main()
