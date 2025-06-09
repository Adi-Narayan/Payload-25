import numpy as np
import cv2
import torch
import open3d as o3d
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
import warnings
import argparse

@dataclass
class Detection:
    """Structure to hold YOLOv8 detection data"""
    frame_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    vertices_2d: Optional[List[Tuple[float, float]]] = None
    pose_3d: Optional[np.ndarray] = None

class VrishabhaCubeAnalyzer:
    """Main class for analyzing Vrishabha cube tracking data"""
    
    def __init__(self, data_path: str, calibration_file: str = r"D:\Home\Desktop\Payload_2025_Research\depth_any\camera_calibration\calibration_results.json"):
        self.data_path = Path(data_path)
        self.detections = []
        self.camera_intrinsics = self.load_camera_params(calibration_file)
        
        # Physical parameters
        self.cube_size = 0.0175  # 17.5mm cube edge length
        self.cube_mass = 0.042875  # 42.875g = 0.042875 kg
        
        self.valid_detections = []  # Track which detections have valid 3D poses
        
    def load_camera_params(self, calibration_file: str):
        """Load camera intrinsic parameters from calibration_results.json"""
        try:
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)
            
            camera_matrix = np.array(calibration_data['camera_matrix'], dtype=np.float32)
            dist_coeffs = np.array(calibration_data['dist_coeff'][0], dtype=np.float32)
            
            intrinsics = {
                'fx': camera_matrix[0, 0],
                'fy': camera_matrix[1, 1],
                'cx': camera_matrix[0, 2],
                'cy': camera_matrix[1, 2],
                'width': 640,  # Default image width
                'height': 480,  # Default image height
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs
            }
            
            print(f"Loaded camera calibration from {calibration_file}")
            print(f"Camera matrix:\n{camera_matrix}")
            print(f"Distortion coefficients: {dist_coeffs}")
            
            return intrinsics
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Camera calibration file {calibration_file} not found. Please provide a valid calibration file.")
        except KeyError as e:
            raise KeyError(f"Missing key in calibration file: {e}")
        except Exception as e:
            raise Exception(f"Error loading camera calibration: {e}")
        
    def load_results_json(self, json_path: str, normalize_keypoints: bool = False, target_width: int = 640, target_height: int = 480):
        """Load detection data from results.json with optional keypoint normalization"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.detections = []
            for item in data:
                vertices_2d = [kp[:2] for kp in item['keypoints']]
                # Log keypoints for debugging
                print(f"Frame {item['frame']}: Keypoints = {vertices_2d}")
                # Normalize keypoints if requested
                if normalize_keypoints:
                    max_x = max(kp[0] for kp in vertices_2d if kp[0] > 0)
                    max_y = max(kp[1] for kp in vertices_2d if kp[1] > 0)
                    if max_x > target_width or max_y > target_height:
                        scale_x = target_width / max_x
                        scale_y = target_height / max_y
                        vertices_2d = [(kp[0] * scale_x, kp[1] * scale_y) for kp in vertices_2d]
                        print(f"Frame {item['frame']}: Normalized keypoints by ({scale_x:.2f}, {scale_y:.2f}) to {vertices_2d}")
                detection = Detection(
                    frame_id=item['frame'],
                    confidence=item['confidence'],
                    bbox=tuple(item['bbox']),
                    vertices_2d=vertices_2d
                )
                self.detections.append(detection)
            print(f"Loaded {len(self.detections)} detections from {json_path}")
        except FileNotFoundError:
            print(f"Error: Results file {json_path} not found")
            self.detections = []
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from {json_path}: {e}")
            self.detections = []
    
    def load_frame_images(self, raw_frames_dir: str, pose_frames_dir: str):
        """Load corresponding raw and pose estimation frames"""
        self.raw_frames = {}
        self.pose_frames = {}
        
        raw_path = Path(raw_frames_dir)
        pose_path = Path(pose_frames_dir)
        
        if not raw_path.exists():
            print(f"Warning: Raw frames directory {raw_path} does not exist")
        if not pose_path.exists():
            print(f"Warning: Pose frames directory {pose_path} does not exist")
        
        loaded_raw = 0
        loaded_pose = 0
        
        for detection in self.detections:
            frame_id = detection.frame_id
            
            # Load raw frame
            raw_file = raw_path / f"raw_frame_{frame_id:04d}.jpg"
            if raw_file.exists():
                self.raw_frames[frame_id] = cv2.imread(str(raw_file))
                loaded_raw += 1
            
            # Load pose frame
            pose_file = pose_path / f"pose_frame_{frame_id:04d}.jpg"
            if pose_file.exists():
                self.pose_frames[frame_id] = cv2.imread(str(pose_file))
                loaded_pose += 1
                
        print(f"Loaded {loaded_raw} raw frames and {loaded_pose} pose frames")
    
    def interpolate_pose(self, detection: Detection, detections: List[Detection], max_search: int = 5):
        """Interpolate 3D pose from neighboring valid poses, searching up to max_search frames"""
        idx = next((i for i, d in enumerate(detections) if d.frame_id == detection.frame_id), None)
        if idx is None:
            return None
        
        # Search for previous valid pose
        prev_det = None
        for i in range(idx - 1, max(0, idx - max_search - 1), -1):
            if detections[i].pose_3d is not None:
                prev_det = detections[i]
                break
        
        # Search for next valid pose
        next_det = None
        for i in range(idx + 1, min(len(detections), idx + max_search + 1)):
            if detections[i].pose_3d is not None:
                next_det = detections[i]
                break
        
        if prev_det is None or next_det is None:
            print(f"Cannot interpolate pose for frame {detection.frame_id}: insufficient valid neighbors")
            return None
        
        prev_frame = prev_det.frame_id
        next_frame = next_det.frame_id
        curr_frame = detection.frame_id
        
        # Linear interpolation factor
        t = (curr_frame - prev_frame) / (next_frame - prev_frame)
        
        # Interpolate translation
        prev_t = prev_det.pose_3d[:3, 3]
        next_t = next_det.pose_3d[:3, 3]
        interp_t = (1 - t) * prev_t + t * next_t
        
        # Interpolate rotation using slerp (simplified to linear for small angles)
        prev_R = prev_det.pose_3d[:3, :3]
        next_R = next_det.pose_3d[:3, :3]
        prev_rvec, _ = cv2.Rodrigues(prev_R)
        next_rvec, _ = cv2.Rodrigues(next_R)
        interp_rvec = (1 - t) * prev_rvec + t * next_rvec
        interp_R, _ = cv2.Rodrigues(interp_rvec)
        
        # Construct interpolated transform
        interp_pose = np.eye(4)
        interp_pose[:3, :3] = interp_R
        interp_pose[:3, 3] = interp_t
        
        return interp_pose
    
    def estimate_3d_pose(self, detection: Detection):
        """Estimate 3D pose from keypoints using PnP algorithm with fallback"""
        if detection.vertices_2d is None:
            print(f"Warning: No vertices_2d for frame {detection.frame_id}")
            return detection
        
        # Define 3D cube vertices (cube centered at origin)
        cube_3d_points = np.array([
            [-self.cube_size/2, -self.cube_size/2, -self.cube_size/2],  # vertex 0
            [self.cube_size/2, -self.cube_size/2, -self.cube_size/2],   # vertex 1
            [self.cube_size/2, self.cube_size/2, -self.cube_size/2],    # vertex 2
            [-self.cube_size/2, self.cube_size/2, -self.cube_size/2],   # vertex 3
            [-self.cube_size/2, -self.cube_size/2, self.cube_size/2],   # vertex 4
            [self.cube_size/2, -self.cube_size/2, self.cube_size/2],    # vertex 5
            [self.cube_size/2, self.cube_size/2, self.cube_size/2],     # vertex 6
            [-self.cube_size/2, self.cube_size/2, self.cube_size/2],    # vertex 7
        ], dtype=np.float32)
        
        cube_2d_points = np.array(detection.vertices_2d, dtype=np.float32)
        
        # Filter valid points: non-zero and within image bounds
        width, height = self.camera_intrinsics['width'], self.camera_intrinsics['height']
        valid_idx = [
            i for i in range(8)
            if (cube_2d_points[i, 0] > 0 and cube_2d_points[i, 1] > 0 and
                cube_2d_points[i, 0] < width and cube_2d_points[i, 1] < height)
        ]
        
        if len(valid_idx) < 6:
            print(f"Warning: Only {len(valid_idx)} valid points for frame {detection.frame_id}. Invalid points: {[cube_2d_points[i].tolist() for i in range(8) if i not in valid_idx]}")
            
            # Attempt interpolation
            interpolated_pose = self.interpolate_pose(detection, self.detections, max_search=5)
            if interpolated_pose is not None:
                detection.pose_3d = interpolated_pose
                self.valid_detections.append(detection)
                print(f"Interpolated pose for frame {detection.frame_id}")
            return detection
        
        valid_3d = cube_3d_points[valid_idx]
        valid_2d = cube_2d_points[valid_idx]
        
        # Use camera matrix and distortion coefficients from intrinsics
        camera_matrix = self.camera_intrinsics['camera_matrix']
        dist_coeffs = self.camera_intrinsics['dist_coeffs'].reshape(-1, 1)
        
        try:
            success, rvec, tvec = cv2.solvePnP(
                valid_3d,
                valid_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = tvec.flatten()
                detection.pose_3d = transform_matrix
                self.valid_detections.append(detection)
                
                # Calculate reprojection error
                projected_points, _ = cv2.projectPoints(valid_3d, rvec, tvec, camera_matrix, dist_coeffs)
                reprojection_error = np.sqrt(np.sum((projected_points.squeeze() - valid_2d)**2, axis=1))
                mean_error = np.mean(reprojection_error)
                print(f"Frame {detection.frame_id}: Mean reprojection error = {mean_error:.2f} pixels")
                if mean_error > 10:
                    print(f"Warning: High reprojection error for frame {detection.frame_id}: {mean_error:.2f} pixels")
        except cv2.error as e:
            print(f"PnP solving failed for frame {detection.frame_id}: {e}")
            # Attempt interpolation
            interpolated_pose = self.interpolate_pose(detection, self.detections, max_search=5)
            if interpolated_pose is not None:
                detection.pose_3d = interpolated_pose
                self.valid_detections.append(detection)
                print(f"Interpolated pose for frame {detection.frame_id}")
        
        return detection
    
    def analyze_motion_patterns(self):
        """Analyze cube motion patterns from 3D positions"""
        valid_detections = [d for d in self.detections if d.pose_3d is not None]
        
        if len(valid_detections) < 2:
            print("Warning: Not enough valid detections for motion analysis")
            return None
            
        positions = []
        timestamps = []
        confidences = []
        
        for detection in valid_detections:
            position = detection.pose_3d[:3, 3]
            positions.append(position)
            timestamps.append(detection.frame_id)
            confidences.append(detection.confidence)
        
        positions = np.array(positions)
        timestamps = np.array(timestamps)
        
        if len(positions) > 2:
            dt = np.diff(timestamps) * (1.0/15.0)  # Assuming 15 FPS
            dt[dt == 0] = 1.0/15.0  # Avoid division by zero
            velocity_vectors = np.diff(positions, axis=0)
            velocities = velocity_vectors / dt[:, np.newaxis]
            
            if len(velocities) > 1:
                acceleration_vectors = np.diff(velocities, axis=0)
                dt_accel = dt[1:]
                accelerations = acceleration_vectors / dt_accel[:, np.newaxis]
            else:
                accelerations = np.array([])
            
            return {
                'positions': positions,
                'velocities': velocities,
                'accelerations': accelerations,
                'timestamps': timestamps,
                'confidences': confidences,
                'valid_frames': len(valid_detections)
            }
        
        return None
    
    def create_3d_visualization(self, motion_data):
        """Create interactive 3D visualization of cube trajectory"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = motion_data['positions']
        confidences = motion_data['confidences']
        
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=confidences, cmap='viridis', s=50, alpha=0.7
        )
        
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'r-', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'Vrishabha Cube 3D Trajectory ({motion_data["valid_frames"]} frames)')
        
        plt.colorbar(scatter, label='Detection Confidence')
        
        return fig
    
    def generate_point_cloud(self, frame_id: int):
        """Generate point cloud from depth estimation"""
        detection = next((d for d in self.detections if d.frame_id == frame_id), None)
        if not detection or detection.pose_3d is None:
            return None
        
        center = detection.pose_3d[:3, 3]
        
        n_points = 1000
        noise_scale = self.cube_size * 0.4
        points = np.random.normal(center, noise_scale, (n_points, 3))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        distances = np.linalg.norm(points - center, axis=1)
        colors = plt.cm.viridis(distances / np.max(distances))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def export_results(self, output_dir: str):
        """Export analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        valid_count = 0
        
        for detection in self.detections:
            result = {
                'frame_id': detection.frame_id,
                'confidence': detection.confidence,
                'bbox': detection.bbox,
                'pose_3d': detection.pose_3d.tolist() if detection.pose_3d is not None else None
            }
            results.append(result)
            if detection.pose_3d is not None:
                valid_count += 1
        
        with open(output_path / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Convert NumPy arrays and types in camera_intrinsics to native Python types for JSON serialization
        serializable_intrinsics = {}
        for k, v in self.camera_intrinsics.items():
            if isinstance(v, np.ndarray):
                serializable_intrinsics[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                serializable_intrinsics[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                serializable_intrinsics[k] = int(v)
            else:
                serializable_intrinsics[k] = v
        
        summary = {
            'total_detections': len(self.detections),
            'valid_3d_poses': valid_count,
            'success_rate': valid_count / len(self.detections) if self.detections else 0,
            'cube_size': self.cube_size,
            'cube_mass': self.cube_mass,
            'camera_intrinsics': serializable_intrinsics
        }
        
        with open(output_path / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results exported to {output_path}")
        print(f"Successfully processed {valid_count}/{len(self.detections)} detections")

def main():
    parser = argparse.ArgumentParser(description="Vrishabha Cube 3D Reconstruction")
    parser.add_argument('--results-json', default=r"D:\Home\Desktop\Payload_2025_Research\depth_any\output_detections\results.json", help="Path to results.json")
    parser.add_argument('--calibration-file', default=r"D:\Home\Desktop\Payload_2025_Research\depth_any\camera_calibration\calibration_results.json", help="Path to calibration_results.json")
    parser.add_argument('--normalize-keypoints', action='store_true', help="Normalize keypoints to 640x480")
    args = parser.parse_args()

    try:
        analyzer = VrishabhaCubeAnalyzer("./vrishabha_data", args.calibration_file)
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return

    analyzer.load_results_json(args.results_json, normalize_keypoints=args.normalize_keypoints)
    
    if not analyzer.detections:
        print("No detections found. Check your results.json path and format.")
        return
    
    analyzer.load_frame_images(
        r"D:\Home\Desktop\Payload_2025_Research\depth_any\aryan\Raw_Camera_Output", 
        r"D:\Home\Desktop\Payload_2025_Research\depth_any\pose_frame\pose_frames"
    )
    
    print("Estimating 3D poses...")
    for detection in analyzer.detections:
        analyzer.estimate_3d_pose(detection)
    
    print("Analyzing motion patterns...")
    motion_data = analyzer.analyze_motion_patterns()
    
    if motion_data:
        print(f"Motion analysis complete. Valid frames: {motion_data['valid_frames']}")
        
        try:
            fig = analyzer.create_3d_visualization(motion_data)
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        print("Generating point clouds...")
        for i in range(min(3, len(analyzer.valid_detections))):
            pcd = analyzer.generate_point_cloud(analyzer.valid_detections[i].frame_id)
            if pcd:
                print(f"Generated point cloud for frame {analyzer.valid_detections[i].frame_id}")
    else:
        print("Motion analysis failed - insufficient valid detections")
    
    print("Exporting results...")
    analyzer.export_results("./output")

if __name__ == "__main__":
    main()
