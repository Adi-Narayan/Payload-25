import numpy as np
import cv2
import torch
import open3d as o3d
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path

@dataclass
class Detection:
    """Structure to hold YOLOv8 detection data"""
    frame_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    vertices_2d: List[Tuple[float, float]] = None
    pose_3d: np.ndarray = None

class VrishabhaCubeAnalyzer:
    """Main class for analyzing Vrishabha cube tracking data"""
    
    def __init__(self, data_path: str, camera_intrinsics: Dict = None):
        self.data_path = Path(data_path)
        self.detections = []
        self.camera_intrinsics = camera_intrinsics or self.default_camera_params()
        self.cube_size = 0.05  # Assume 5cm cube
        
    def default_camera_params(self):
        """Default camera intrinsic parameters - adjust based on your setup"""
        return {
            'fx': 800, 'fy': 800,  # focal lengths
            'cx': 320, 'cy': 240,  # principal point
            'width': 640, 'height': 480
        }
    
    def parse_detection_file(self, detection_file: str):
        """Parse the YOLOv11 detection text file"""
        detections = []
        current_frame = -1
        
        with open(detection_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('--- Frame'):
                    current_frame = int(line.split()[2])
                elif line.startswith('Detection'):
                    # Parse: Detection 0: Confidence 0.8078, BBox (264.36, 184.25, 459.46, 446.12)
                    parts = line.split(':')
                    conf_bbox = parts[1].strip()
                    
                    # Extract confidence
                    conf_str = conf_bbox.split(',')[0].replace('Confidence', '').strip()
                    confidence = float(conf_str)
                    
                    # Extract bbox
                    bbox_str = conf_bbox.split('BBox')[1].strip()
                    bbox_coords = bbox_str.replace('(', '').replace(')', '').split(',')
                    bbox = tuple(float(x.strip()) for x in bbox_coords)
                    
                    detection = Detection(
                        frame_id=current_frame,
                        confidence=confidence,
                        bbox=bbox
                    )
                    detections.append(detection)
        
        self.detections = detections
        return detections
    
    def load_frame_images(self, raw_frames_dir: str, pose_frames_dir: str):
        """Load corresponding raw and pose estimation frames"""
        self.raw_frames = {}
        self.pose_frames = {}
        
        raw_path = Path(raw_frames_dir)
        pose_path = Path(pose_frames_dir)
        
        for detection in self.detections:
            frame_id = detection.frame_id
            
            # Load raw frame
            raw_file = raw_path / f"frame_{frame_id:04d}.jpg"
            if raw_file.exists():
                self.raw_frames[frame_id] = cv2.imread(str(raw_file))
            
            # Load pose frame
            pose_file = pose_path / f"frame_{frame_id:04d}.jpg"
            if pose_file.exists():
                self.pose_frames[frame_id] = cv2.imread(str(pose_file))
    
    def estimate_3d_pose_from_bbox(self, detection: Detection):
        """Estimate 3D pose from 2D bounding box using PnP algorithm"""
        x1, y1, x2, y2 = detection.bbox
        
        # Define 3D cube vertices (assuming cube at origin)
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
        
        # Approximate 2D points from bounding box corners
        # This is simplified - in reality, you'd extract from pose estimation
        cube_2d_points = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2],  # visible face corners
            [x1+10, y1+10], [x2-10, y1+10], [x2-10, y2-10], [x1+10, y2-10]  # approximated back face
        ], dtype=np.float32)
        
        # Camera matrix
        camera_matrix = np.array([
            [self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']],
            [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Solve PnP to get rotation and translation
        dist_coeffs = np.zeros((4,1))  # Assuming no distortion
        success, rvec, tvec = cv2.solvePnP(
            cube_3d_points[:4],  # Use only visible face for simplicity
            cube_2d_points[:4],
            camera_matrix,
            dist_coeffs
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Create 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = tvec.flatten()
            
            detection.pose_3d = transform_matrix
            
        return detection
    
    def analyze_motion_patterns(self):
        """Analyze cube motion patterns and spring dynamics"""
        positions = []
        timestamps = []
        confidences = []
        
        for detection in self.detections:
            if detection.pose_3d is not None:
                # Extract position from transformation matrix
                position = detection.pose_3d[:3, 3]
                positions.append(position)
                timestamps.append(detection.frame_id)
                confidences.append(detection.confidence)
        
        positions = np.array(positions)
        
        # Calculate velocities and accelerations
        if len(positions) > 2:
            velocities = np.diff(positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            
            return {
                'positions': positions,
                'velocities': velocities,
                'accelerations': accelerations,
                'timestamps': timestamps,
                'confidences': confidences
            }
        
        return None
    
    def create_3d_visualization(self, motion_data):
        """Create interactive 3D visualization of cube trajectory"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = motion_data['positions']
        confidences = motion_data['confidences']
        
        # Color-code trajectory by confidence
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=confidences, cmap='viridis', s=50, alpha=0.7
        )
        
        # Plot trajectory line
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'r-', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('Vrishabha Cube 3D Trajectory')
        
        # Add colorbar for confidence
        plt.colorbar(scatter, label='Detection Confidence')
        
        return fig
    
    def simulate_spring_physics(self, motion_data):
        """Simulate spring dynamics based on observed motion"""
        positions = motion_data['positions']
        
        if len(positions) < 3:
            return None
        
        # Estimate spring parameters using least squares
        # Assuming simple harmonic motion: F = -kx
        accelerations = motion_data['accelerations']
        displacements = positions[2:] - np.mean(positions, axis=0)  # Center around mean
        
        # Fit spring constant for each axis
        spring_constants = []
        for axis in range(3):
            if len(displacements) > 0 and len(accelerations) > 0:
                # k = -F/x = -ma/x (assuming unit mass)
                k = -np.mean(accelerations[:, axis] / (displacements[:, axis] + 1e-6))
                spring_constants.append(abs(k))
        
        return {
            'spring_constants': spring_constants,
            'natural_frequency': np.sqrt(np.mean(spring_constants)),
            'damping_estimate': self.estimate_damping(positions)
        }
    
    def estimate_damping(self, positions):
        """Estimate damping coefficient from position decay"""
        # Simple amplitude decay analysis
        amplitudes = np.linalg.norm(positions - np.mean(positions, axis=0), axis=1)
        
        if len(amplitudes) > 10:
            # Fit exponential decay
            t = np.arange(len(amplitudes))
            try:
                coeffs = np.polyfit(t, np.log(amplitudes + 1e-6), 1)
                damping = -coeffs[0]  # Decay rate
                return max(0, damping)
            except:
                return 0.1  # Default damping
        
        return 0.1
    
    def generate_point_cloud(self, frame_id: int):
        """Generate point cloud from depth estimation (placeholder)"""
        # This would integrate with your Depth-Anything-V2 results
        # For now, create a synthetic point cloud around the cube
        
        detection = next((d for d in self.detections if d.frame_id == frame_id), None)
        if not detection or detection.pose_3d is None:
            return None
        
        # Create point cloud around estimated cube position
        center = detection.pose_3d[:3, 3]
        
        # Generate random points around the cube
        n_points = 1000
        points = np.random.normal(center, 0.02, (n_points, 3))
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.random.rand(n_points, 3))
        
        return pcd
    
    def export_results(self, output_dir: str):
        """Export analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export detections with 3D poses
        results = []
        for detection in self.detections:
            result = {
                'frame_id': detection.frame_id,
                'confidence': detection.confidence,
                'bbox': detection.bbox,
                'pose_3d': detection.pose_3d.tolist() if detection.pose_3d is not None else None
            }
            results.append(result)
        
        with open(output_path / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {output_path}")

# Example usage
def main():
    # Initialize analyzer
    analyzer = VrishabhaCubeAnalyzer("./vrishabha_data")
    
    # Parse detection file
    analyzer.parse_detection_file("pose_estimation_log.txt")
    
    # Load frame images
    analyzer.load_frame_images(r"D:\Home\Desktop\Payload_2025_Research\depth_any\Raw_Camera\Raw_Camera_Output", r"D:\Home\Desktop\Payload_2025_Research\depth_any\pose_frame\pose_frames")
    
    # Estimate 3D poses for all detections
    for detection in analyzer.detections:
        analyzer.estimate_3d_pose_from_bbox(detection)
    
    # Analyze motion patterns
    motion_data = analyzer.analyze_motion_patterns()
    
    if motion_data:
        # Create visualization
        fig = analyzer.create_3d_visualization(motion_data)
        plt.show()
        
        # Simulate physics
        physics_data = analyzer.simulate_spring_physics(motion_data)
        print("Spring Analysis Results:", physics_data)
        
        # Generate point clouds for first few frames
        for i in range(min(5, len(analyzer.detections))):
            pcd = analyzer.generate_point_cloud(analyzer.detections[i].frame_id)
            if pcd:
                o3d.visualization.draw_geometries([pcd])
    
    # Export results
    analyzer.export_results("./output")

if __name__ == "__main__":
    main()