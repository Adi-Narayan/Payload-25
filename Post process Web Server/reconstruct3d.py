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
    
    def __init__(self, data_path: str, camera_intrinsics: Dict = None):
        self.data_path = Path(data_path)
        self.detections = []
        self.camera_intrinsics = camera_intrinsics or self.default_camera_params()
        
        # Physical parameters
        self.cube_size = 0.0175  # 17.5mm
        self.cube_mass = 0.042875  # 42.875g = 0.042875 kg
        self.cube_weight = self.cube_mass * 9.81  # Weight in Newtons (0.421N)
        
        # Conical spring parameters (corrected)
        self.spring_d1 = 0.01051    # small diameter (m) - 9mm
        self.spring_d2 = 0.04051    # large diameter (m) - 39mm
        self.wire_diameter = 0.0015  # 1.5mm wire diameter
        self.spring_coils = 3     # active coils
        self.shear_modulus = 81.37e9  # Pa (steel wire)

        # Expected spring stiffness from document
        self.expected_k_document = 0.77103  # N/m from document calculation
        # Expected deflection from document: f = 19.08 × 10⁻³ m
        self.expected_deflection = 0.01908  # m
        
        # Calculate conical spring constant
        self.expected_k = self.calculate_conical_spring_constant()
        self.expected_freq = self.expected_natural_frequency()


        

        
        # Adjust filtering threshold based on expected deflection
        self.displacement_threshold = self.expected_deflection * 0.1  # 10% of expected deflection
        
        self.valid_detections = []  # Track which detections have valid 3D poses
        
    def default_camera_params(self):
        """Default camera intrinsic parameters - scale corrected based on observed vs expected displacement"""
        # Scale factor: observed 49mm vs expected 130mm ≈ 0.38
        return {
            'fx': 800 * 0.38, 'fy': 800 * 0.38,  # Scale-corrected focal lengths
            'cx': 320, 'cy': 240,
            'width': 640, 'height': 480
        }
        
    def calculate_conical_spring_constant(self):
        """Calculate spring constant for conical spring using proper formula"""
        d = self.wire_diameter
        D1 = self.spring_d1  
        D2 = self.spring_d2
        n = self.spring_coils
        G = self.shear_modulus
        
        # Conical spring formula (simplified approximation)
        # k = (G * d^4) / (8 * n) * (1/D1^3 + 1/D2^3)
        k = (G * d**4) / (8 * n) * (1/D1**3 + 1/D2**3)
        
        print(f"Conical Spring Parameters (Updated):")
        print(f"  Wire diameter: {d*1000:.2f}mm")
        print(f"  Small diameter: {D1*1000:.2f}mm")
        print(f"  Large diameter: {D2*1000:.2f}mm")
        print(f"  Active coils: {n}")
        print(f"  Calculated spring constant: {k:.5f} N/m")
        print(f"  Document spring constant: {self.expected_k_document:.5f} N/m")
        print(f"  Ratio (calculated/document): {k/self.expected_k_document:.3f}")
        
        return k


    def expected_natural_frequency(self):
        """Calculate expected natural frequency for the conical spring system"""
        k = self.expected_k
        m = self.cube_mass
        omega_n = np.sqrt(k / m)  # rad/s
        freq_hz = omega_n / (2 * np.pi)  # Hz
        
        # Also calculate using document spring constant
        omega_n_doc = np.sqrt(self.expected_k_document / m)
        freq_hz_doc = omega_n_doc / (2 * np.pi)
        
        print(f"Expected System Parameters:")
        print(f"  Cube mass: {m*1000:.1f}g")
        print(f"  Cube weight: {self.cube_weight:.3f}N")
        print(f"  Expected natural frequency (code): {freq_hz:.2f} Hz")
        print(f"  Expected natural frequency (document): {freq_hz_doc:.2f} Hz")
        print(f"  Expected deflection (document): {self.expected_deflection*1000:.1f}mm")
        print(f"  Expected max displacement: {self.cube_weight/k*1000:.1f}mm")
        
        return freq_hz
        
    def validate_scale(self):
        """Check if estimated positions have reasonable scale"""
        if not self.valid_detections:
            return False
            
        positions = np.array([d.pose_3d[:3, 3] for d in self.valid_detections])
        max_displacement = np.max(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1))
        
        print(f"\nScale Validation:")
        print(f"  Max displacement observed: {max_displacement*1000:.1f}mm")
        print(f"  Expected deflection from document: {self.expected_deflection*1000:.1f}mm")
        print(f"  Expected range for conical spring: {self.expected_deflection*500:.0f}-{self.expected_deflection*2000:.0f}mm")
        
        # Compare with document expectations
        if max_displacement > self.expected_deflection * 10:  # 10x expected
            print("  WARNING: Displacements much larger than expected - check camera calibration")
            return False
        elif max_displacement < self.expected_deflection * 0.1:  # 0.1x expected
            print("  WARNING: Displacements much smaller than expected - cube may not be oscillating properly")
            return False
        else:
            print("  Scale appears reasonable based on spring calculations")
            return True
        
    def parse_detection_file(self, detection_file: str):
        """Parse the YOLOv11 detection text file"""
        detections = []
        current_frame = -1
        
        try:
            with open(detection_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        if line.startswith('--- Frame'):
                            current_frame = int(line.split()[2])
                        elif line.startswith('Detection'):
                            # Parse: Detection 0: Confidence 0.8078, BBox (264.36, 184.25, 459.46, 446.12)
                            parts = line.split(':')
                            if len(parts) < 2:
                                continue
                                
                            conf_bbox = parts[1].strip()
                            
                            # Extract confidence
                            if 'Confidence' not in conf_bbox:
                                continue
                            conf_str = conf_bbox.split(',')[0].replace('Confidence', '').strip()
                            confidence = float(conf_str)
                            
                            # Extract bbox
                            if 'BBox' not in conf_bbox:
                                continue
                            bbox_str = conf_bbox.split('BBox')[1].strip()
                            bbox_coords = bbox_str.replace('(', '').replace(')', '').split(',')
                            
                            if len(bbox_coords) != 4:
                                continue
                                
                            bbox = tuple(float(x.strip()) for x in bbox_coords)
                            
                            detection = Detection(
                                frame_id=current_frame,
                                confidence=confidence,
                                bbox=bbox
                            )
                            detections.append(detection)
                            
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line {line_num}: {line}")
                        continue
                        
        except FileNotFoundError:
            print(f"Error: Detection file {detection_file} not found")
            return []
            
        self.detections = detections
        print(f"Successfully parsed {len(detections)} detections")
        return detections
    
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
    
    def estimate_3d_pose_from_bbox(self, detection: Detection):
        """Estimate 3D pose from 2D bounding box using PnP algorithm"""
        x1, y1, x2, y2 = detection.bbox
        
        # Validate bounding box
        if x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid bbox for frame {detection.frame_id}: {detection.bbox}")
            return detection
            
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
        
        # Improved 2D point estimation based on cube geometry
        # Assume we're looking at the front face primarily
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Use perspective projection approximation for depth estimation
        depth_offset = min(bbox_width, bbox_height) * 0.1  # More reasonable depth offset
        
        cube_2d_points = np.array([
            [x1, y1],           # front bottom-left
            [x2, y1],           # front bottom-right  
            [x2, y2],           # front top-right
            [x1, y2],           # front top-left
            [x1 + depth_offset, y1 + depth_offset],  # back bottom-left (estimated)
            [x2 + depth_offset, y1 + depth_offset],  # back bottom-right (estimated)
            [x2 + depth_offset, y2 + depth_offset],  # back top-right (estimated)
            [x1 + depth_offset, y2 + depth_offset],  # back top-left (estimated)
        ], dtype=np.float32)
        
        # Camera matrix
        camera_matrix = np.array([
            [self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']],
            [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Solve PnP to get rotation and translation
        dist_coeffs = np.zeros((4,1))  # Assuming no distortion
        
        try:
            success, rvec, tvec = cv2.solvePnP(
                cube_3d_points,
                cube_2d_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE  # More robust method
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Create 4x4 transformation matrix
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = tvec.flatten()
                
                detection.pose_3d = transform_matrix
                self.valid_detections.append(detection)
                
        except cv2.error as e:
            print(f"PnP solving failed for frame {detection.frame_id}: {e}")
            
        return detection
    
    def analyze_motion_patterns(self):
        """Analyze cube motion patterns and spring dynamics"""
        # Use only detections with valid 3D poses
        valid_detections = [d for d in self.detections if d.pose_3d is not None]
        
        if len(valid_detections) < 2:
            print("Warning: Not enough valid detections for motion analysis")
            return None
            
        positions = []
        timestamps = []
        confidences = []
        
        for detection in valid_detections:
            # Extract position from transformation matrix
            position = detection.pose_3d[:3, 3]
            positions.append(position)
            timestamps.append(detection.frame_id)
            confidences.append(detection.confidence)
        
        positions = np.array(positions)
        timestamps = np.array(timestamps)
        
        # Calculate velocities and accelerations with proper time intervals
        if len(positions) > 2:
            # Calculate time differences (assuming frame rate, adjust as needed)
            dt = np.diff(timestamps) * (1.0/15.0) # Assuming 15 FPS, adjust as needed
            dt[dt == 0] = 1.0/15.0 # Avoid division by zero
            
            # Calculate velocities
            velocity_vectors = np.diff(positions, axis=0)
            velocities = velocity_vectors / dt[:, np.newaxis]
            
            # Calculate accelerations
            if len(velocities) > 1:
                acceleration_vectors = np.diff(velocities, axis=0)
                dt_accel = dt[1:]  # Time intervals for acceleration
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
        ax.set_title(f'Vrishabha Cube 3D Trajectory ({motion_data["valid_frames"]} frames)')
        
        # Add colorbar for confidence
        plt.colorbar(scatter, label='Detection Confidence')
        
        return fig
    
    def simulate_spring_physics(self, motion_data):
        """Simulate spring dynamics based on observed motion - corrected for conical spring"""
        positions = motion_data['positions']
        
        if len(positions) < 3:
            print("Warning: Not enough data points for spring physics simulation")
            return None
        
        accelerations = motion_data.get('accelerations', np.array([]))
        if len(accelerations) == 0:
            print("Warning: No acceleration data available")
            return None
        
        # Center positions around mean
        mean_position = np.mean(positions, axis=0)
        displacements = positions[2:] - mean_position  # Align with accelerations
        
        # Estimate spring parameters using least squares
        spring_constants = []
        valid_axes = []
        
        for axis in range(3):
            # Filter out small displacements - adjusted threshold for conical spring
            non_zero_mask = np.abs(displacements[:, axis]) > self.displacement_threshold
            
            if np.sum(non_zero_mask) >= 10:  # Need at least 10 points
                disp_filtered = displacements[non_zero_mask, axis]
                accel_filtered = accelerations[non_zero_mask, axis]
                
                # Linear regression: a = -(k/m)*x for spring-mass system
                try:
                    # Calculate k/m ratio
                    k_over_m = -np.mean(accel_filtered / disp_filtered)
                    k_estimated = k_over_m * self.cube_mass
                    
                    if k_estimated > 0:  # Spring constant should be positive
                        spring_constants.append(k_estimated)
                        valid_axes.append(axis)
                        print(f"  Axis {axis}: k = {k_estimated:.2f} N/m")
                except (ValueError, RuntimeWarning, ZeroDivisionError):
                    print(f"  Axis {axis}: Could not estimate spring constant")
                    continue
            else:
                print(f"  Axis {axis}: Insufficient data points ({np.sum(non_zero_mask)})")
        
        if len(spring_constants) == 0:
            print("Warning: Could not estimate spring constants - try adjusting displacement threshold")
            print(f"Current threshold: {self.displacement_threshold*1000:.1f}mm")
            print(f"Max displacement observed: {np.max(np.abs(displacements))*1000:.1f}mm")
            return None
        
        avg_spring_constant = np.mean(spring_constants)
        measured_freq = np.sqrt(avg_spring_constant / self.cube_mass) / (2 * np.pi)
        
        print(f"\nSpring Analysis Results:")
        print(f"  Measured spring constants: {spring_constants}")
        print(f"  Average measured k: {avg_spring_constant:.2f} N/m")
        print(f"  Expected k (conical): {self.expected_k:.2f} N/m")
        print(f"  Ratio (measured/expected): {avg_spring_constant/self.expected_k:.2f}")
        print(f"  Measured natural frequency: {measured_freq:.2f} Hz")
        print(f"  Expected natural frequency: {self.expected_freq:.2f} Hz")
        
        return {
            'spring_constants': spring_constants,
            'valid_axes': valid_axes,
            'avg_spring_constant': avg_spring_constant,
            'expected_spring_constant': self.expected_k,
            'natural_frequency': measured_freq,
            'expected_frequency': self.expected_freq,
            'damping_estimate': self.estimate_damping(positions),
            'analysis_quality': len(spring_constants) / 3.0,  # 0-1 quality score
            'measurement_ratio': avg_spring_constant / self.expected_k
        }
    
    def estimate_damping(self, positions):
        """Estimate damping coefficient from position decay"""
        # Calculate amplitude decay analysis
        mean_position = np.mean(positions, axis=0)
        amplitudes = np.linalg.norm(positions - mean_position, axis=1)
        
        if len(amplitudes) < 10:
            return 0.1  # Default damping for insufficient data
        
        # Remove zero amplitudes for log calculation
        non_zero_amplitudes = amplitudes[amplitudes > 1e-6]
        
        if len(non_zero_amplitudes) < 5:
            return 0.1
        
        try:
            # Fit exponential decay: A(t) = A0 * exp(-damping * t)
            t = np.arange(len(non_zero_amplitudes))
            log_amplitudes = np.log(non_zero_amplitudes)
            
            # Linear fit to log(amplitude) vs time
            coeffs = np.polyfit(t, log_amplitudes, 1)
            damping = -coeffs[0]  # Decay rate
            
            # Clamp to reasonable range
            return max(0.001, min(1.0, damping))
            
        except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
            return 0.1  # Default damping on fitting failure

    def generate_point_cloud(self, frame_id: int):
        """Generate point cloud from depth estimation (placeholder)"""
        # This would integrate with your Depth-Anything-V2 results
        # For now, create a synthetic point cloud around the cube
        
        detection = next((d for d in self.detections if d.frame_id == frame_id), None)
        if not detection or detection.pose_3d is None:
            return None
        
        # Create point cloud around estimated cube position
        center = detection.pose_3d[:3, 3]
        
        # Generate random points around the cube with realistic noise
        n_points = 1000
        # Use cube size to determine point cloud extent
        noise_scale = self.cube_size * 0.4
        points = np.random.normal(center, noise_scale, (n_points, 3))
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color based on distance from center
        distances = np.linalg.norm(points - center, axis=1)
        colors = plt.cm.viridis(distances / np.max(distances))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def export_results(self, output_dir: str):
        """Export analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export detections with 3D poses
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
        
        # Export main results
        with open(output_path / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Export summary statistics
        summary = {
            'total_detections': len(self.detections),
            'valid_3d_poses': valid_count,
            'success_rate': valid_count / len(self.detections) if self.detections else 0,
            'cube_size': self.cube_size,
            'cube_mass': self.cube_mass,
            'cube_weight': self.cube_weight,
            'expected_spring_constant': self.expected_k,
            'expected_natural_frequency': self.expected_freq,
            'camera_intrinsics': self.camera_intrinsics,
            'displacement_threshold': self.displacement_threshold
        }
        
        with open(output_path / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results exported to {output_path}")
        print(f"Successfully processed {valid_count}/{len(self.detections)} detections")

# Example usage
def main():
    # Initialize analyzer
    analyzer = VrishabhaCubeAnalyzer("./vrishabha_data")
    
    # Parse detection file
    detections = analyzer.parse_detection_file("pose_estimation_log.txt")
    
    if not detections:
        print("No detections found. Check your detection file path and format.")
        return
    
    # Load frame images
    analyzer.load_frame_images(
        r"D:\Home\Desktop\Payload_2025_Research\depth_any\Raw_Camera\Raw_Camera_Output", 
        r"D:\Home\Desktop\Payload_2025_Research\depth_any\pose_frame\pose_frames"
    )
    
    # Estimate 3D poses for all detections
    print("Estimating 3D poses...")
    for detection in analyzer.detections:
        analyzer.estimate_3d_pose_from_bbox(detection)
    
    # Validate scale
    print("Validating measurement scale...")
    analyzer.validate_scale()
    
    # Analyze motion patterns
    print("Analyzing motion patterns...")
    motion_data = analyzer.analyze_motion_patterns()
    
    if motion_data:
        print(f"Motion analysis complete. Valid frames: {motion_data['valid_frames']}")
        
        # Create visualization
        try:
            fig = analyzer.create_3d_visualization(motion_data)
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        # Simulate physics
        print("Simulating spring physics...")
        physics_data = analyzer.simulate_spring_physics(motion_data)
        if physics_data:
            print(f"Analysis quality: {physics_data['analysis_quality']:.2f}")
        
        # Generate point clouds for first few frames
        print("Generating point clouds...")
        for i in range(min(3, len(analyzer.valid_detections))):
            pcd = analyzer.generate_point_cloud(analyzer.valid_detections[i].frame_id)
            if pcd:
                print(f"Generated point cloud for frame {analyzer.valid_detections[i].frame_id}")
                # Uncomment to visualize
                # o3d.visualization.draw_geometries([pcd])
    else:
        print("Motion analysis failed - insufficient valid detections")
    
    # Export results
    print("Exporting results...")
    analyzer.export_results("./output")

if __name__ == "__main__":
    main()
