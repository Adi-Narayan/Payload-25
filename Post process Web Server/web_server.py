from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import json
import numpy as np
from pathlib import Path
import base64
import cv2

# Import your analyzer class
from reconstruct3d import VrishabhaCubeAnalyzer

app = Flask(__name__)
CORS(app)

class VrishabhaWebServer:
    def __init__(self, analyzer_results_path="./output"):
        self.results_path = Path(analyzer_results_path)
        self.mission_data = self.load_analysis_results()
        
    def load_analysis_results(self):
        """Load the analysis results from the JSON file"""
        try:
            results_file = self.results_path / 'analysis_results.json'
            print(f"Looking for results file at: {results_file}")
            
            if not results_file.exists():
                print("Analysis results file not found. Creating output directory and using simulated data.")
                self.results_path.mkdir(exist_ok=True)
                return self.generate_simulated_data()
                
            with open(results_file, 'r') as f:
                raw_data = json.load(f)
                
            print(f"Loaded {len(raw_data)} detections from file")
            
            # Convert to web-friendly format
            mission_data = []
            for i, detection in enumerate(raw_data):
                if detection['pose_3d'] is not None:
                    pose_matrix = np.array(detection['pose_3d'])
                    position = pose_matrix[:3, 3]
                    rotation = self.rotation_matrix_to_euler(pose_matrix[:3, :3])
                    
                    frame_data = {
                        'frame': detection['frame_id'],
                        'position': {
                            'x': float(position[0]),
                            'y': float(position[1]),
                            'z': float(position[2])
                        },
                        'rotation': {
                            'x': float(rotation[0]),
                            'y': float(rotation[1]),
                            'z': float(rotation[2])
                        },
                        'confidence': float(detection['confidence']),
                        'bbox': detection['bbox']
                    }
                    mission_data.append(frame_data)
                    
            print(f"Processed {len(mission_data)} frames with valid 3D poses")
            return mission_data
            
        except Exception as e:
            print(f"Error loading analysis results: {e}")
            print("Using simulated data instead.")
            return self.generate_simulated_data()

    
    def rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to Euler angles"""
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        
        return [x, y, z]
    
    def generate_simulated_data(self):
        """Generate simulated data if real data isn't available"""
        data = []
        frames = 200
        
        for i in range(frames):
            t = i * 0.1
            x = 0.3 * np.sin(t * 2) * np.exp(-t * 0.1) + (np.random.random() - 0.5) * 0.05
            y = 0.2 * np.cos(t * 1.5) * np.exp(-t * 0.08) + (np.random.random() - 0.5) * 0.05
            z = 0.4 * np.sin(t * 0.8) * np.exp(-t * 0.12) + (np.random.random() - 0.5) * 0.03
            
            confidence = max(0.3, 1.0 - i * 0.002 + (np.random.random() - 0.5) * 0.3)
            
            data.append({
                'frame': i,
                'position': {'x': x, 'y': y, 'z': z},
                'rotation': {'x': t * 0.5, 'y': t * 0.3, 'z': t * 0.7},
                'confidence': min(1.0, confidence),
                'bbox': [200 + x * 100, 150 + y * 100, 300 + x * 100, 250 + y * 100]
            })
        
        return data

# Global server instance
server = VrishabhaWebServer()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'vrishabha_visualizer.html')

@app.route('/debug_vrishabha.html')
def debug_page():
    """Serve the debug HTML page"""
    return send_from_directory('.', 'debug_vrishabha.html')

@app.route('/api/mission-data')
def get_mission_data():
    """API endpoint to get mission data"""
    return jsonify(server.mission_data)

@app.route('/api/mission-stats')
def get_mission_stats():
    """Get mission statistics"""
    if not server.mission_data:
        return jsonify({'error': 'No data available'})
    
    positions = [[d['position']['x'], d['position']['y'], d['position']['z']] for d in server.mission_data]
    positions = np.array(positions)
    
    stats = {
        'total_frames': len(server.mission_data),
        'avg_confidence': np.mean([d['confidence'] for d in server.mission_data]),
        'max_displacement': np.max(np.linalg.norm(positions, axis=1)),
        'position_std': {
            'x': float(np.std(positions[:, 0])),
            'y': float(np.std(positions[:, 1])),
            'z': float(np.std(positions[:, 2]))
        },
        'confidence_range': {
            'min': float(np.min([d['confidence'] for d in server.mission_data])),
            'max': float(np.max([d['confidence'] for d in server.mission_data]))
        }
    }
    
    return jsonify(stats)

@app.route('/api/frame/<int:frame_id>')
def get_frame_data(frame_id):
    """Get data for a specific frame"""
    frame_data = next((d for d in server.mission_data if d['frame'] == frame_id), None)
    if frame_data:
        return jsonify(frame_data)
    else:
        return jsonify({'error': 'Frame not found'}), 404

@app.route('/api/trajectory')
def get_trajectory():
    """Get trajectory data for plotting"""
    trajectory = [
        {
            'x': d['position']['x'],
            'y': d['position']['y'], 
            'z': d['position']['z'],
            'frame': d['frame'],
            'confidence': d['confidence']
        }
        for d in server.mission_data
    ]
    return jsonify(trajectory)

@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check server status"""
    return jsonify({
        'server_status': 'running',
        'data_loaded': len(server.mission_data) > 0,
        'frame_count': len(server.mission_data),
        'results_path': str(server.results_path),
        'sample_data': server.mission_data[:2] if server.mission_data else []
    })


if __name__ == '__main__':
    print("🚀 Vrishabha Web Visualizer Server Starting...")
    print("📊 Loading mission data...")
    print(f"✅ Loaded {len(server.mission_data)} frames of data")
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("🔧 Make sure 'vrishabha_visualizer.html' is in the same directory")
    print("🐛 Debug page available at: http://localhost:5000/debug_vrishabha.html")
    print("\n⚡ Server starting on port 5000...")
    
    app.run(host='0.0.0.0', port=5000, debug=True)