import json
import numpy as np
from datetime import datetime
import glob
import re
import warnings
import os
warnings.filterwarnings('ignore')

def install_plyfile():
    """Install plyfile library if not available"""
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile==0.9"])
        print("✅ Installed plyfile library")
    except Exception as e:
        print(f"❌ Could not install plyfile: {e}")
        return False
    return True

def install_open3d():
    """Install open3d library if not available"""
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "open3d"])
        print("✅ Installed open3d library")
    except Exception as e:
        print(f"❌ Could not install open3d: {e}")
        return False
    return True

def detect_coordinate_issues(points):
    """Detect and analyze coordinate system issues"""
    if len(points) == 0:
        return {}
    
    points = np.array(points)
    issues = {}
    
    abs_max = np.max(np.abs(points[np.isfinite(points)])) if np.any(np.isfinite(points)) else 0
    if abs_max > 1e10:
        issues['extreme_values'] = f"Maximum absolute coordinate: {abs_max:.2e}"
    
    inf_count = np.sum(np.isinf(points))
    if inf_count > 0:
        issues['infinite_values'] = f"Found {inf_count} infinite values"
    
    nan_count = np.sum(np.isnan(points))
    if nan_count > 0:
        issues['nan_values'] = f"Found {nan_count} NaN values"
    
    extreme_count = np.sum(np.any(np.abs(points) > 1e5, axis=1))
    if extreme_count > 0:
        issues['extreme_count'] = f"Found {extreme_count} points with abs value > 1e5"
    
    ranges = np.ptp(points[np.isfinite(points).all(axis=1)], axis=0) if np.any(np.isfinite(points).all(axis=1)) else [0, 0, 0]
    issues['coordinate_ranges'] = f"X: {ranges[0]:.2e}, Y: {ranges[1]:.2e}, Z: {ranges[2]:.2e}"
    
    if abs_max > 1e3:
        issues['unit_warning'] = "Coordinates likely in millimeters (expected meters)"
    
    invalid_mask = np.any(np.isnan(points) | np.isinf(points) | (np.abs(points) > 1e5), axis=1)
    if np.any(invalid_mask):
        invalid_points = points[invalid_mask][:5]
        issues['sample_invalid_points'] = invalid_points.tolist()
    
    return issues

def normalize_coordinates(points, method='robust_standardize', unit_conversion='auto'):
    """Normalize coordinates to reasonable ranges"""
    if len(points) == 0:
        return np.array([]).reshape(0, 3), {}
    
    points = np.array(points, dtype=np.float64)
    original_shape = points.shape
    
    median_abs = np.median(np.abs(points[np.isfinite(points)])) if np.any(np.isfinite(points)) else 0
    if unit_conversion == 'auto':
        scale = 1e-3 if median_abs > 1e3 else 1.0
    elif unit_conversion == 'mm_to_m':
        scale = 1e-3
    else:
        scale = 1.0
    points = points * scale
    
    print(f"  📏 Estimated median abs value: {median_abs:.2e}, scale factor: {scale}")
    
    threshold = 1e5 if median_abs > 1e3 else 1e3
    valid_mask = np.isfinite(points).all(axis=1)
    valid_mask &= np.all(np.abs(points) < threshold, axis=1)
    valid_points = points[valid_mask]
    
    valid_percentage = len(valid_points) / len(points) * 100
    if valid_percentage < 50:
        print(f"  ⚠️ Only {valid_percentage:.1f}% points remain after filtering, relaxing threshold")
        valid_mask = np.isfinite(points).all(axis=1)
        valid_points = points[valid_mask]
    
    if len(valid_points) < 100:
        return np.array([]).reshape(0, 3), {'error': f'Only {len(valid_points)} valid points after filtering, need at least 100'}
    
    normalization_info = {
        'original_count': int(len(points)),
        'valid_count': int(len(valid_points)),
        'removed_count': int(len(points) - len(valid_points)),
        'extreme_filtered': int(np.sum(np.any(np.abs(points) >= threshold, axis=1))),
        'threshold_used': float(threshold),
        'valid_percentage': float(valid_percentage),
        'unit_scale': float(scale)
    }
    
    if method == 'robust_standardize':
        median_point = np.median(valid_points, axis=0)
        mad = np.median(np.abs(valid_points - median_point), axis=0)
        mad = np.where(mad == 0, 1, mad)
        normalized_points = (valid_points - median_point) / mad
        normalization_info.update({
            'method': 'robust_standardize',
            'median_center': median_point.tolist(),
            'mad_scale': mad.tolist()
        })
    else:
        normalized_points = valid_points
        normalization_info['method'] = 'none'
    
    return normalized_points, normalization_info

def read_ply_file_robust(filepath):
    """Read PLY file with improved error handling"""
    try:
        from plyfile import PlyData
        if not hasattr(PlyData, 'read'):
            print("  ❌ plyfile module is corrupted or incompatible")
            return read_ply_file_open3d(filepath)
    except ImportError:
        print("plyfile library not found. Installing...")
        if not install_plyfile():
            return read_ply_file_open3d(filepath)
        from plyfile import PlyData
    
    try:
        plydata = PlyData.read(filepath)
        vertex_data = plydata['vertex']
        
        try:
            prop_names = vertex_data.dtype.names
            print(f"  📋 Available properties: {list(prop_names)}")
            prop_types = {name: str(vertex_data[name].dtype) for name in prop_names}
            print(f"  📋 Property types: {prop_types}")
            if len(vertex_data) > 0:
                print(f"  📋 First point: {vertex_data[0]}")
        except AttributeError as e:
            print(f"  ❌ Error accessing vertex properties: {str(e)}")
            print(f"  📋 PLY elements: {[el.name for el in plydata.elements]}")
            return read_ply_file_open3d(filepath)
        
        coord_names = [
            ('x', 'y', 'z'),
            ('X', 'Y', 'Z'),
            ('px', 'py', 'pz'),
            ('pos_x', 'pos_y', 'pos_z')
        ]
        
        points = []
        colors = []
        for x_name, y_name, z_name in coord_names:
            try:
                if all(name in prop_names for name in [x_name, y_name, z_name]):
                    x_coords = vertex_data[x_name].astype(np.float64)
                    y_coords = vertex_data[y_name].astype(np.float64)
                    z_coords = vertex_data[z_name].astype(np.float64)
                    points = np.column_stack((x_coords, y_coords, z_coords))
                    print(f"  ✅ Found coordinates using: {x_name}, {y_name}, {z_name}")
                    if all(c in prop_names for c in ['red', 'green', 'blue']):
                        r = vertex_data['red'].astype(np.uint8)
                        g = vertex_data['green'].astype(np.uint8)
                        b = vertex_data['blue'].astype(np.uint8)
                        colors = np.column_stack((r, g, b))
                        print(f"  ✅ Found colors: red, green, blue")
                        print(f"  📋 First 5 colors: {colors[:5]}")
                    break
            except:
                continue
        
        if len(points) == 0:
            numeric_props = [name for name in prop_names 
                           if vertex_data[name].dtype.kind in 'fc']
            if len(numeric_props) >= 3:
                points = np.column_stack([vertex_data[prop].astype(np.float64) for prop in numeric_props[:3]])
                print(f"  📊 Using first 3 numeric properties: {numeric_props[:3]}")
        
        return points, colors if colors else None
        
    except Exception as e:
        print(f"  ❌ plyfile error: {str(e)}")
        return read_ply_file_open3d(filepath)

def read_ply_file_open3d(filepath):
    """Read PLY file using open3d as fallback"""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d library not found. Installing...")
        if not install_open3d():
            return read_ply_file_manual(filepath)
        import open3d as o3d
    
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors else None
        if colors is not None:
            if np.max(colors) <= 1.0:  # Normalize [0, 1] to [0, 255]
                colors = (colors * 255).astype(np.uint8)
            else:  # Assume already in [0, 255]
                colors = colors.astype(np.uint8)
            print(f"  ✅ Loaded colors")
            print(f"  📋 First 5 colors: {colors[:5]}")
        print(f"  ✅ Loaded {len(points)} points using open3d")
        if len(points) > 0:
            print(f"  📋 First point: {points[0]}")
        return points, colors
    except Exception as e:
        print(f"  ❌ open3d error: {str(e)}")
        return read_ply_file_manual(filepath)

def read_ply_file_manual(filepath):
    """Manual PLY file reader with improved binary support"""
    import struct
    
    points = []
    colors = []
    
    try:
        with open(filepath, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    break
                
                try:
                    line_str = line.decode('utf-8', errors='ignore').strip()
                except:
                    line_str = line.decode('latin-1', errors='ignore').strip()
                
                header_lines.append(line_str)
                if line_str == 'end_header':
                    break
            
            vertex_count = 0
            is_binary = False
            properties = []
            
            for line in header_lines:
                if line.startswith('format'):
                    if 'binary' in line:
                        is_binary = True
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    if len(parts) >= 3:
                        prop_type = parts[1]
                        prop_name = parts[2]
                        properties.append((prop_type, prop_name))
            
            print(f"  📋 Format: {'Binary' if is_binary else 'ASCII'}")
            print(f"  📊 Vertices: {vertex_count}")
            print(f"  🏷️  Properties: {[p[1] for p in properties]}")
            
            if vertex_count == 0:
                return np.array([]).reshape(0, 3), None
            
            xyz_indices = {}
            rgb_indices = {}
            for i, (prop_type, prop_name) in enumerate(properties):
                if prop_name.lower() in ['x', 'y', 'z']:
                    xyz_indices[prop_name.lower()] = i
                if prop_name.lower() in ['red', 'green', 'blue']:
                    rgb_indices[prop_name.lower()] = i
            
            if len(xyz_indices) < 3:
                print(f"  ⚠️  Could not find x,y,z coordinates. Using first 3 properties.")
                xyz_indices = {'x': 0, 'y': 1, 'z': 2}
            
            if is_binary:
                points, colors = read_binary_vertices(f, vertex_count, properties, xyz_indices, rgb_indices)
            else:
                points, colors = read_ascii_vertices(f, vertex_count, xyz_indices, rgb_indices)
                
            if points:
                print(f"  📋 First point: {points[0]}")
                if colors:
                    print(f"  📋 First color: {colors[0]}")
                
    except Exception as e:
        print(f"  ❌ Manual reader error: {str(e)}")
        return np.array([]).reshape(0, 3), None
    
    return np.array(points, dtype=np.float64) if points else np.array([]).reshape(0, 3), colors if colors else None

def read_binary_vertices(f, vertex_count, properties, xyz_indices, rgb_indices):
    """Read binary vertex data"""
    import struct
    
    points = []
    colors = []
    
    format_chars = []
    for prop_type, prop_name in properties:
        if prop_type in ['float', 'float32']:
            format_chars.append('f')
        elif prop_type in ['double', 'float64']:
            format_chars.append('d')
        elif prop_type in ['uchar', 'uint8']:
            format_chars.append('B')
        else:
            format_chars.append('B')
    format_string = '<' + ''.join(format_chars)
    struct_size = struct.calcsize(format_string)
    
    x_idx = xyz_indices.get('x', 0)
    y_idx = xyz_indices.get('y', 1)
    z_idx = xyz_indices.get('z', 2)
    r_idx = rgb_indices.get('red', None)
    g_idx = rgb_indices.get('green', None)
    b_idx = rgb_indices.get('blue', None)
    
    for i in range(vertex_count):
        try:
            data = f.read(struct_size)
            if len(data) < struct_size:
                break
                
            values = struct.unpack(format_string, data)
            if len(values) > max(x_idx, y_idx, z_idx):
                x = float(values[x_idx])
                y = float(values[y_idx])
                z = float(values[z_idx])
                points.append([x, y, z])
                if all(idx is not None for idx in [r_idx, g_idx, b_idx]):
                    r = values[r_idx]
                    g = values[g_idx]
                    b = values[b_idx]
                    colors.append([r, g, b])
        except Exception:
            continue
    
    return points, colors if colors else None

def read_ascii_vertices(f, vertex_count, xyz_indices, rgb_indices):
    """Read ASCII vertex data"""
    points = []
    colors = []
    
    x_idx = xyz_indices.get('x', 0)
    y_idx = xyz_indices.get('y', 1)
    z_idx = xyz_indices.get('z', 2)
    r_idx = rgb_indices.get('red', None)
    g_idx = rgb_indices.get('green', None)
    b_idx = rgb_indices.get('blue', None)
    
    for i in range(vertex_count):
        try:
            line = f.readline()
            if not line:
                break
                
            line_str = line.decode('utf-8', errors='ignore').strip()
            if line_str:
                values = line_str.split()
                if len(values) > max(x_idx, y_idx, z_idx):
                    x = float(values[x_idx])
                    y = float(values[y_idx])
                    z = float(values[z_idx])
                    points.append([x, y, z])
                    if all(idx is not None for idx in [r_idx, g_idx, b_idx]) and len(values) > max(r_idx, g_idx, b_idx):
                        r = int(values[r_idx])
                        g = int(values[g_idx])
                        b = int(values[b_idx])
                        colors.append([r, g, b])
        except (ValueError, IndexError):
            continue
    
    return points, colors if colors else None

def save_points(points, filename, colors=None):
    """Save points to a PLY file for inspection"""
    try:
        from plyfile import PlyData, PlyElement
        if colors is not None and len(colors) == len(points):
            # Validate colors
            colors = np.array(colors, dtype=np.uint8)
            if np.all(colors == 0):
                print(f"  ⚠️ All colors are black in {filename}")
            else:
                print(f"  📋 Color range in {filename}: min={np.min(colors, axis=0)}, max={np.max(colors, axis=0)}")
            vertex = np.array([(float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2]))
                              for p, c in zip(points, colors)],
                             dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        else:
            vertex = np.array([(float(p[0]), float(p[1]), float(p[2])) for p in points],
                             dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(filename)
        print(f"  💾 Saved points to: {filename}")
    except Exception as e:
        print(f"  ❌ Error saving points to {filename}: {str(e)}")

def extract_cube_centroid_improved(points, frame_index, colors=None):
    """Extract cube centroid with coordinate normalization"""
    if len(points) == 0:
        print("    ❌ No points to process")
        return None, 0.0, {}
    
    print(f"    📊 Processing {len(points)} points")
    
    try:
        save_points(points, f"raw_frame_{frame_index}.ply", colors)
    except Exception as e:
        print(f"    ❌ Failed to save raw points: {str(e)}")
    
    issues = detect_coordinate_issues(points)
    if issues:
        print("    ⚠️  Coordinate issues detected:")
        for key, value in issues.items():
            if key == 'sample_invalid_points':
                print(f"      - {key}:")
                for pt in value:
                    print(f"        {pt}")
            else:
                print(f"      - {key}: {value}")
    
    normalized_points, norm_info = normalize_coordinates(points, method='robust_standardize')
    
    if len(normalized_points) == 0:
        print("    ❌ No valid points after normalization: " + norm_info.get('error', 'Unknown error'))
        return None, 0.0, norm_info
    
    print(f"    ✅ {len(normalized_points)} valid points after normalization")
    
    centroid = np.mean(normalized_points, axis=0)
    
    if not np.isfinite(centroid).all():
        print("    ❌ Invalid centroid with inf/nan values")
        return None, 0.0, norm_info
    if np.any(np.abs(centroid) > 1e3):
        print(f"    ❌ Centroid values too large: {centroid}")
        return None, 0.0, norm_info
    
    quality = calculate_quality_score(normalized_points, centroid)
    
    print(f"    ✅ Normalized centroid: ({centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f})")
    print(f"    📊 Quality score: {quality:.3f}")
    
    return centroid.tolist(), float(quality), norm_info

def calculate_quality_score(points, centroid):
    """Calculate quality score based on point distribution and density"""
    if len(points) == 0:
        return 0.0
    
    count_quality = min(1.0, len(points) / 5000.0)
    
    distances = np.linalg.norm(points - centroid, axis=1)
    
    if len(distances) > 1:
        mean_dist = np.mean(distances)
        if mean_dist > 0:
            cv = np.std(distances) / mean_dist
            consistency_quality = max(0.1, 1.0 - cv / 2.0)
        else:
            consistency_quality = 0.5
    else:
        consistency_quality = 0.5
    
    overall_quality = (count_quality * 0.4 + consistency_quality * 0.6)
    
    return min(1.0, max(0.0, overall_quality))

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename"""
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6})', filename)
    if match:
        timestamp_str = match.group(1)
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S-%f')
            return dt.timestamp()
        except ValueError:
            print(f"  ⚠️ Invalid timestamp format in {filename}")
            return None
    print(f"  ⚠️ No timestamp found in {filename}")
    return None

def process_vrishabha_data_improved(data_folder, max_files=None, start_from=0):
    """Process Vrishabha data with improved coordinate handling"""
    print(f"🚀 Processing Vrishabha mission data from: {data_folder}")
    print("="*60)
    
    ply_pattern = os.path.join(data_folder, "*_pointcloud.ply")
    ply_files = glob.glob(ply_pattern)
    
    print(f"📁 Found {len(ply_files)} PLY files")
    
    if len(ply_files) == 0:
        print("❌ No PLY files found! Check the folder path.")
        return None
    
    ply_files.sort(key=lambda x: extract_timestamp_from_filename(os.path.basename(x)) or 0)
    
    if start_from > 0:
        ply_files = ply_files[start_from:]
        print(f"📍 Starting from file {start_from}")
    
    if max_files:
        ply_files = ply_files[:max_files]
        print(f"📊 Processing maximum {max_files} files")
    
    for ply_file in ply_files:
        timestamp = extract_timestamp_from_filename(os.path.basename(ply_file))
        print(f"  ⏰ File: {os.path.basename(ply_file)}, Timestamp: {timestamp}")
    
    frames_data = []
    start_time = None
    successful_frames = 0
    failed_frames = 0
    normalization_data = []
    
    for i, ply_file in enumerate(ply_files):
        print(f"\n📊 Processing {i+1}/{len(ply_files)}: {os.path.basename(ply_file)}")
        
        try:
            points, colors = read_ply_file_robust(ply_file)
            
            if len(points) == 0:
                print(f"  ⚠️  No points found in {ply_file}")
                failed_frames += 1
                continue
            
            print(f"  📈 Loaded {len(points)} points")
            
            centroid, quality, norm_info = extract_cube_centroid_improved(points, i, colors)
            
            if centroid is None:
                print(f"  ❌ Could not extract centroid from {ply_file}")
                failed_frames += 1
                continue
            
            normalized_points, _ = normalize_coordinates(points, method='robust_standardize')
            if len(normalized_points) > 0:
                save_points(normalized_points, f"normalized_frame_{i}.ply", colors)
            
            normalization_data.append({
                'frame': int(successful_frames),
                'filename': os.path.basename(ply_file),
                'normalization_info': norm_info
            })
            
            timestamp = extract_timestamp_from_filename(os.path.basename(ply_file))
            if start_time is None and timestamp:
                start_time = timestamp
            
            relative_time = float(timestamp - start_time) if timestamp and start_time else float(i * 0.033)
            
            frame_data = {
                'id': int(successful_frames),
                'filename': os.path.basename(ply_file),
                'timestamp': float(relative_time),
                'position': {
                    'x': float(centroid[0]),
                    'y': float(centroid[1]),
                    'z': float(centroid[2])
                },
                'quality': float(quality),
                'point_count': int(len(points)),
                'velocity': float(0.0),
                'gForce': float(0.0)
            }
            
            frames_data.append(frame_data)
            successful_frames += 1
            
            print(f"  ✅ SUCCESS! Frame {successful_frames} processed")
            
        except Exception as e:
            print(f"  ❌ Error processing {ply_file}: {str(e)}")
            failed_frames += 1
            continue
    
    if not frames_data:
        print("\n❌ No valid frames processed!")
        return None
    
    print(f"\n🧮 Calculating physics data...")
    for i in range(1, len(frames_data)):
        prev_frame = frames_data[i-1]
        curr_frame = frames_data[i]
        
        dt = curr_frame['timestamp'] - prev_frame['timestamp']
        if dt <= 0:
            print(f"  ⚠️ Invalid time difference for frame {i}: dt={dt}, using default 0.033s")
            dt = 0.033
        
        dx = curr_frame['position']['x'] - prev_frame['position']['x']
        dy = curr_frame['position']['y'] - prev_frame['position']['y']
        dz = curr_frame['position']['z'] - prev_frame['position']['z']
        
        velocity = np.sqrt(dx*dx + dy*dy + dz*dz) / dt
        curr_frame['velocity'] = float(velocity)
        
        if i > 1:
            prev_velocity = frames_data[i-1]['velocity']
            acceleration = abs(velocity - prev_velocity) / dt
            g_force = acceleration / 9.81
            curr_frame['gForce'] = float(g_force)
    
    print(f"\n🎯 MISSION DATA PROCESSING COMPLETE!")
    print("="*60)
    print(f"✅ Successfully processed: {len(frames_data)} frames")
    print(f"❌ Failed frames: {failed_frames}")
    print(f"📊 Success rate: {len(frames_data)/(len(frames_data)+failed_frames)*100:.1f}%")
    if frames_data:
        print(f"⏱️  Time span: {frames_data[-1]['timestamp']:.2f} seconds")
        print(f"🎯 Average quality: {np.mean([f['quality'] for f in frames_data]):.3f}")
        print(f"🚀 Max velocity: {max([f['velocity'] for f in frames_data]):.6f} normalized units/s")
        print(f"⚡ Max G-force: {max([f['gForce'] for f in frames_data]):.6f} normalized G")
    
    return {
        'frames': frames_data,
        'normalization_data': normalization_data
    }

def save_trajectory_data(processed_data, output_file):
    """Save processed trajectory data to JSON file"""
    frames_data = processed_data['frames']
    normalization_data = processed_data['normalization_data']
    
    def convert_to_json_serializable(obj):
        """Convert NumPy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        return obj
    
    output_data = {
        'mission': 'Vrishabha',
        'description': 'Real flight data from cube tracking experiment with coordinate normalization',
        'coordinate_system': 'normalized',
        'total_frames': int(len(frames_data)),
        'duration': float(frames_data[-1]['timestamp']) if frames_data else 0.0,
        'avg_quality': float(np.mean([f['quality'] for f in frames_data])) if frames_data else 0.0,
        'max_velocity': float(max([f['velocity'] for f in frames_data])) if frames_data else 0.0,
        'max_gforce': float(max([f['gForce'] for f in frames_data])) if frames_data else 0.0,
        'frames': frames_data,
        'normalization_metadata': normalization_data
    }
    
    output_data = convert_to_json_serializable(output_data)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Trajectory data saved to: {output_file}")

if __name__ == "__main__":
    DATA_FOLDER = r"D:\Home\Desktop\Payload_2025_Research\depth_any\vrishabha_output"
    OUTPUT_FILE = "vrishabha_trajectory_data.json"
    
    print("🚖 VRISHABHA DATA PROCESSOR")
    print("="*80)
    
    try:
        processed_data = process_vrishabha_data_improved(
            DATA_FOLDER, 
            max_files=674, 
            start_from=0
        )
        
        if processed_data:
            save_trajectory_data(processed_data, OUTPUT_FILE)
            
            print(f"\n🎉 SUCCESS! Processing complete!")
            print("="*80)
            print("🎯 Payload validated!")
            print("📈 Extreme coords normalized!")
            print("🚖 Ready for visualization!")
            print(f"\n📌 Next steps:")
            print("1. 📂 Use normalized data for visualization")
            print("2. 📊 Check data integrity")
            print("3. 📈 Visualize trajectory")
            
            frames = processed_data['frames']
            if len(frames) > 0:
                print(f"\n📈 Normalized trajectory points:")
                for i in range(min(5, len(frames))):
                    frame = frames[i]
                    pos = frame['position']
                    print(f"  Frame {i}: ({pos['x']:.6f}, {pos['y']:.6f}, {pos['z']:.6f}) -> Quality: {frame['quality']:.3f}")
        else:
            print("\n❌ Processing failed!")
            print("Check the troubleshooting tips")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Check traceback for details")
        raise
