import re
import json

def parse_detections(file_path):
    detections = []
    current_frame = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Match frame header
            frame_match = re.match(r'--- Frame (\d+)', line)
            if frame_match:
                current_frame = int(frame_match.group(1))
                continue
            
            # Match detection line
            if line.startswith('Detection'):
                try:
                    # Extract confidence
                    confidence_match = re.search(r'Confidence (\d+\.\d+)', line)
                    if not confidence_match:
                        print(f"Warning: Could not parse confidence in line: {line}")
                        continue
                    confidence = float(confidence_match.group(1))
                    
                    # Extract bounding box
                    bbox_match = re.search(r'BBox \(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\)', line)
                    if not bbox_match:
                        print(f"Warning: Could not parse BBox in line: {line}")
                        continue
                    bbox = [float(bbox_match.group(i)) for i in range(1, 5)]
                    
                    # Extract keypoints
                    keypoints_match = re.search(r'Keypoints \[(.*?)\]', line)
                    if not keypoints_match:
                        print(f"Warning: Could not parse keypoints in line: {line}")
                        continue
                    keypoints_str = keypoints_match.group(1)
                    keypoints = []
                    for kp in keypoints_str.split('), ('):
                        kp = kp.strip('()')
                        x, y, conf = map(float, kp.split(','))
                        keypoints.append([x, y, conf])
                    
                    # Extract orientation
                    orientation_match = re.search(r'Orientation ([\d.-]+)', line)
                    if not orientation_match:
                        print(f"Warning: Could not parse orientation in line: {line}")
                        continue
                    orientation = float(orientation_match.group(1))
                    
                    # Extract pose confidence
                    pose_conf_match = re.search(r'PoseConf ([\d.]+)', line)
                    if not pose_conf_match:
                        print(f"Warning: Could not parse PoseConf in line: {line}")
                        continue
                    pose_conf = float(pose_conf_match.group(1))
                    
                    # Store detection
                    detection = {
                        'frame': current_frame,
                        'confidence': confidence,
                        'bbox': bbox,
                        'keypoints': keypoints,
                        'orientation': orientation,
                        'pose_confidence': pose_conf
                    }
                    detections.append(detection)
                
                except Exception as e:
                    print(f"Warning: Could not parse line: {line} - Error: {str(e)}")
                    continue
    
    return detections

def export_results(detections, output_file):
    with open(output_file, 'w') as f:
        json.dump(detections, f, indent=4)
    print(f"Enhanced results exported to {output_file}")

# Example usage
file_path = r'D:\Home\Desktop\Payload_2025_Research\depth_any\output_detections\detections.txt'
output_file = 'results.json'
detections = parse_detections(file_path)
print(f"Successfully parsed {len(detections)} enhanced detections")
export_results(detections, output_file)