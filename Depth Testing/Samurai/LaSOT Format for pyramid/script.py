import os
import cv2
from ultralytics import YOLO

def yolo_to_lasot(yolo_model_path, img_folder, output_folder):
    """
    Converts YOLO model predictions to LaSOT format with detailed frame processing.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open files for writing
    groundtruth_file = open(os.path.join(output_folder, "groundtruth.txt"), "w")
    full_occlusion_file = open(os.path.join(output_folder, "full_occlusion.txt"), "w")
    out_of_view_file = open(os.path.join(output_folder, "out_of_view.txt"), "w")
    nlp_file = open(os.path.join(output_folder, "nlp.txt"), "w")

    try:
        # Load YOLO model
        model = YOLO(yolo_model_path)
        print(f"Loaded YOLO model from {yolo_model_path}")

        # Get sorted list of image files (specifically for frame_00001 format)
        frame_files = sorted([f for f in os.listdir(img_folder) 
                               if f.startswith('frame_') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        print(f"Found {len(frame_files)} image files to process")

        # Process each frame
        for frame_idx, frame_file in enumerate(frame_files, start=1):
            frame_path = os.path.join(img_folder, frame_file)
            
            # Read image
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Could not read image: {frame_path}")
                groundtruth_file.write("-1,-1,-1,-1\n")
                full_occlusion_file.write("1\n")
                out_of_view_file.write("1\n")
                nlp_file.write("Image could not be read.\n")
                continue

            # Convert image to RGB (YOLO expects RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run inference
            results = model(img_rgb)

            # Extract predictions
            if len(results[0].boxes) > 0:
                # Take the first detected object
                box = results[0].boxes[0]
                
                # Get bounding box coordinates
                x_center, y_center, width, height = box.xywh[0].tolist()
                
                # Convert to LaSOT format (x_min, y_min, width, height)
                x_min = int(x_center - width/2)
                y_min = int(y_center - height/2)
                abs_width = int(width)
                abs_height = int(height)

                # Prepare output strings
                bbox_str = f"{x_min},{y_min},{abs_width},{abs_height}"
                
                # Write to files
                groundtruth_file.write(f"{bbox_str}\n")
                full_occlusion_file.write("0\n")
                out_of_view_file.write("0\n")
                nlp_file.write("Object detected.\n")

                # Print debugging information
                print(f"Frame {frame_file} - Bounding Box: {bbox_str}")
                print(f"Confidence: {box.conf[0]:.2f}, Class: {box.cls[0]}")
            else:
                # No object detected
                print(f"Frame {frame_file} - No objects detected")
                groundtruth_file.write("-1,-1,-1,-1\n")
                full_occlusion_file.write("1\n")
                out_of_view_file.write("1\n")
                nlp_file.write("No object detected.\n")

            # Print progress every 50 frames
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames...")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close all files
        groundtruth_file.close()
        full_occlusion_file.close()
        out_of_view_file.close()
        nlp_file.close()

    print(f"LaSOT files saved in {output_folder}")

# Example usage
if __name__ == "__main__":
    yolo_model_path = "yolov8_pyramid.pt"  # Use the standard YOLOv8 nano model
    img_folder = "Frames_pyramid"  # Path to your image folder
    output_folder = "LaSOT Format for pyramid"  # Output folder for LaSOT files

    yolo_to_lasot(yolo_model_path, img_folder, output_folder)