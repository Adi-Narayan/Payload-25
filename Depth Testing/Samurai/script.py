import cv2
import os

def split_video_to_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Frame counter
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Finished processing video")
            break

        # Save frame as image file
        frame_filename = os.path.join(output_dir, f"{frame_count:08d}.jpg")
        cv2.imwrite(frame_filename, frame)

        print(f"Saved: {frame_filename}")
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"All frames saved in {output_dir}")

if __name__ == "__main__":
    video_path = "C:/Users/newpassword/Desktop/thrustMIT/Depth Testing/Samurai/Frames_pyramid.mp4"  # Replace with your video file path
    output_dir = "C:/Users/newpassword/Desktop/thrustMIT/Depth Testing/Samurai/Frames visualizer"  # Replace with the directory to save frames
    split_video_to_frames(video_path, output_dir)
