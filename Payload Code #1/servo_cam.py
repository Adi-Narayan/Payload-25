import numpy as np
import cv2
from adafruit_servokit import ServoKit
import datetime
import threading
import queue
from collections import deque
import time

class FrameGrabber(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(maxsize=2)  # Small queue to prevent delay
        self.stopped = False
        
    def run(self):
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        
        # Updated camera parameters for 640x480
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Changed from 320
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Changed from 240
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                if self.queue.full():
                    try:
                        self.queue.get_nowait()  # Discard old frame
                    except queue.Empty:
                        pass
                self.queue.put(frame)
        cap.release()

    def get_frame(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True

class ServoController:
    def __init__(self):
        self.kit = ServoKit(channels=16, address=0x40)
        self.kit.servo[0].set_pulse_width_range(500, 2500)
        self.current_angle = 90
        self.target_angle = 90
        self.last_update = time.time()
        
    def update(self, target):
        self.target_angle = target
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Smooth movement
        max_speed = 180  # degrees per second
        max_movement = max_speed * dt
        
        diff = self.target_angle - self.current_angle
        movement = np.clip(diff, -max_movement, max_movement)
        
        self.current_angle += movement
        self.current_angle = np.clip(self.current_angle, 0, 180)
        
        self.kit.servo[0].angle = self.current_angle
        self.last_update = current_time

def main():
    print('!SERVO ACTIVE!')
    print(datetime.datetime.now())
    
    # Constants adjusted for 640x480 resolution
    TILT_GAIN = 0.05  # Reduced from 0.1 due to larger resolution
    CENTER_Y = 240  # Half of 480
    
    # Initialize components
    frame_grabber = FrameGrabber()
    frame_grabber.start()
    
    servo_controller = ServoController()
    position_history = deque(maxlen=3)  # For motion prediction
    
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = frame_grabber.get_frame()
            
            # Color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Simplified mask creation
            mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            mask = cv2.add(mask, mask2)
            
            # Adjusted kernel size for higher resolution
            kernel = np.ones((5,5), np.uint8)  # Increased from 3x3
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                biggest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(biggest_contour) > 200:  # Increased from 50 due to larger resolution
                    x, y, w, h = cv2.boundingRect(biggest_contour)
                    center_y = y + (h/2.0)
                    
                    # Motion prediction
                    position_history.append(center_y)
                    if len(position_history) == 3:
                        # Simple linear prediction
                        velocity = (position_history[-1] - position_history[-2])
                        predicted_y = center_y + velocity
                        
                        # Use predicted position for servo control
                        error_y = predicted_y - CENTER_Y
                        target_angle = servo_controller.current_angle + (error_y * TILT_GAIN)
                        servo_controller.update(target_angle)
                    
                    # Drawing
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)  # Increased line thickness
                    
                    # Add performance metrics with adjusted font size and position
                    cv2.putText(frame, f"Pos: {center_y:.1f}", (20, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # Increased size and thickness
                    cv2.putText(frame, f"Angle: {servo_controller.current_angle:.1f}", (20, 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            cv2.imshow('Tracking', frame)
            #cv2.imshow('Mask', mask)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    finally:
        frame_grabber.stop()
        frame_grabber.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error occurred: {e}")