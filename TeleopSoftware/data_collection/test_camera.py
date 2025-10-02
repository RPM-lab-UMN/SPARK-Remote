from camera import RealSenseCamera
import cv2
import numpy as np

# Define the serial number for your camera
CAMERA_SERIAL = 'f1380660'

# Initialize the camera
cam = RealSenseCamera(serial_number=CAMERA_SERIAL)

print("Press 'q' to quit.")

# Start a continuous streaming loop
try:
    # We use a 'while True' loop for continuous capture
    while True:
        # 1. Get the latest color frame from the camera
        frame = cam.get_color_frame()
        
        # 2. Check if a frame was successfully captured (important for RealSense)
        if frame is not None:
            # 3. Display the frame in a window named 'RealSense Stream'
            cv2.imshow('RealSense Stream', frame)
            
            # 4. Check for a key press to exit the loop
            # 'cv2.waitKey(1)' waits for 1 millisecond.
            # '0xFF == ord('q')' checks if the key pressed was 'q'.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Could not retrieve a frame.")
            
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 5. Clean up: close all OpenCV windows and stop the camera stream
    cv2.destroyAllWindows()
    cam.stop()
    print("Streaming stopped and resources released.")