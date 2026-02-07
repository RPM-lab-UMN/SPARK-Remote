from camera import RealSenseCamera
import cv2
import numpy as np

# Define the serial number for your camera
# CAMERA_SERIAL = '128422270284'  # D405 (old)
# CAMERA_SERIAL = '130322273305'  # D405 (wrist)
CAMERA_SERIAL = 'f1380660'  # L515 (scene)

# Initialize the camera
cam = RealSenseCamera(serial_number=CAMERA_SERIAL)

print("Press 'q' to quit.")

# Start a continuous streaming loop
try:
    # We use a 'while True' loop for continuous capture
    while True:
        # 1. Get the latest color frame (and ignore the depth frame with '_')
        color_frame, _ = cam.get_frames()
        frame = color_frame # Assign to the variable used for display
        
        # 2. Check if a frame was successfully captured (important for RealSense)
        if frame is not None:
            # 3. Display the frame in a window named 'RealSense Stream'
            cv2.imshow('RealSense Stream', frame)
            
            # 4. Check for a key press to exit the loop
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