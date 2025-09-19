from camera import RealSenseCamera
import cv2
import numpy as np

CAMERA_SERIAL = 'f1380660'

cam = RealSenseCamera(serial_number=CAMERA_SERIAL)
try:
    frame = cam.get_color_frame()
    print("Frame shape:", frame.shape)
    cv2.imwrite("test_img.jpg", frame)
finally:
    cam.stop()