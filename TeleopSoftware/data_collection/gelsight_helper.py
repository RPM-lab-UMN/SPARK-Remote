"""
    Description:  Helper Class for GelSight Sensor
                - Minimal implementation of GelSight SDK https://github.com/gelsightinc/gsrobotics/tree/main
                - Read, Write, Image processing, recording helpers
                
    Feel free to reach out to Pranay Junare (junar002@umn.edu) in case of any queries.
    Date: Oct 2025
"""

import cv2
import platform
import glob
import time
from cv2.typing import MatLike
import os
import re
import datetime
from typing import Optional
import numpy as np

FULLWIDTH = 3280
FULLHEIGHT= 2464

class Camera:
    def __init__(self, device):
        """
        Initialize the Camera instance.

        Args:
            device: A numeric index (for Windows/macOS) or device path (for Linux).
        """
        self.device = device
        self.cap = None

    def open(self) -> None:
        """
        Open the camera device using OpenCV.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device: {self.device}")

    def read_frame(self) -> MatLike:
        """
        Read a frame from the camera.

        Returns:
            MatLike: The captured frame.

        Raises:
            RuntimeError: If the camera is not opened or frame capture fails.
        """
        if self.cap is None:
            raise RuntimeError("Camera is not opened.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from device.")
        return frame

    def release(self) -> None:
        """
        Release the camera resource.
        """
        if self.cap:
            self.cap.release()
            self.cap = None

    @staticmethod
    def list_devices() -> dict:
        """
        Enumerate available camera devices.

        On Linux, returns unique device paths from /dev/v4l/by-id/.
        On Windows/macOS, tests numeric indices 0..5.

        Returns:
            dict: A mapping from index to device identifier.
        """
        devices = {}
        os_name = platform.system()
        if os_name == "Linux":
            paths = glob.glob("/dev/v4l/by-id/*")
            for idx, path in enumerate(paths):
                devices[idx] = path
        else:
            for idx in range(6):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    devices[idx] = f"Camera {idx}"
                    cap.release()
        return devices

    def find_cameras_windows(camera_name):
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()

        # get the device name
        allcams = graph.get_input_devices() # list of camera device
        description = ""
        for cam in allcams:
            if camera_name in cam:
                description = cam

        try:
            device = graph.get_input_devices().index(description)
        except ValueError as e:
            print("Device is not in this list")
            print(graph.get_input_devices())
            import sys
            sys.exit()

        return (device,description)


class GelSightMini:
    def __init__(
        self,
        device_idx: int=0,
        target_width: int = 320,
        target_height: int = 240,
        border_fraction: float = 0.15,
    ):
        """
        Initialize the CameraStream.

        Args:
            target_width (int, optional): Desired width of the camera feed. Defaults to 320.
            target_height (int, optional): Desired height of the camera feed. Defaults to 240.
            border_fraction: (float, optional): Desired border size for the image crop Defaults to 15%.
        """
        self.camera: Camera = None
        self.recording: bool = False
        self.record_filepath: str = None
        self.frame_count: int = 0
        self.time_prev: float = time.time()
        self.fps: float = 0
        self.current_frame_rgb: MatLike = None
        self.current_frame: MatLike = None
        self.target_width: int = target_width
        self.target_height: int = target_height
        self.border_fraction: float = border_fraction
        self.video_writer = None
        self.serial_number = None

        # initialize
        self.select_device(device_idx=device_idx)
        self.start()

    def get_device_list(self) -> dict:
        """
        Get a dictionary of available camera devices.

        Returns:
            dict: Mapping of device indices to device identifiers.
        """
        return Camera.list_devices()


    def select_device(self, device_idx=None) -> None:
        """
        Select and open a camera device with the desired resolution.

        Args:
            device_idx (int/str): The index of the device to select.
            eg: /dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._GelSight_Mini_R0B_28F5-K4RR_28F5K4RR-video-index0
        """

        device_id = device_idx
        if device_idx==None and platform.system() == "Windows":
                (dev, desc) = Camera.find_cameras_windows("GelSight Mini")
                print("Found: ", desc, ", dev: ", dev)
                device_idx = dev
                # Parse serial number from description
                match = re.search("[A-Z0-9]{4}-[A-Z0-9]{4}", desc)
                if match:
                    self.serial_number = match.group()


        if platform.system() == "Linux":
            devices = Camera.list_devices()
            for ix in range(0,len(devices)):
                print("Device: ", devices[ix])

            if isinstance(devices.get(device_idx), str):
                device_id = devices[device_idx]
        else:
            device_id = device_idx


        if self.camera:
            self.camera.release()

        self.camera = Camera(device=device_id)
        try:
            self.camera.open()
            # Set the camera resolution to target width and height.
            self.camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

            current_width = self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            current_height = self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(
                f"Camera opened successfully with resolution {current_width}x{current_height}!"
            )
        except Exception as e:
            print(f"Could not open selected device: {e}")

    def start(self) -> None:
        """
        Start the camera stream.
        """
        if not self.camera:
            print("Please select a device first!")
            return
        self.recording = False
        self.frame_count = 0

    def start_recording(self, filepath: str = None) -> None:
        """
        Start recording the camera feed to a video file.

        Args:
            filepath (str, optional): Directory path to save the recording. If not provided or invalid,
                a default folder is created.
        """
        if not self.camera:
            print("Please select a device first!")
            return

        if filepath is None or not os.path.isdir(filepath):
            filepath = self.create_folder()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(filepath, f"recording_{timestamp}.mp4")

        if not filepath:
            print("Error: File path is empty!")
            return

        fps = self.fps if self.fps > 0 else 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            filepath, fourcc, fps, (self.target_width, self.target_height)
        )
        self.record_filepath = filepath

        self.recording = True
        self.frame_count = 0
        print(f"Started recording to {filepath}")

    def stop_recording(self) -> None:
        """
        Stop recording the camera feed.
        """
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"Recording saved to {self.record_filepath}")
            self.record_filepath = None

    def crop_and_resize(
        self,
        image: np.ndarray,
        target_size: Optional[tuple[int, int]] = None,
        border_fraction: float = 0.15,
    ) -> np.ndarray:
        """
        Crop a fraction of the image along the borders while keeping its ratio, and optionally resize
        the cropped image if a target size is provided.

        Args:
            image (np.ndarray): Image to modify.
            target_size (Optional[tuple[int, int]]): Tuple (target_width, target_height) to which the
                image will be resized. If None, only cropping is occurs.
            border_fraction (float, optional): Fraction of the image dimensions to crop from each border.
                Is clamped to range [0, 0.49]
                Defaults to 0.15.

        Returns:
            np.ndarray: The modified image.
        """
        # clamp border fraction
        border_fraction = min(max(0, border_fraction), 0.49)
        # Calculate border sizes
        border_x = int(image.shape[0] * border_fraction)
        border_y = int(image.shape[1] * border_fraction)

        # Crop image
        modified_image = image[
            border_x : image.shape[0] - border_x, border_y : image.shape[1] - border_y
        ]

        # If a target size is provided, resize the cropped image
        if target_size is not None:
            modified_image = cv2.resize(modified_image, target_size)

        return modified_image

    def update(self, dt = None) -> Optional[MatLike]:
        """
        Capture and return a frame from the camera feed, update FPS, overlay FPS text, and record if enabled.

        Args:
            dt (float): Unused parameter; frame timing is computed internally.

        Returns:
            Optional[MatLike]: The current RGB frame with FPS overlay, or None on error.
        """
        if not self.camera:
            return None

        try:
            frame = self.camera.read_frame()
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

        time_now = time.time()
        dt = time_now - self.time_prev
        self.fps = 1.0 / dt if dt > 0 else 0
        self.time_prev = time_now

        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.frame_count += 1
        self.current_frame = self.crop_and_resize(
            image=self.current_frame,
            target_size=(self.target_width, self.target_height),
            border_fraction=self.border_fraction,
        )

        if self.recording and self.video_writer is not None:
            # Convert color back to BGR
            self.video_writer.write(cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))

        return self.current_frame

    def create_folder(self) -> str:
        """
        Create a folder on the Desktop named with the current date.

        Returns:
            str: The created folder path.
        """
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        folder_name = datetime.datetime.now().strftime("%Y-%m-%d")
        folder_path = os.path.join(desktop, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def save_screenshot(self, filepath: str = None) -> bool:
        """
        Save a screenshot of the current frame.

        Args:
            filepath (str, optional): Directory path to save the screenshot.

        Returns:
            bool: True if saving succeeded, False otherwise.
        """
        saved = False
        if filepath is None:
            return saved

        if self.current_frame is not None:
            now = datetime.datetime.now()
            filename = os.path.join(
                filepath, f"screenshot_{now.strftime('%Y%m%d_%H%M%S')}.png"
            )

            try:
                cv2.imwrite(
                    filename,
                    cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR),
                )
                saved = True
                print(f"Screenshot saved to {filename}")
            except Exception as e:
                print(f"Failed to save image: {e}")

        return saved



if __name__=="__main__":
    DEVICE_ID = 0 # USB ID
    img_w = 640 # Target FULLWIDTH
    img_h = 480 # Target FULLHEIGHT
    brd_frac_crop = 0.15 

    gelsight = GelSightMini(
        device_idx=DEVICE_ID,
        target_width=img_w,
        target_height=img_h,
        border_fraction=brd_frac_crop)

    while True:
        img = gelsight.update()

        if img.shape[0] > 0:
            imbgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame_rgb', imbgr)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
