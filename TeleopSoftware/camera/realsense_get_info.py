import pyrealsense2 as rs
import cv2
import numpy as np

def get_device_info():
    """
    Checks for connected RealSense devices, prints detailed info, and returns a list of their serial numbers.
    """
    context = rs.context()
    devices = context.query_devices()
    serial_numbers = []
    
    print(f"Found {len(devices)} RealSense device(s).")

    if not devices:
        print("Please check your camera connections.")
        return []

    for i, device in enumerate(devices):
        serial_number = device.get_info(rs.camera_info.serial_number)
        serial_numbers.append(serial_number)
        name = device.get_info(rs.camera_info.name)
        fw_version = device.get_info(rs.camera_info.firmware_version)
        
        print(f"\n--- Device #{i+1} ---")
        print(f"    Name:              {name}")
        print(f"    Serial Number:     {serial_number}")
        print(f"    Firmware Version:  {fw_version}")
        
    return serial_numbers


def get_images(serial_num):
    """
    Captures and returns aligned color and depth images from a specific camera.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_num)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame or not color_frame:
            print(f"Error: Could not get frames from device {serial_num}")
            return None, None
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
        
    finally:
        # Crucial: stop the pipeline to release the camera for the next use
        pipeline.stop()


def get_valid_configs(serial_num):
    """
    Gets available stream configurations for a device's depth sensor.
    Note: This is slow as it initializes a pipeline just for querying.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_num)
    
    try:
        pipeline.start(config)
        device = pipeline.get_active_profile().get_device()
        sensor = device.first_depth_sensor()
        valid_configs = []
        for profile in sensor.get_stream_profiles():
            if profile.is_video_stream_profile():
                vsp = profile.as_video_stream_profile()
                valid_configs.append({
                    "stream": vsp.stream_type(),
                    "format": vsp.format(),
                    "resolution": (vsp.width(), vsp.height()),
                    "fps": vsp.fps()
                })
        return valid_configs
    finally:
        pipeline.stop()
    

if __name__ == '__main__':
    # Get serial numbers and print diagnostic info first
    serial_numbers = get_device_info()

    if not serial_numbers:
        print("\nExiting program.")
    else:
        for i, sn in enumerate(serial_numbers):
            print(f"\n--- Processing Camera {i+1} ({sn}) ---")
            
            # Uncomment the lines below if you need to see the valid configurations
            # print("Valid depth sensor configurations:")
            # configs = get_valid_configs(sn)
            # for cfg in configs:
            #     print(f"  - {cfg}")

            depth_image, color_image = get_images(sn)
            
            if depth_image is not None and color_image is not None:
                # Apply a colormap to the depth image for better visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )

                # Create unique window names for each camera
                depth_window = f'Depth - Cam {i+1} ({sn})'
                color_window = f'Color - Cam {i+1} ({sn})'
                
                cv2.imshow(depth_window, depth_colormap)
                cv2.imshow(color_window, color_image)
                
                print(f"Displaying images for camera {sn}. Press any key to continue...")
                cv2.waitKey(0)
                # Close this camera's windows before opening the next ones
                cv2.destroyWindow(depth_window)
                cv2.destroyWindow(color_window)

        print("\nFinished processing all cameras.")