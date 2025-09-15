import pyrealsense2 as rs
import cv2
import numpy as np

def get_serial_numbers():
    context = rs.context()
    devices = context.query_devices()
    serial_numbers = []
    for device in devices:
        serial_numbers.append(device.get_info(rs.camera_info.serial_number))
    return serial_numbers

def get_images(serial_num):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_num)
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not depth_frame or not color_frame:
        print(depth_frame, color_frame)
        return
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image

def get_valid_configs(serial_num):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_num)
    pipeline.start(config)
    device = pipeline.get_active_profile().get_device()
    sensor = device.first_depth_sensor()
    depth_scale = sensor.get_depth_scale()
    valid_configs = []
    for profile in device.query_sensors()[0].get_stream_profiles():
        if profile.is_video_stream_profile():
            valid_configs.append(profile.as_video_stream_profile())
    return valid_configs
    

if __name__ == '__main__':
    sn = get_serial_numbers()
    for i, s in enumerate(sn):
        print(f"Camera {i}: {s}")
        # print(get_valid_configs(s))
        # depth_image, color_image = get_images(s)
        # cv2.imshow('Depth', depth_image)
        # cv2.imshow('Color', color_image)
        # cv2.waitKey(0)