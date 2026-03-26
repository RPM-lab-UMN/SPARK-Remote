#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Int32
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import pickle
import os
import sys
import termios
import tty
import select
import cv2

from cv_bridge import CvBridge
from camera import RealSenseCamera
from gelsight_helper import GelSightMini
import numpy as np
import pandas as pd

from factors_utils import gen_factors


# ==============================================================
# TODO:
# ===============================================================


# data collection settings
SAVE_DIR = "/data/UR_teleop/putgreeninpot_lid_on/observed_failures/backview/"
os.makedirs(SAVE_DIR, exist_ok=True)
# LANG_INSTRUCTION = "Pick up the blue block"
LANG_INSTRUCTION = "Put the green block in the pot"
# LANG_INSTRUCTION = "Set the cup upright"
USE_FACTORS = True # for Andrew's project; set to False otherwise
# MAKE SURE TO CHANGE VALUES IN gen_factors()
CSV_FILENAME = "train_factors.csv" # save directory for factor values used in demo (Andrew's project)
STEP_HZ = 15
STEP_DT = 1.0 / STEP_HZ
# Replace with your camera serial numbers
# WRIST_CAMERA_SERIAL = '128422270284'  # D405
WRIST_CAMERA_SERIAL = '130322273305'  # D405
SCENE_CAMERA_SERIAL = 'f1380660'  # L515
RESIZE = False # whether to resize images
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 240

# ========== function：Non-blocking keyboard input ==========
def get_key():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

def set_cbreak_mode():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old_settings

def restore_terminal_mode(old_settings):
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ========== ROS2 StateSubscriber ==========
class StateSubscriber(Node):
    def __init__(self):
        super().__init__('lightning_state_subscriber')

        self.eef_pose = None
        self.joint_positions = None
        self.gripper_value = None
        self.ft_data = None
        self.spark_command_angles = None
        self.spark_command_gripper = None       

        self.create_subscription(Float32MultiArray, '/lightning_cartesian_eef', self.eef_callback, 10)
        self.create_subscription(Float32MultiArray, '/lightning_q', self.q_callback, 10)
        self.create_subscription(Int32, '/lightning_gripper', self.gripper_callback, 10)
        self.create_subscription(Float32MultiArray, '/lightning_ft', self.ft_callback, 10)
        self.create_subscription(Float32MultiArray, '/lightning_spark_command_angles', self.spark_angles_callback, 10)
        self.create_subscription(Float32, '/lightning_spark_command_gripper', self.spark_gripper_callback, 10)        

    def eef_callback(self, msg):
        self.eef_pose = msg.data

    def q_callback(self, msg):
        self.joint_positions = msg.data

    def gripper_callback(self, msg):
        self.gripper_value = msg.data
    
    def ft_callback(self, msg):
        self.ft_data = msg.data

    def spark_angles_callback(self, msg):
        self.spark_command_angles = msg.data

    def spark_gripper_callback(self, msg):
        self.spark_command_gripper = msg.data

# ============= preprocess function ===========
def preprocess_frame(frames):
    if len(frames) < 2:
        return

    pos_list = []
    rpy_list = []
    gripper_list = []

    for frame in frames:
        eef_pose = frame["eef_pose"]
        pos_list.append(eef_pose["position"])
        rpy_list.append(eef_pose["orientation_rpy"])
        gripper_list.append(frame.get("gripper_state", 0))  # Default to 0 if gripper_state is missing

    normalized_gripper_list = [x / 255.0 for x in gripper_list]  # Normalize gripper state to [0, 1]
    actions = compute_eef_action(pos_list, rpy_list, normalized_gripper_list)  # calculated from s{t+1} - s{t} (NOT the actions SPARK sends to the UR5 arms but ideally they are the same)

    # print('actions:', actions)
    
    # only save N-1 frames
    frames = frames[:len(actions)]

    for i in range(len(frames)):
        frames[i]["action"] = actions[i].tolist()

    return frames

def compute_eef_action(pos_list, rpy_list, gripper_list):
    pos = np.array(pos_list)  # shape (N, 3)
    rpy = np.array(rpy_list)  # shape (N, 3)
    gripper = np.array(gripper_list).reshape(-1, 1)  # shape (N, 1)

    d_pos = np.diff(pos, axis=0)  # shape (N-1, 3)
    d_rpy = np.diff(rpy, axis=0)  # shape (N-1, 3)
    d_gripper = np.diff(gripper, axis=0)  # shape (N-1, 1)

    return np.hstack([d_pos, d_rpy, d_gripper])  # shape (N-1, 7)

# ====== extra helper functions (Andrew's project; ignore otherwise) =======
def save_factors(traj_id, factors):
    """Appends the trajectory ID and its associated factors to a CSV file."""
    csv_path = os.path.join(SAVE_DIR, CSV_FILENAME)
    try:
        data = factors.copy()
        data['traj_id'] = traj_id
        df = pd.DataFrame([data])
        # Check if file exists to determine if we need to write the header
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"[INFO] Factors logged to {csv_path}")

    except Exception as e:
        print(f"[ERROR] Failed to log to CSV: {e}")


# ========== main function ==========

rclpy.init()
state_sub = StateSubscriber()
wrist_cam = RealSenseCamera(serial_number=WRIST_CAMERA_SERIAL, width=640, height=480, fps=30)
scene_cam = RealSenseCamera(serial_number=SCENE_CAMERA_SERIAL, width=640, height=480, fps=30)
cams = [wrist_cam, scene_cam]
# cams = [wrist_cam]


# Gelsight
DEVICE_ID = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._GelSight_Mini_R0B_28F5-K4RR_28F5K4RR-video-index0"
img_w = 640
img_h = 480
brd_frac_crop = 0.15
gelsight = GelSightMini(
    device_idx=DEVICE_ID,
    target_width=img_w,
    target_height=img_h,
    border_fraction=brd_frac_crop)

# # Start MultiThreadedExecutor in a background thread
# executor = MultiThreadedExecutor(num_threads=4)
# executor.add_node(state_sub)
# executor_thread = threading.Thread(target=executor.spin, daemon=True)
# executor_thread.start()

def main():
    recording = False

    print("Press 's' to start recording, 'e' to end recording, 'q' to quit.")

    old_terminal_settings = set_cbreak_mode()
    print("[INFO] Number of trajectories collected:", len([f for f in os.listdir(SAVE_DIR) if f.endswith('.pkl')]))
    if USE_FACTORS:
        current_factors = gen_factors()

    try:
        while True:
            key = get_key()

            if key == 's' and not recording:
                print("[INFO] Start recording...")
                recording = True
                frames = []
                time.sleep(0.2)
                step_end_time = time.time()

            elif key == 'e' and recording:
                print("[INFO] Stop recording. Saving...")
                # frames processing: (1) gripper state normalization, (2) compute eef action
                frames = preprocess_frame(frames)
                traj_id = int(time.time())
                save_trajectory(frames, traj_id)
                recording = False
                print("[INFO] Number of trajectories collected:", len([f for f in os.listdir(SAVE_DIR) if f.endswith('.pkl')]))
                if USE_FACTORS:
                    save_factors(traj_id, current_factors) # Andrew's project; ignore otherwise
                    current_factors = gen_factors()
                time.sleep(0.2)

            elif key == 'q':
                print("[INFO] Quit program.")
                break

            if recording:
                for i in range(3): # 3 times to ensure we get the latest state, still need to be improved.
                    rclpy.spin_once(state_sub, timeout_sec=0.01)
                # print('time:', now-last_step_time)
                # print(state_sub.eef_pose, state_sub.joint_positions, state_sub.gripper_value)

                # make sure to record at the specified frequency--15hz
                now = time.time()
                if now - step_end_time < STEP_DT:
                    time.sleep(STEP_DT - (now - step_end_time))
                    # print('sleep:', STEP_DT - (now - step_end_time))
                frame = collect_one_frame()
                step_end_time = time.time()
                # print('ft data:', state_sub.ft_data)
                if frame is not None:
                    frames.append(frame)

    finally:
        restore_terminal_mode(old_terminal_settings)
        for cam in cams:
            cam.stop()
        state_sub.destroy_node()
        rclpy.shutdown()


def collect_one_frame():
    if (
        state_sub.eef_pose is None or
        state_sub.joint_positions is None or
        state_sub.gripper_value is None or
        state_sub.ft_data is None or
        state_sub.spark_command_angles is None or
        state_sub.spark_command_gripper is None
    ):
        return None

    wrist_color_image, wrist_depth_frame = wrist_cam.get_frames()  # BGR
    scene_color_image, scene_depth_frame = scene_cam.get_frames()  # BGR
    if RESIZE:
        wrist_color_image = cv2.resize(wrist_color_image, (RESIZE_WIDTH, RESIZE_HEIGHT))
        scene_color_image = cv2.resize(scene_color_image, (RESIZE_WIDTH, RESIZE_HEIGHT))
    wrist_depth_image = cv2.convertScaleAbs(wrist_depth_frame, alpha=0.03) # depth to 8-bit image
    wrist_color_image = cv2.cvtColor(wrist_color_image, cv2.COLOR_BGR2RGB)
    scene_depth_image = cv2.convertScaleAbs(scene_depth_frame, alpha=0.03) # depth to 8-bit image
    scene_color_image = cv2.cvtColor(scene_color_image, cv2.COLOR_BGR2RGB)
    # gelsight_depth_image = gelsight.update()

    timestamp = time.time()
    frame = {
        "timestamp": timestamp,
        "rgb_wrist": wrist_color_image,
        "rgb_scene": scene_color_image,
        "depth_wrist": wrist_depth_image,
        "depth_scene": scene_depth_image,
        # "gelsight_scene": gelsight_depth_image,
        "joint_positions": state_sub.joint_positions,
        "eef_pose": {
            "position": state_sub.eef_pose[:3],
            "orientation_rpy": state_sub.eef_pose[3:]
        },
        "gripper_state": state_sub.gripper_value,
        "ft_data": state_sub.ft_data,
        "lang_instruction": LANG_INSTRUCTION,
        "spark_command_angles": state_sub.spark_command_angles,
        "spark_command_gripper": state_sub.spark_command_gripper        
    }
    return frame


def save_trajectory(frames, traj_id):
    traj = {
        "meta": {
            "traj_id": traj_id,
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(traj_id)),
        },
        "frames": frames
    }
    save_path = os.path.join(SAVE_DIR, f"traj_{traj_id}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Trajectory saved: {save_path}")


if __name__ == "__main__":
    main()
