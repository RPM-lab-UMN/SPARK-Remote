import time
import math
import numpy as np
from check_pickle import load_trajectory
from ur5.ur5 import RobotController

FREQUENCY = 15.0  # Hz

# Initialize the RobotController for Lightning
lightning = RobotController('lightning', robot_ip='10.33.55.90', need_control=True, need_gripper=True)



traj = load_trajectory("/data/UR_teleop/traj_1758239867.pkl")

meta = traj.get('meta', {})
frames = traj.get('frames', [])

print("\n===== Trajectory Summary =====")
print(f"Trajectory ID: {meta.get('traj_id', 'N/A')}")
print(f"Date: {meta.get('date', 'N/A')}")
print(f"Total frames: {len(frames)}")


# go to initial position
initial_joint_positions = frames[0]['joint_positions']
lightning.moveJ(initial_joint_positions, 0.5, 0.1)


traj = load_trajectory("/data/UR_teleop/traj_1758239867.pkl")


for i in range(len(frames)):
    start = time.time()
    arm_action = list(frames[i]['spark_command_angles'])
    gripper_action = frames[i]['spark_command_gripper']
    print(f"Step {i}: arm action = {arm_action}")
    print(f"Step {i}: gripper action = {gripper_action}")

    lightning.servoJ(arm_action)
    lightning.gripper_open(int(gripper_action*255))  # scale to 0-255

    elapsed = time.time() - start
    sleep_time = max(0.0, (1.0 / FREQUENCY) - elapsed)
    time.sleep(sleep_time)