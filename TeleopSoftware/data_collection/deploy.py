import time
import math
import numpy as np
from check_pickle import load_trajectory
from ur5.ur5 import RobotController
import Pyro5.api
from matplotlib import pyplot as plt
import cv2

# Initialize the RobotController for Lightning
lightning = RobotController('lightning', robot_ip='10.33.55.90', need_control=True, need_gripper=True)

CONTROL_FREQUENCY = 15

frame_buffer = []

# reset to home position
# TODO: set to your home position
lightning.go_home()

EXECUTION_NUM = 15  # number of actions to execute in one step


@Pyro5.api.expose
class ControllerInterface:
    def __init__(self):
        self.step_count = 0

    def step(self, data_dict):  # data_dict: {'action': [...], 'step': int}
        action_chunk = data_dict["data"] # action_list 
        print('steps:', self.step_count)
        for i in range(EXECUTION_NUM):
            # print(f"Executing sub-step {i+1}/{EXECUTION_NUM} of control step {self.step_count}")
            start_time = time.time()
            action = action_chunk[i]
            action = np.array(action) 
            arm_action = list(action[:6])
            gripper_action = action[6]

            # print("Action: ", arm_action, gripper_action)

            # TODO: move the robot according to the action
            if self.step_count != 0:
                lightning.servoJ(arm_action)
                # lightning.servoJ(arm, (angles[:6], 0.0, 0.0, ur_time, ur_lookahead_time, ur_gain))
                lightning.gripper_open(int(gripper_action*255))  # scale to 0-255
            #     # time.sleep(0.5)  # small delay to ensure command is sent

            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / CONTROL_FREQUENCY:
                time.sleep(1 / CONTROL_FREQUENCY - elapsed_time)

        print(f"control Step {self.step_count} | Received action: {action}")

        # get robot and gripper state
        robot_EEF_state = lightning.get_eff_pose()
        robot_joint_state = lightning.get_joint_angles()
        gripper_state = [lightning.get_current_position()]
        # robot_state = robot_joint_state + robot_EEF_state + gripper_state
        robot_state = robot_joint_state + gripper_state  # without EEF state
        
        self.step_count += 1
        
        return {
            "robot_state": robot_state,
            # "gripper_state": gripper_state,
            "step": self.step_count
        }

# Pyro5 server
daemon = Pyro5.api.Daemon(host="localhost")
# daemon = Pyro5.api.Daemon(host="10.131.235.100", port = 9090)  # Make a Pyro daemon
ns = Pyro5.api.locate_ns()  # Locate the name server
uri = daemon.register(ControllerInterface)
ns.register("pi0_controller", uri)  # Register the object with the name server
print("Controller server running at:")
print(uri)
daemon.requestLoop()