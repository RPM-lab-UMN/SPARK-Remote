import time
# import math
import numpy as np
# from check_pickle import load_trajectory
from ur5.ur5 import RobotController
import Pyro5.api
# from matplotlib import pyplot as plt
# import cv2

# Initialize the RobotController for Lightning
lightning = RobotController('lightning', robot_ip='10.33.55.90', need_control=True, need_gripper=True)

CONTROL_FREQUENCY = 15 # Hz, matches data_collection.py step_hz
STEP_DT = 1.0 / CONTROL_FREQUENCY
EXECUTION_NUM = 16  # number of actions to execute in one step
START_OFFSET = 0  # number of initial actions in the chunk to skip (for testing; trying to fix initial jerk back)

# reset to home position
lightning.go_home()

@Pyro5.api.expose
class ControllerInterface:
    def __init__(self):
        self.step_count = 0

    def get_robot_state(self):
        """Gets the current robot state (joints, gripper, EEF pose, FT data)."""
        EEF_state = lightning.get_eff_pose()
        joint_state = lightning.get_joint_angles()
        gripper_state = lightning.get_current_position() # Get raw value (0-255)
        ft_data = lightning.get_tcp_force()

        return {
            "joint_positions": joint_state,
            "eef_pose": {
                "position": EEF_state[:3],
                "orientation_rpy": EEF_state[3:]
            },
            "gripper_state": gripper_state, # Return raw 0-255 value
            "ft_data": ft_data
        }
    
    def step(self, data_dict):  # data_dict: {'type': 'action', 'data': [...], 'step': int}
        """Executes EXECUTION_NUM action sub-steps and returns collected data + final state."""
        action_chunk = data_dict["data"] # action_list 
        current_policy_step = data_dict["step"] # Get main step number from caller
        sub_step_data = [] # List to store data for each sub-step execution
        current_state = {} # Variable to store the state *before* each sub-step command
        
        # --- OFFSET LOGIC ---
        # Total available actions in the chunk
        total_available = len(action_chunk) 
        # Calculate the actual range to execute
        start_idx = START_OFFSET
        end_idx = min(start_idx + EXECUTION_NUM, total_available)
        # Extract only the slice we want to execute
        action_chunk_subset = action_chunk[start_idx:end_idx]
        actual_execution_num = len(action_chunk_subset)
        
        overall_start_time = time.time()
        
        # for i in range(EXECUTION_NUM):
        for i in range(actual_execution_num):
            # print(f"Executing sub-step {i+1}/{EXECUTION_NUM} of control step {current_policy_step}")
            sub_step_start_time = time.time() # Timestamp for this specific sub-step

            # 1. Get current state *before* commanding this sub-step action
            current_state = self.get_robot_state()

            # 2. Get the action for this sub-step
            # action = np.array(action_chunk[i]) # Should be shape (7,)
            action = np.array(action_chunk_subset[i])
            arm_action = list(action[:6])
            gripper_action_normalized = action[6] # Expecting normalized [0, 1] from policy
            gripper_action_int = int(np.clip(gripper_action_normalized * 255.0, 0, 255)) # Scale to 0-255            

            # 3. Store data for saving *before* executing
            # Store the state observed before the action & the action
            sub_step_data.append({
                "timestamp": sub_step_start_time,
                "state_before_action": current_state, # Full state dict
                "spark_command_angles": arm_action,
                "spark_command_gripper": gripper_action_normalized # Save normalized action
            })

            # 4. Send command to UR5
            # Don't command robot on the very first "step" call if it's just for getting initial state
            # (Assuming step 0 in pi0_deploy sends a dummy action chunk for init state)
            if current_policy_step > 0: # Avoid command on first policy step
                 lightning.servoJ(arm_action)
                 lightning.gripper_open(gripper_action_int)

            # 5. Maintain control frequency
            elapsed_time = time.time() - sub_step_start_time
            sleep_time = STEP_DT - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Get the *final* robot state after the last sub-step
        final_state = self.get_robot_state()
        overall_end_time = time.time()
        print(f"Controller policy step {current_policy_step} took {overall_end_time - overall_start_time:.4f}s total for {actual_execution_num} sub-steps.")

        self.step_count += 1 # Increment main step count

        # Return the collected sub-step data and the final state
        return {
            "sub_step_data": sub_step_data,
            "final_state": final_state
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