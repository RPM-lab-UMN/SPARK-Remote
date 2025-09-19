from ur5.ur5 import RobotController
import numpy as np
import time

# Initialize the RobotController for Lightning
lightning = RobotController('lightning', robot_ip='10.33.55.90', need_control=True, need_gripper=False)
lightning = RobotController('thunder', robot_ip='10.33.55.89', need_control=True, need_gripper=False)
print("Lightning Initialized")

def print_curr_positions():
    '''   
        Lightning Home Position
        Current EEF Pose: [-0.6499776767939542, 0.23253254732712947, 0.11578951062572167, -1.0675595654177858, -1.4004035252144698, -1.3957112013774338]
        Joint angles: [3.141575574874878, -2.356199403802389, -1.5707868337631226, -1.5708042583861292, 1.0699452104745433e-05, 0.5236055850982666]

        Thunder Home Position
        Current EEF Pose: [0.8256396702917271, 0.24515367580317599, 0.2135475310309233, -1.1468989590511633, 1.2864694831884216, 1.3221565439533352]
        Joint angles: [3.157637357711792, -0.5073470634273072, 0.927502457295553, -2.0310217342772425, 0.021031498908996582, 0.16951298713684082]
    '''
    curr_eef_pose = lightning.get_eff_pose()
    curr_joint_angles = lightning.get_joint_angles()
    print(f"\n[ROBOT] Current EEF Pose: {curr_eef_pose}")
    print(f"[ROBOT] Joint angles: {curr_joint_angles}")

FLAG_FREEDRIVE = True
if FLAG_FREEDRIVE:
    print_curr_positions()
    lightning.freeDrive()
    print_curr_positions()

FLAG_MOVE_HOME = True
if FLAG_MOVE_HOME:    
    input("Press Enter to go to Home Position & close gripper...")
    lightning.go_home()
    lightning.gripper_close()
    print(f"Joint angles: {lightning.get_joint_angles()}")

FLAG_MOVE_L = False
if FLAG_MOVE_L:
    new_pose = curr_eef_pose + np.array([0.0, -0.3, 0.0, 0, 0, 0])
    lightning.moveL(new_pose)

FLAG_MOVE_J = False
if FLAG_MOVE_J:
    new_joints = curr_joint_angles + np.array([0, 0, 0, 0, 0.0, -1.57])
    print("Moving to new joint configuration...", new_joints)
    lightning.moveJ(new_joints, async_flag=True)


while True:
    curr_eef_pose = lightning.get_eff_pose()
    curr_joint_angles = lightning.get_joint_angles()
    print(f"Current EEF Pose: {curr_eef_pose}")
    print(f"Joint angles: {curr_joint_angles}")
    time.sleep(1)