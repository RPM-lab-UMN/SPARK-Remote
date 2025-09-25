import tkinter as tk
from std_msgs.msg import Bool
import random

freedrive = {}
gripper = {}

buttons = ['connect', 'freedrive', 'gripper', 'home']


# Dashboard functions
def db_connect(arm, fields, URs, colors):
    if URs.init_dashboard(arm):
        fields[arm]['db_reset'].config(state=tk.NORMAL)
        fields[arm]['db_reset'].config(background="pink")
        fields[arm]['db_connect'].config(background='grey')
        return True
    return False

def db_reset(arm, fields, URs, colors, control_modes, pubs, enable_control):
    URs.get_dashboard(arm).unlockProtectiveStop()
    URs.get_dashboard(arm).close_popup()
    connect_fun(arm, fields, URs, colors, control_modes, enable_control)
    control_modes[arm] = None
    for button in fields[arm]['run_buttons']:
        button.config(bg=colors[arm])
    pubs[arm+"_reset_estop"].publish(Bool(data=True))
    print(arm + ": Reset E-Stop")


# UR Arm functions
def init_button_colors(fields, color):
    fields['db_connect'].config(state=tk.NORMAL)
    fields['db_connect'].config(bg=color)
    fields['db_reset'].config(state=tk.DISABLED)
    fields['db_reset'].config(bg='grey')
    for button in buttons:
        fields[button].config(state=tk.DISABLED)
        fields[button].config(bg='grey')
    fields['emergency'].config(state=tk.DISABLED)
    fields['emergency'].config(bg='grey')
    fields['connect'].config(state=tk.NORMAL)
    fields['connect'].config(bg=color)
    

global sm_start_pose
def connect_fun(arm, fields, URs, colors, control_modes, enable_control):
    if URs.init_arm(arm, enable_control=enable_control):
        for button in buttons:
            fields[arm][button].config(state=tk.NORMAL)
            fields[arm][button].config(background=colors[arm])
        fields[arm]['emergency'].config(state=tk.NORMAL)
        fields[arm]['emergency'].config(bg='red')
        if arm in freedrive:
            del freedrive[arm]
            freedrive_fun(arm, fields, URs, colors, control_modes)
        fields[arm]['connect'].config(background='grey')
        # fields[arm]['connect'].config(state=tk.DISABLED)

def freedrive_fun(arm, fields, URs, colors, control_modes):
    # Print arms 'cartesian' position
    print(arm + ": " + str(URs.get_receive(arm).getActualTCPPose()))
    print(arm + ": " + str(URs.get_receive(arm).getActualQ()))
    for button in fields[arm]['run_buttons']:
        button.config(bg=colors[arm])
    if control_modes[arm]:
        print(arm + ": End " + control_modes[arm])
        control_modes[arm] = None

    if arm in freedrive:
        print(arm + ": End Free Drive")
        fields[arm]['freedrive'].config(background=colors[arm])
        URs.freeDrive(arm, False)
        del freedrive[arm]
    else:
        print(arm + ": Free Drive")
        fields[arm]['freedrive'].config(background='orange')
        URs.freeDrive(arm, True)
        freedrive[arm] = True

def gripper_fun(arm, fields, URs, colors):
    if URs.enable_grippers[arm] == False:
        print(arm + ": No Gripper Enabled")
        return
    
    if arm in gripper:
        print(arm + ": Close Gripper")
        URs.get_gripper(arm).set(255)
        del gripper[arm]
    else:
        print(arm + ": Open Gripper")
        URs.get_gripper(arm).set(0)
        gripper[arm] = True

def home_fun(arm, fields, URs, colors, homes, control_modes, pos=None):
    control_modes[arm] = None
    print(arm)
    for button in fields[arm]['run_buttons']:
        button.config(bg=colors[arm])
    if arm in freedrive:
        freedrive_fun(arm, fields, URs, colors, control_modes)
    print(arm + ": Home")
    URs.get_gripper(arm).set(0)
    # print(URs.get_receive(arm).getActualQ())
    URs.stop(arm)
    if pos == "Spark":
        URs.moveJ(arm, (homes[arm+"_spark"], 0.5, 0.5))
    else:
        URs.moveJ(arm, (homes[arm], 0.5, 0.5))
    zero_ft(arm, URs)


def emergency_stop(arm, fields, URs, colors, control_modes):
    control_modes[arm] = None
    for button in fields[arm]['run_buttons']:
        button.config(bg=colors[arm])
    URs.triggerProtectiveStop(arm)
    # URs.get_control(arm).powerOff()
    print(arm + ": Emergency Stop")

#partial(invert_fun, arms[i], fields, control_modes)
def invert_fun(arms, fields, control_modes):
    if "SM_Invert" in control_modes:
        control_modes["SM_Invert"] = None
        for arm in arms:
                fields[arm]['invert'].config(bg='grey')
                fields[arm]['invert'].text = 'Currently Not Inverted'
    else:
        control_modes["SM_Invert"] = 'invert'
        for arm in arms:
            fields[arm]['invert'].config(bg='orange')
            fields[arm]['invert'].text = 'Currently Inverted'
    print(arm + ": Invert=" + str(control_modes[arm] == 'invert'))

def zero_ft(arm, URs):
    URs.zeroFtSensor(arm)

def ft_home(arm, URs, ros_data):
    off = 0.30
    if arm == 'Thunder':
        # home = [-0.55, 0.10, 0.5, -1.5705949183832149, 0.0, 0.0]
        home = [0.3, 0.10- off, 0.5, -1.5705949183832149, 0.0, 0.0]
    elif arm == 'Lightning':
        # home = [0.55, 0.17, 0.5, -1.5705949183832149, 0.0, 0.0]
        home = [-0.3, 0.17+ off, 0.5, -1.5705949183832149, 0.0, 0.0]
    URs.moveL(arm, (home, 0.5, 0.5, False))
    if arm == "Lightning":
        cartesian = URs.get_receive(arm).getActualTCPPose()
        if 'offsets' in ros_data:
            dx = ros_data['offsets'][0]
            dz = ros_data['offsets'][1]
        else:
            dx = 0
            dz = 0
            print("No offsets")
        if False: # Random demonstration
            var = 0.04
            dx = random.uniform(-var, var)
            dz = random.uniform(-var, var)
        
        if 'grasp_offsets' in ros_data:
            grasp_dx = ros_data['grasp_offsets'][0]
            grasp_dz = ros_data['grasp_offsets'][1]
        else:
            grasp_dx = 0
            grasp_dz = 0
        cartesian[0] += dx + grasp_dx
        cartesian[2] += dz + grasp_dz
        URs.moveL(arm, (cartesian, 0.5, 0.5, False))
        print(f"dx: {dx}, dz: {dz}")

   