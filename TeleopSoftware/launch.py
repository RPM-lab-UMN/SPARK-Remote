#! /bin/env python3

import time
from functools import partial
import tkinter as tk
from tkinter import ttk
import time

from UR.arms import *
from launch_helpers.opt import UR5eForceControl
from launch_helpers.run import *
from launch_helpers.tk_functions import *

# import rospy
# from std_msgs.msg import Float32MultiArray, String, Bool, Float32, Int32
# Update to ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Bool, Float32, Int32

class GUI(Node):
    def __init__(self):
        super().__init__('gui_node')
        pass

    def main(self):
        # rospy.init_node('Main', anonymous=True)
        # rclpy.init()
        self.ros_data = {}
        control_modes = {}


        # store the data in a global variable so it can be accessed from the main loop
        # rospy.Subscriber("/SpaceMouseThunder", Float32MultiArray, lambda data: globals().update({'thunder_data': data.data}))
        # rospy.Subscriber("/SpaceMouseThunderLog", Float32MultiArray, lambda data: globals().update({'thunder_data': data.data}))


        thunder_ip = "10.33.55.89"
        lightning_ip = "10.33.55.90"

        arms = ["Thunder", "Lightning"]
        ips = [thunder_ip, lightning_ip]
        enable_control = {
            # "Thunder": False,
            "Thunder": True,
            "Lightning": True,
        }
        URs = UR(arms, ips, enable_grippers=True)
        optimize = UR5eForceControl(URs)

        pubs = dict()
        for arm in arms:
            # pubs[arm+"_reset_estop"] = rospy.Publisher("/reset_estop", Bool, queue_size=10)
            # pubs[arm+"_ft"] = rospy.Publisher(f"/{arm.lower()}_ft", Float32MultiArray, queue_size=10)
            # pubs[arm+"_ft_raw"] = rospy.Publisher(f"/{arm.lower()}_raw_ft_raw", Float32MultiArray, queue_size=10)
            # pubs[arm+"_q"] = rospy.Publisher(f"/{arm.lower()}_q", Float32MultiArray, queue_size=10)
            # pubs[arm+"_cartesian"] = rospy.Publisher(f"/{arm.lower()}_cartesian_eef", Float32MultiArray, queue_size=10)
            # pubs[arm+"_speed"] = rospy.Publisher(f"/{arm.lower()}_speed", Float32MultiArray, queue_size=10)
            # pubs[arm+"_gripper"] = rospy.Publisher(f"/{arm.lower()}_gripper", Float32, queue_size=10)
            # pubs[arm+"_enable"] = rospy.Publisher(f"/{arm.lower()}_enable", Bool, queue_size=10)
            # pubs[arm+"_safety_mode"] = rospy.Publisher(f"/{arm.lower()}_safety_mode", Int32, queue_size=10)
            # # Force offset
            # pubs[arm+"_force_offset"] = rospy.Publisher(f"/{arm.lower()}_force_offset", Float32MultiArray, queue_size=10)
            pubs[arm+"_reset_estop"] = self.create_publisher(Bool, "/reset_estop", 10)
            pubs[arm+"_ft"] = self.create_publisher(Float32MultiArray, f"/{arm.lower()}_ft", 10)
            pubs[arm+"_ft_raw"] = self.create_publisher(Float32MultiArray, f"/{arm.lower()}_raw_ft_raw", 10)
            pubs[arm+"_q"] = self.create_publisher(Float32MultiArray, f"/{arm.lower()}_q", 10)
            pubs[arm+"_cartesian"] = self.create_publisher(Float32MultiArray, f"/{arm.lower()}_cartesian_eef", 10)
            pubs[arm+"_speed"] = self.create_publisher(Float32MultiArray, f"/{arm.lower()}_speed", 10)
            pubs[arm+"_gripper"] = self.create_publisher(Int32, f"/{arm.lower()}_gripper", 10)
            pubs[arm+"_enable"] = self.create_publisher(Bool, f"/{arm.lower()}_enable", 10)
            pubs[arm+"_safety_mode"] = self.create_publisher(Int32, f"/{arm.lower()}_safety_mode", 10)
            # # Force offset
            pubs[arm+"_force_offset"] = self.create_publisher(Float32MultiArray, f"/{arm.lower()}_force_offset", 10)

        colors = ["light blue", "light green"]
        spark_homes = [(+0.000, -1.15192, -2.26893, 0.244346, +1.5708, +0.000), # Thunder Not used
                (+0.000, -2.1293, 2.44346, -3.49066, -1.5708, +0.000)] # Lightning Not used
        # ur_homes = [(-180, -130, 130, -180, -90, 0), # Thunder
        #            (-180, -50, -130, -0, 90, +0)] # Lightning

        ur_homes = [(-180, -130, 130, -180, -90, -90), # Thunder
                    (-180, -50, -130, -0, 90, +90)] # Lightning
        
        col = {}
        homes = {}
        for name, color, ur_home, spark_home in zip(arms, colors, ur_homes, spark_homes):
            col[name] = color
            homes[name] = [angle/180*3.14159 for angle in ur_home]
            homes[name+"_spark"] = spark_home # Not used

        root = tk.Tk()
        root.title("Teleop Control")

        # make the grid cells expand to fill the window
        for i in range(2):
            root.grid_columnconfigure(i, weight=1)
        root.geometry("1600x800")
        # Split into two columns one for each robotic arm:
        # The window should be split down the middle with one arm on each side.
        fields = {}
        for i in range(len(arms)):
            fields[arms[i]] = {}
            row = 0
            label = tk.Label(root, text=arms[i], font=("Helvetica", 40))
            label.grid(row=row, column=i)
            row += 1

            grid = tk.Frame(root)
            grid.grid(row=row, column=i)

            fields[arms[i]]['db_connect'] = tk.Button(grid, text="Dashboard Connect",
                    command=partial(db_connect, arms[i], fields, URs, col))
            fields[arms[i]]['db_connect'].grid(row=0, column=0)

            fields[arms[i]]['db_reset'] = tk.Button(grid, text="Reset E-Stop",
                    command=partial(db_reset, arms[i], fields, URs, col, control_modes, pubs, enable_control))
            fields[arms[i]]['db_reset'].grid(row=0, column=1)

            fields[arms[i]]['connect'] = tk.Button(grid, text=" UR Connect", 
                    command=partial(connect_fun, arms[i], fields, URs, col, control_modes, enable_control))
            # fields[arms[i]]['connect'].grid(row=0, column=2)

            fields[arms[i]]['freedrive'] = tk.Button(grid, text="Free Drive", 
                    command=partial(freedrive_fun, arms[i], fields, URs, col, control_modes))
            fields[arms[i]]['freedrive'].grid(row=0, column=3)

            fields[arms[i]]['home'] = tk.Button(grid, text="Home",
                    command=partial(home_fun, arms[i], fields, URs, col, homes, control_modes))
            fields[arms[i]]['home'].grid(row=0, column=4)
            
            fields[arms[i]]['gripper'] = tk.Button(grid, text="Gripper",
                    command=partial(gripper_fun, arms[i], fields, URs, col))
            fields[arms[i]]['gripper'].grid(row=0, column=5)

            fields[arms[i]]['emergency'] = tk.Button(grid, text="Emergency Stop",
                    command=partial(emergency_stop, arms[i], fields, URs, col, control_modes))
            fields[arms[i]]['emergency'].grid(row=0, column=6)

            # fields[arms[i]]['zero_ft'] = tk.Button(grid, text="Zero FT",
            #         command=partial(zero_ft, arms[i], URs))
            # fields[arms[i]]['zero_ft'].grid(row=0, column=7)


            
            # Add sub-panels for each control mode: tabControl = ttk.Notebook(root)
            modes = ttk.Notebook(root)
            fields[arms[i]]['Spark'] = tk.Frame(modes)
            # fields[arms[i]]['Optimization'] = tk.Frame(modes)
            fields[arms[i]]['Force'] = tk.Frame(modes)
            fields[arms[i]]['SM'] = tk.Frame(modes)
            fields[arms[i]]['VR'] = tk.Frame(modes)
            modes.add(fields[arms[i]]['Spark'] , text='Spark')
            # modes.add(fields[arms[i]]['Optimization'], text='Optimization')
            modes.add(fields[arms[i]]['Force'], text='Force')
            modes.add(fields[arms[i]]['SM'] , text='SpaceMouse')
            modes.add(fields[arms[i]]['VR'], text='VR')

            fields[arms[i]]['run_buttons'] = []
            
            def run_fun(arm, mode):
                if arm not in control_modes:
                    print(f"{arm} not ready")
                    return
                for button in fields[arm]['run_buttons']:
                    button.config(bg=col[arm])

                # End Free Drive
                if arm in freedrive:
                    print(arm + ": End Free Drive")
                    fields[arm]['freedrive'].config(background=col[arm])
                    URs.freeDrive(arm, False)
                    del freedrive[arm]

                # Toggle the button on or off
                if control_modes[arm] == mode:
                    control_modes[arm] = None
                else:
                    control_modes[arm] = mode
                    self.ros_data[arm.lower() + '_change_mode'] = True
                    fields[arm][mode+"_run"].config(bg='orange')
                print(f"{arm}: Run {mode} - {control_modes[arm]}")

                if mode == "Optimization":
                    if control_modes[arm] == mode:
                        optimize.start_ik_thread(arm)
                    else:
                        optimize.end_ik_thread(arm)



            # Spark -----------------------------------------------------------------------------------------------------------------
            # Plot for the Spark data
            height = 600
            width = 600
            center =  (height/2, width/2)
            size = 20
            meter_width = 20
            meter_padding = 2
            meter_target_height = 40
            
            # Spark Run button:
            run = tk.Button(fields[arms[i]]['Spark'], text="Run Spark", bg=colors[i], command=partial(run_fun, arms[i], "Spark"), width=20)
            run.grid(row=0, column=0)
            fields[arms[i]]['Spark_run'] = run
            fields[arms[i]]['run_buttons'].append(run)

            # Additional Run button for Optimization
            run_opt= tk.Button(fields[arms[i]]['Spark'], text="Optimization", bg=colors[i], command=partial(run_fun, arms[i], "Optimization"), width=20)
            run_opt.grid(row=0, column=1)
            fields[arms[i]]['Optimization_run'] = run_opt
            fields[arms[i]]['run_buttons'].append(run_opt)

            # Canvas for the Spark plot
            fields[arms[i]]["hwc"] = height, width, center
            SparkPlot = tk.Canvas(fields[arms[i]]['Spark'], width=width, height=height)
            fields[arms[i]]["point"] = SparkPlot.create_oval(190, 190, 210, 210, fill="blue")
            SparkPlot.create_oval(center[0]-size, center[1]-size, center[0]+size, center[1]+size, fill="", outline="black", width=5)
            SparkPlot.grid(row=1, column=0)
            fields[arms[i]]['Spark_plot'] = SparkPlot

            # Canvas for the z-depth meter
            SparkMeter = tk.Canvas(fields[arms[i]]['Spark'], width=meter_width+2*meter_padding, height=height)
            fields[arms[i]]['Spark_z_meter'] = SparkMeter.create_rectangle(meter_padding, 0, meter_width, height, fill="red")
            SparkMeter.create_rectangle(meter_padding, 300-meter_target_height/2, meter_width, 300+meter_target_height/2, outline="black", width=2)
            # fields[arms[i]]['Spark_z_meter'] = SparkMeter.create_rectangle(meter_padding, 300+10, meter_width, 300-10, fill="green")
            SparkMeter.grid(row=1, column=1)
            fields[arms[i]]['Spark_meter'] = SparkMeter

            # Spark Home button:
            ur_home = tk.Button(fields[arms[i]]['Spark'], text="Home Spark",
                    command=partial(home_fun, arms[i], fields, URs, col, homes, control_modes, "Spark"))
            ur_home.grid(row=2, column=0)
            fields[arms[i]]['Spark_home'] = ur_home

            # Optimization ----------------------------------------------------------------------------------------------------------
            # Optimization Run button:
            # run = tk.Button(fields[arms[i]]['Optimization'], text="Run Optimization", bg=colors[i], command=partial(run_fun, arms[i], "Optimization"), width=100)
            # run.grid(row=0, column=0)
            # fields[arms[i]]['Optimization_run'] = run
            # fields[arms[i]]['run_buttons'].append(run)
            

            # SM --------------------------------------------------------------------------------------------------------------------
            # Text box for the SpaceMouse log data
            SMLog = tk.Text(fields[arms[i]]['SM'], height=10, width=50)
            SMLog.grid(row=1, column=0)
            fields[arms[i]]['SMLog'] = SMLog

            # SpaceMouse Run button:
            run = tk.Button(fields[arms[i]]['SM'], text="Run SpaceMouse", bg=colors[i], command=partial(run_fun, arms[i], "SpaceMouse"), width=100)
            run.grid(row=0, column=0)
            fields[arms[i]]['SpaceMouse_run'] = run
            fields[arms[i]]['run_buttons'].append(run)

            # Invert Roll and Pitch
            invert = tk.Button(fields[arms[i]]['SM'], text="Invert Roll and Pitch", 
                    command=partial(invert_fun, arms, fields, control_modes))
            invert.grid(row=2, column=0)
            fields[arms[i]]['invert'] = invert

            # VR --------------------------------------------------------------------------------------------------------------------
            # VR Run button:
            run = tk.Button(fields[arms[i]]['VR'], text="Run VR", bg=colors[i], command=partial(run_fun, arms[i], "VR"), width=100)
            run.grid(row=0, column=0)
            fields[arms[i]]['VR_run'] = run
            fields[arms[i]]['run_buttons'].append(run)

            VRPlot = tk.Canvas(fields[arms[i]]['VR'], width=width, height=height)
            fields[arms[i]]["vr_point"] = VRPlot.create_oval(190, 190, 210, 210, fill="blue")
            VRPlot.create_oval(center[0]-size, center[1]-size, center[0]+size, center[1]+size, fill="green")
            VRPlot.grid(row=1, column=0)
            fields[arms[i]]['VR_plot'] = VRPlot

            modes.grid(row=6, column=i)
            
            init_button_colors(fields[arms[i]], colors[i])

            # Force -----------------------------------------------------------------------------------------------------------------
            # Force Run button:
            run = tk.Button(fields[arms[i]]['Force'], text="Run Force", bg=colors[i], command=partial(run_fun, arms[i], "Force"), width=100)
            run.grid(row=0, column=0)
            fields[arms[i]]['Force_run'] = run
            zero_ft_button = tk.Button(fields[arms[i]]['Force'], text="Zero FT", command=partial(zero_ft, arms[i], URs))
            zero_ft_button.grid(row=1, column=0)
            fields[arms[i]]['FT_zero'] = zero_ft_button

            # Force Home button:
            ft_home_button = tk.Button(fields[arms[i]]['Force'], text="FT Home",
                    command=partial(ft_home, arms[i], URs, self.ros_data))
            fields[arms[i]]['FT_home'] = ft_home_button
            ft_home_button.grid(row=2, column=0)
            fields[arms[i]]['FT_home'] = ft_home_button



        # ROS Subscribers: --------------------------------------------------------------------------------------------------------
        
        # Startup Button Presses: -------------------------------------------------------------------------------------------------
        # for arm in ["Lightning"]:
        for arm in ["Thunder", "Lightning"]:
            print(f"\tStarting {arm}")
            root.update()
            fields[arm]['db_connect'].invoke()
            time.sleep(1)
            root.update()
            fields[arm]['db_reset'].invoke()
            time.sleep(1)
            
        self.create_subscription(String, "/SpaceMouseThunderLog", self.thunder_sm_log, 10)
        self.create_subscription(String, "/SpaceMouseLightningLog", self.lightning_sm_log, 10)
        self.create_subscription(Float32MultiArray, "/SpaceMouseThunder", self.thunder_sm_data, 10)
        self.create_subscription(Float32MultiArray, "/SpaceMouseLightning", self.lightning_sm_data, 10)
        self.create_subscription(Float32MultiArray, "/Spark_angle/thunder", self.spark_angle_thunder, 10)
        self.create_subscription(Float32MultiArray, "/Spark_angle/lightning", self.spark_angle_lightning, 10)
        self.create_subscription(Bool, "/Spark_enable/thunder", self.spark_enable_thunder, 10)
        self.create_subscription(Bool, "/Spark_enable/lightning", self.spark_enable_lightning, 10)
        
        print("\tStarting Main Loop")
        try:
            while root.winfo_exists():
                root.update()
                # rospy.sleep(0.01)
                # ROS2 Sleep 
                rclpy.spin_once(self, timeout_sec=0.01)
                ros_update(fields, self.ros_data, control_modes, URs, pubs, optimize)
                # fields['Thunder']['Spark_plot'].move(point, 1, 1)
                # fields['Lightning']['Spark_plot'].move(point, -1, -1)
        except tk.TclError:
            print("Window closed")
            pass
        

    def thunder_sm_log(self, data):
        self.ros_data['thunder_sm_log'] = data.data
        print(data.data)
    def lightning_sm_log(self, data):
        self.ros_data['lightning_sm_log'] = data.data
        print(data.data)
    # rospy.Subscriber("/SpaceMouseThunderLog", String, thunder_sm_log) 
    # rospy.Subscriber("/SpaceMouseLightningLog", String, lightning_sm_log)
    # self.create_subscription("/SpaceMouseLightningLog", String, lightning_sm_log)

    def thunder_sm_data(self, data):
        self.ros_data['thunder_sm_data'] = data.data
    def lightning_sm_data(self, data):
        self.ros_data['lightning_sm_data'] = data.data
        # print(data.data)
    # rospy.Subscriber("/SpaceMouseThunder", Float32MultiArray, thunder_sm_data)
    # rospy.Subscriber("/SpaceMouseLightning", Float32MultiArray, lightning_sm_data)
    # self.create_subscription("/SpaceMouseThunder", Float32MultiArray, thunder_sm_data)
    # self.create_subscription("/SpaceMouseLightning", Float32MultiArray, lightning_sm_data)

    def spark_angle_thunder(self, data):
        self.ros_data['thunder_spark_angle'] = data.data
    def spark_angle_lightning(self, data):
        self.ros_data['lightning_spark_angle'] = data.data
        # print(data.data)
    # rospy.Subscriber("/Spark_angle/thunder", Float32MultiArray, spark_angle_thunder)
    # rospy.Subscriber("/Spark_angle/lightning", Float32MultiArray, spark_angle_lightning)
    # self.create_subscription("/Spark_angle/thunder", Float32MultiArray, spark_angle_thunder)
    # self.create_subscription("/Spark_angle/lightning", Float32MultiArray, spark_angle_lightning)

    def spark_enable_thunder(self, data):
        self.ros_data['thunder_spark_enable'] = data.data
    def spark_enable_lightning(self, data):
        self.ros_data['lightning_spark_enable'] = data.data
    # rospy.Subscriber("/Spark_enable/thunder", Bool, spark_enable_thunder)
    # rospy.Subscriber("/Spark_enable/lightning", Bool, spark_enable_lightning)
    # self.create_subscription("/Spark_enable/thunder", Bool, spark_enable_thunder)
    # self.create_subscription("/Spark_enable/lightning", Bool, spark_enable_lightning)

    # def vr_data_thunder(data):
    #     ros_data['thunder_vr_data'] = data.data
    # def vr_data_lightning(data):
    #     ros_data['lightning_vr_data'] = data.data
    # rospy.Subscriber("/VR/thunder", Float32MultiArray, vr_data_thunder)
    # rospy.Subscriber("/VR/lightning", Float32MultiArray, vr_data_lightning)

    # def force_data_thunder(data):
    #     ros_data['thunder_force_ctl'] = data.data
    # rospy.Subscriber("/Force/force_ctl", Float32MultiArray, force_data_thunder)
    # def force_start_thunder(data):
    #     ros_data['lightning_force_start'] = data.data
    # rospy.Subscriber("/Force/start", Bool, force_start_thunder)
    # def force_stop_thunder(data):
    #     ros_data['lightning_force_stop'] = data.data
    # rospy.Subscriber("/Force/stop", Bool, force_stop_thunder)
    # def force_replay_thunder(data):
    #     ros_data['thunder_replay_eef'] = data.data
    # rospy.Subscriber("thunder_replay_eef", Float32MultiArray, force_replay_thunder)
    # def actions_thunder(data):
    #     ros_data['thunder_actions'] = data.data
    #     print(data.data)
    # rospy.Subscriber("/actions", Float32MultiArray, actions_thunder)

    # def offsets_data(data):
    #     ros_data['offsets'] = data.data
    # rospy.Subscriber("/Force/offsets", Float32MultiArray, offsets_data)
    # def grasp_offsets_data(data):
    #     ros_data['grasp_offsets'] = data.data
    # rospy.Subscriber("/Force/grasp_offsets", Float32MultiArray, grasp_offsets_data)


if __name__ == '__main__':
    rclpy.init()
    gui = GUI()
    gui.main()
    gui.destroy_node()
    rclpy.shutdown()
