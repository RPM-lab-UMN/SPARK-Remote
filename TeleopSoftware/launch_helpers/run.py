import tkinter as tk
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from UR.fk import forward_6
from std_msgs.msg import Float32MultiArray, Float32, Bool, Int32

# ROS functions ------------------------------------------------------------------------------
current_file = os.path.abspath(__file__)
ur_time = 0.001
ur_lookahead_time = 0.05
ur_gain = 200
start_pose = {}
vr_start_pose = {}
homes = {}

spark_enable = {}

offset = [0,0,0,0]

publish_ft = True

def ros_update(fields, ros_data, control_modes, URs, pubs, optimize):
    if 'thunder_sm_log' in ros_data:
        fields['Thunder']['SMLog'].insert(tk.END, ros_data['thunder_sm_log'])
        fields['Thunder']['SMLog'].see(tk.END)
        del ros_data['thunder_sm_log']
    if 'lightning_sm_log' in ros_data:
        fields['Lightning']['SMLog'].insert(tk.END, ros_data['lightning_sm_log'])
        fields['Lightning']['SMLog'].see(tk.END)
        del ros_data['lightning_sm_log']

    for arm in control_modes:
        # Spark control: ------------------------------------------------------------------------
        # Optimization Control: ------------------------------------------------------------------------
        if control_modes[arm] == 'Spark' or control_modes[arm] == 'Optimization':
            if arm.lower()+'_spark_angle' in ros_data:
                angles = ros_data[arm.lower()+'_spark_angle']
                if ros_data[arm.lower() + '_change_mode'] == True:
                    # if arm == "Thunder":
                    #     homes[arm] = [-np.pi, -np.pi/2, 0.0, -np.pi/2, -np.pi, 0.0, 0.0]
                    # elif arm == "Lightning":
                    #     homes[arm] = [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0, 0.0]
                    if arm == "Thunder":
                        homes[arm] = [-np.pi, -np.pi/2, 0.0, -np.pi/2, -np.pi, -1/2*np.pi, 0]
                    elif arm == "Lightning":
                        homes[arm] = [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 1/2*np.pi, 0]
                    dq = [a-u+h for a, u, h in zip(angles, URs.getActualQ(arm), homes[arm])]
                    # print("Diff: ", dq)
                    # print("Home: ", homes[arm])
                    if dq[0] > np.pi:
                        homes[arm][0] = - 2*np.pi
                        # print("Subbed")
                    elif dq[0] < -np.pi:
                        homes[arm][0] = + 2*np.pi
                        # print("Added")
                    # print("Home: ", homes[arm])
                    # dq = [a-u+h for a, u, h in zip(angles, URs.getActualQ(arm), homes[arm])]
                    # print("Diff: ", dq)
                    ros_data[arm.lower() + '_change_mode'] = False
                    
                angles = [angle + homes[arm][i] for i, angle in enumerate(angles)]
                # print(angles)
                if arm == "Lightning":
                    gripper = np.clip((-1.0-angles[6]*2), 0, 1)
                else:
                    gripper = np.clip((2.1-angles[6]*2), 0, 1)
                gripper = round(gripper*10)/10
                
                # Calculate forward kinematics: 
                ur_Q = URs.getActualQ(arm)
                ur_pos = forward_6(ur_Q)[0]
                spark_pos = forward_6(angles[:6])[0]
                height, width, center = fields[arm]['hwc']
                # for angle in spark_pos:
                #     print(angle)
                # print("-----")
                if arm == 'Thunder':
                    x = -spark_pos[1] + ur_pos[1]
                    y = -spark_pos[0] + ur_pos[0]
                elif arm == 'Lightning':
                    x = +spark_pos[1] - ur_pos[1]
                    y = +spark_pos[0] - ur_pos[0]
                z = spark_pos[2] - ur_pos[2]
                x = x*width/2 + center[0]
                y = y*height/2 + center[1]
                z = z*300 + 300
                fields[arm]['Spark_plot'].itemconfig(fields[arm]['point'], fill='red' if z > 0 else 'blue')
                fields[arm]['Spark_plot'].moveto(fields[arm]['point'], x-10, y-10)
                fields[arm]['Spark_meter'].moveto(fields[arm]['Spark_z_meter'], 0, z)
                # fields[arm]['Spark_meter'].resize(fields[arm]['Spark_z_meter'], 0, z)
                # enable_topic = arm.lower() + '_spark_enable'
                enable_topic = 'lightning_spark_enable'
                if control_modes[arm] == 'Spark' or control_modes[arm] == 'Optimization':
                    if enable_topic in ros_data:
                        global spark_enable
                        old_spark_enable = spark_enable[arm] if arm in spark_enable else False
                        spark_enable[arm] = ros_data[enable_topic]
                        if spark_enable[arm] == True:
                            if old_spark_enable == False: # Zero the FT sensor
                                URs.zeroFtSensor(arm)
                                print(f"Zeroed FT sensor on {arm}")
                            if control_modes[arm] == 'Spark':
                                URs.servoJ(arm, (angles[:6], 0.0, 0.0, ur_time, ur_lookahead_time, ur_gain))
                                URs.get_gripper(arm).set(int(gripper*255))
                
                if control_modes[arm] == 'Optimization':
                    # print("Optimization")
                    optimize.set_spark_angle(arm, angles)
                    optimize.set_enable(ros_data[enable_topic])
                    if ros_data[enable_topic]:
                        URs.get_gripper(arm).set(int(gripper*255))
                
                del ros_data[arm.lower()+'_spark_angle'] 

        # SpaceMouse control: ----------------------------------------------------------------
        elif control_modes[arm] == 'SpaceMouse':
            data_name = arm.lower()+"_sm_data"
            if data_name in ros_data:
                data = ros_data[data_name]
                del ros_data[data_name]
                # Velocity control: 
                if arm not in start_pose or ros_data[arm.lower() + '_change_mode'] == True:
                    pose = URs.get_receive(arm).getActualTCPPose()
                    start_pose[arm] = pose[:3], R.from_euler('xyz', pose[3:])
                    ros_data[arm.lower() + '_change_mode'] = False
                else:
                    xyz_scale = 0.005 * 200
                    # rot_scale = 0.01 * 300
                    rot_scale = 0.0
                    if arm == 'Thunder':
                        sm_data = np.array([
                        -data[2], data[0], -data[1], 
                        data[5], -data[4], -data[3]])
                    elif arm == 'Lightning':
                        sm_data = np.array([
                        data[2], -data[0], -data[1], 
                        -data[5], data[4], -data[3]])
                    if "SM_Invert" in control_modes:
                        sm_data[4] = -sm_data[4]
                        sm_data[5] = -sm_data[5]
                    sm_xyz = sm_data[:3] * xyz_scale
                    sm_rpy = sm_data[3:] * rot_scale
                    print(sm_xyz, sm_rpy)
                    # sm_rot = R.from_euler('xyz', sm_rpy)
                    # ee_xyz, ee_rot = start_pose[arm]
                    # new_xyz = sm_xyz + ee_xyz
                    # new_rot = ee_rot * sm_rot  # Correct order of multiplication
                    # new_rpy = new_rot.as_euler('xyz', degrees=False)
                    # print(new_rpy)
                    # new_pose = np.append(new_xyz, new_rpy)
                    # URs.servoL(arm, (new_pose, 0.0, 0.0, ur_time, ur_lookahead_time, ur_gain))

                    URs.speedL(arm, (np.append(sm_xyz, sm_rpy), 2, ur_time))

                    # start_pose[arm] = new_xyz, new_rot
                    if data[6] == 1:
                        URs.get_gripper(arm).set(255)
                    elif data[7] == 1:
                        URs.get_gripper(arm).set(0)

        # VR control: --------------------------------------------------------------------------
        elif control_modes[arm] == 'VR':
            if arm not in start_pose or ros_data[arm.lower() + '_change_mode'] == True:
                pose = URs.get_receive(arm).getActualTCPPose()
                start_pose[arm] = pose[:3], R.from_euler('xyz', pose[3:])
                ros_data[arm.lower() + '_change_mode'] = False
            else:
                if arm.lower()+'_vr_data' in ros_data:
                    data = ros_data[arm.lower()+'_vr_data']
                    del ros_data[arm.lower()+'_vr_data']
                    gripper = data[6]
                    gripper = round(gripper*10)/10
                    xyz_button = data[7]
                    rpy_button = data[8]

                    height, width, center = fields[arm]['hwc']
                    x = max(min(data[9]*width/2 + center[0], width-10), 10)
                    y = max(min(data[10]*height/2 + center[1], height-10), 10)
                    fields[arm]['VR_plot'].moveto(fields[arm]['vr_point'], x-10, y-10)

                    if xyz_button > 0.5 or rpy_button > 0.5:
                        if arm not in vr_start_pose: # Set the start pose
                            vr_start_pose[arm] = data
                        ee_xyz, ee_rot = start_pose[arm]

                        vr_xyz = np.array(data[:3]) - np.array(vr_start_pose[arm][:3])
                        vr_rpy = np.array(data[3:6]) - np.array(vr_start_pose[arm][3:6])

                        # Swap vr_xyz[0] and vr_xyz[1] for both arms
                        vr_xyz = [-vr_xyz[2], vr_xyz[1], vr_xyz[0]]
                        vr_rpy = [-vr_rpy[2], vr_rpy[1], vr_rpy[0]]

                        if arm == 'Thunder':
                            new_xyz = (
                                ee_xyz[0] - vr_xyz[1], 
                                ee_xyz[1] - vr_xyz[0], 
                                ee_xyz[2] - vr_xyz[2])
                            new_rot = ee_rot*R.from_euler('xyz', (-vr_rpy[2], vr_rpy[0], vr_rpy[1]))
                        elif arm == 'Lightning':
                            new_xyz = (
                                ee_xyz[0] + vr_xyz[1], 
                                ee_xyz[1] + vr_xyz[0], 
                                ee_xyz[2] - vr_xyz[2])
                            new_rot = ee_rot*R.from_euler('xyz', (-vr_rpy[2], vr_rpy[0], vr_rpy[1]))
                        
                        if not xyz_button > 0.5: # Move the end effector
                            new_xyz = ee_xyz
                        if not rpy_button > 0.5: # Rotate the end effector
                            new_rot = ee_rot

                        new_rpy = R.as_euler(new_rot, 'xyz')
                        URs.servoL(arm, (np.append(new_xyz, new_rpy), 0.0, 0.0, ur_time, ur_lookahead_time, ur_gain))
                        URs.get_gripper(arm).set(int(gripper*255))

                    else: # Reset the start pose
                        if arm in vr_start_pose:
                            del vr_start_pose[arm]
                        if arm in start_pose:
                            del start_pose[arm]

        # Force controller: -------------------------------------------------------------------
        elif control_modes[arm] == 'Force':
            # print(URs.getActualQ(arm))
            if arm == "Lightning":
                if arm.lower()+'_force_start' in ros_data:
                    task_frame = [0, 0, 0, 0, 0, 0]
                    selection_vector = [0, 1, 0, 0, 0, 0]
                    wrench_up = [0, 15, 0, 0, 0, 0]
                    force_type = 2
                    limits = [2, 2, 2, 1, 1, 1]
                    URs.forceMode(arm, (task_frame, selection_vector, wrench_up, force_type, limits))
                    del ros_data[arm.lower()+'_force_start']
                if arm.lower()+'_force_stop' in ros_data:
                    URs.stop(arm)
                    pose = URs.get_receive(arm).getActualTCPPose()
                    pose[1] -= 0.1
                    URs.moveL(arm, (pose, 0.5, 0.5, False))
                    fields[arm]['FT_home'].invoke()
                    fields[arm]['FT_zero'].invoke()
                    del ros_data[arm.lower()+'_force_stop']
                    
            if arm == "Thunder":
                if arm not in start_pose or ros_data[arm.lower() + '_change_mode'] == True:
                    fields[arm]['FT_home'].invoke()
                    fields[arm]['FT_zero'].invoke()
                    start_pose[arm] = URs.get_receive(arm).getActualTCPPose()
                    ros_data[arm.lower() + '_change_mode'] = False
                
                if True: #Teleop
                    if arm.lower()+'_force_ctl' in ros_data:
                        data = ros_data[arm.lower()+'_force_ctl']
                        del ros_data[arm.lower()+'_force_ctl']
                        scale = 0.05
                        offset = [0, 0, 0, 0]
                        new_pose = start_pose[arm].copy()
                        new_pose[0] += data[0]*scale + offset[0]
                        new_pose[2] += data[1]*scale + offset[1]
                        # print("Force: ", new_pose)
                        URs.servoL(arm, (new_pose, 0.0, 0.0, ur_time, ur_lookahead_time, ur_gain))
                        pubs[arm+"_force_offset"].publish(Float32MultiArray(data=offset))
                elif False: #Replay
                    if 'thunder_replay_eef' in ros_data:
                        data = ros_data['thunder_replay_eef']
                        del ros_data['thunder_replay_eef']
                        URs.servoL(arm, (data, 0.0, 0.0, ur_time, ur_lookahead_time, ur_gain))
                elif True: #Policy
                    if 'thunder_actions' in ros_data:
                        print("Policy: ", ros_data['thunder_actions'])
                        data = ros_data['thunder_actions']
                        del ros_data['thunder_actions']
                        start_pose[arm][0] += data[0]
                        start_pose[arm][2] += data[1]
                        # start_pose[arm][0] = data[0]
                        # start_pose[arm][2] = data[1]
                        print("Policy: ", start_pose[arm])
                        URs.servoL(arm, (start_pose[arm], 0.0, 0.0, ur_time, ur_lookahead_time, ur_gain))

        # Always: ----------------------------------------------------------------------
        if publish_ft:
            norm_ft = URs.get_receive(arm).getActualTCPForce()
            pubs[arm+"_ft"].publish(Float32MultiArray(data=norm_ft))
            
            raw_ft = URs.get_receive(arm).getFtRawWrench()
            pubs[arm+"_ft_raw"].publish(Float32MultiArray(data=raw_ft))

            q = URs.getActualQ(arm)
            pubs[arm+"_q"].publish(Float32MultiArray(data=q))

            cartesian = URs.get_receive(arm).getActualTCPPose()
            pubs[arm+"_cartesian"].publish(Float32MultiArray(data=cartesian))

            speed = URs.get_receive(arm).getActualTCPSpeed()
            pubs[arm+"_speed"].publish(Float32MultiArray(data=speed))

            gripper = URs.get_gripper(arm).get_current_position()
            pubs[arm+"_gripper"].publish(Int32(data=gripper))

            enable = spark_enable[arm] if arm in spark_enable else False
            pubs[arm+"_enable"].publish(Bool(data=enable))
            
            saftey_mode = URs.get_receive(arm).getSafetyMode()
            pubs[arm+"_safety_mode"].publish(Int32(data=saftey_mode))
            # print(raw_ft)





