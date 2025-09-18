# SPARK Remote
This folder contains all the software required for running the SPARK platform and connecting to a Universal Robots arm. 

## Conda Environment
TODO

## Setup
1. Follow setup steps for Mamba + ROS2 https://robostack.github.io/GettingStarted.html
2. Run SPARK-Remote/install.sh (./install.sh) (Note: if using L515, we had to use pyrealsense2==2.54.2.5684)

### Note about realsense-viewer
realsense-viewer is a helpful tool for visualizing realsense cameras but I believe L515 is incompatible with Ubuntu 24.04 because the documentation says the latest librealsense SDK that supports L515 is v2.54.2 but this is incompatible with Ubuntu 24.04. Regardless, it is not necessary for this project.  
Source: https://github.com/IntelRealSense/librealsense/releases

## Key Files
- [./launch.py](./launch.py) > The graphical interface script responsible for connecting to the UR arms via RTDE. This subscribes to SPARK and other ROS2 topics for teleoperation, and publishes information about the arms. Helper files are found in [./launch_helpers/](./launch_helpers/). 
  - [./launch_helpers/tk_functions.py](./launch_helpers/tk_functions.py) > Contains callback functions for button presses. 
  - [./launch_helpers/run.py](./launch_helpers/run.py) > The a function callled once each time in the execution loop for teleoperation code.
  - [./launch_helpers/opt.py](./launch_helpers/opt.py) > This handles the code for the compliant SPARK controller (SPARK-FC).
  - [./launch_helpers/recorder.py](./launch_helpers/recorder.py) & [playback.py](./launch_helpers/playback.py)> These can be used to record and playback teleoperations and their sensor readings. These record to ROS2 bags. 
  - [./launch_helpers/check_topics.py](./launch_helpers/check_topics.py) > Use this script to get alerts if any any of its topics stop being published. Useful to ensure cameras are still working. 
- [./launch_devs.py](./launch_devs.py) Launches controller scripts for SPARKS and other USB perefrials (not haptic gloves). This will automatically detect the devices. 
- [./launch_haptic_devs.py](./launch_haptic_devs.py) Launches the controllers for the haptic gloves. 

## Starting SPARK-Remote
1. Turn on UR5e arms
2. Run [./launch_devs.py](./launch_devs.py) and make sure the arms show up as expected
3. Run [./launch.py](./launch.py) and press "Run Spark" in the GUI for each arm you are using

## Device Interfaces
- [./UR/](./UR/) > A wrapper around the RTDE interface for Universal Robotics arms. Our repository uses the addtional sensing capabilities of the *e* models (ie. UR5e). Our arms are configured for use with RobotIQ 2F-85 grippers. 
  - [./UR/arms.py](./UR/arms.py) > Creates *rtde_control* and *rtde_receive* interfaces, and create gripper controllers. This safely calls and transitioning the robot between *servo*, *move*, *speed*, and *freedrive* modes.
     - [./UR/gripper.py](./UR/gripper.py) > [Gripper interface](https://sdurobotics.gitlab.io/ur_rtde/_static/robotiq_gripper.py) from the [API documentation](https://sdurobotics.gitlab.io/ur_rtde/guides/guides.html?highlight=robotiq#use-with-robotiq-gripper).
     - [./UR/dashboard.py](./UR/dashboard.py) > This interfaces with the UR's dashboard server. Useful for reseting e-stops. 
  - [./UR/fk.py](./UR/fk.py) > Helper file for computing the UR5e's forward kinematics.
- [./Spark/](./Spark/) > The firmware for connecting to SPARK via USB and setting the initial offsets. 
  - [./Spark/SparkNode.py](./Spark/SparkNode.py) > Publishes the angles and enable status for each SPARK device.
  - [./Spark/SparkOffsets.py](./Spark/SparkOffsets.py) > Calibrates the zero position of the SPARK unit. The position is saved as a *.pickle* file.
- [./Haptic/](./Haptic/) > The Force Glove USB interface. 
  - [./Haptic/HapticNode.py](./Haptic/HapticNode.py) > Recieves force/torque data over ROS2 and transmits the six haptic motor values to the controller. 
- [./SM/](./SM/) > The SpaceMouse USB interface. 
  - [./SM/SpaceMouseROS.py](./SM/SpaceMouseROS.py) >  Requires *pyspacemouse*; this node publishes the *xyzrpy* readings from the SpaceMouse.
- [./VR/](./VR/) > The VR interface for collecting Controller positions. This utilizes OpenVR and the [Triad OpenVR wrapper interface](https://github.com/TriadSemi/triad_openvr).
  - [./VR/VR_Node.py](./VR/VR_Node.py) > This node interfaces with the VR system to get the *xyzrpy* for each controller.
    - [./VR/triad_openvr.py](./VR/triad_openvr.py) > The Triad Triad OpenVR Python Wrapper. 
  - [./VR/VR_Offsets.py](./VR/VR_Offsets.py) > This program saves the initial offsets for the VR controllers. 
- [./camera/](./camera/) > ROS2 Interfaces for cameras to publish image frames. 
  - [./camera/realsense.py](./camera/realsense.py) > Connects to Intel RealSense RGBD cameras by serial number. The configuration for each camera allows for controling the framerate, resolution and depth information. 
    - [./camera/realsense_get_info.py](./camera/realsense_get_info.py) > This will print the serail number and valid configurations for each camera attatched to the system. 
  - [./camera/generic.py](./camera/generic.py) > Connects to a generic V4L video stream. This can interface with standard USB webcams. 


## WebRTC Streaming
- [./webrtc/](./webrtc/) > Code for creating a WebRTC RGB video stream. 
  - [./webrtc/sender.py](./webrtc/sender.py) & [/sender.py](./webrtc/sender.py) > The IP of the remote video source should be used in both of these files. The *video_dev* argument corisponds to the */dev/video\** V4L stream. 




