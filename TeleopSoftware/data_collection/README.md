## Scripts for data saving
- [./data_collection.py ](./data_collection.py) > Main script for recording data
- [./camera.py ](./camera.py) > Starts up RealSenseCamera, starts camera feed
- [./check_pickle.py ](./check_pickle.py) > Inspects trajectory pickle files, saves videos from RGB image observations 

## Collecting data
0. Make sure the "Starting SPARK-Remote" steps have been done in [../README.md](../README.md)
1. Inside [./data_collection.py ](./data_collection.py), change the "data collection settings" variables as needed like 'LANG_INSTRUCTION'. If using/recording factors (e.g. table height, block position, etc.), change necessary values in [./factor_utils.py ](./factor_utils.py) (e.g. if collecting at table height h, change the table_height in gen_factors(); if collecting demos for a specific area on the table, change the position ranges of block_x and block_y; etc.). Also, comment out the gelsight lines in collect_one_frame() if not using gelsight; otherwise, there will be many annoying error messages.
2. Run [./data_collection.py ](./data_collection.py)
3. Press 's' to record, 'e' to stop, etc.
4. (Optional) After done collecting demos, send to the server for training, converting, etc.
```
rsync -avP /path/to/local/demos/ user@server:/path/to/server/directory
```

## Troubleshooting data collection/deployment
- [./test_camera.py ](./test_camera.py) > Script for testing your cameras
- If you run into the error below during data collection/deployment, it might be because 
    - the sudden motion when you start SPARK jostled the camera cable out of its socket. You can confirm this with [./test_camera.py ](./test_camera.py).
    - or a USB bandwidth issue (try putting the Realsense cameras in different ports)
```
File "/home/andrewliao/SPARK-Remote/TeleopSoftware/data_collection/camera.py", line 17, in get_color_frame
    frames = self.pipeline.wait_for_frames()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Frame didn't arrive within 5000

# or something like "Device/resource busy"
```

## Convert to LeRobot: 
Change necessary fields in [./convert_to_lerobot.py ](./convert_to_lerobot.py) (e.g. image_height, image_width) and run.  
**Note:** Make sure datasets==3.6.0!
**IMPORTANT (this applies to all models besides the ones trained for the putgreeninpot data curation experiments):** Previously, in the state_t variable, I wasn't normalizing the gripper_state (i.e. the model trained with unnormalized gripper state but was outputting 0-1 gripper actions). Thus, if you are deploying the models mentioned above, change the following code in pi0_deploy.ipynb (in openpi/examples/) before you deploy:
```
# CHANGE TO THIS:
gripper_state_raw = np.array([current_obs_state["gripper_state"]], dtype=np.float32)
policy_state_vector = np.concatenate([joint_pos, gripper_state_raw])
```

## Deployment
On the UR tablet:
1. Turn on UR5 arm. Ensure it's in remote control mode.  

In the TeleopSoftware/data_collection folder:
1. ```pip install Pyro5```
2. ```python -m Pyro5.nameserver```
3. In another terminal, start a nameserver: ```pyro5-ns```
4. Run ```python deploy.py```. Make sure to change variables as necessary. Also, change the home position in [ur5.py ](./ur5/ur5.py)
  
In pi0_deploy.ipynb in openpi/examples/:
1. Run the steps to setup, load model, initialize cameras, and run actual deployment.
Make sure to change variables as needed including "config", "checkpoint_dir", "prompt", "SAVE_DIR", etc.