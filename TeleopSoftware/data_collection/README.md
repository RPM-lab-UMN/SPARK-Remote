## Scripts for data saving
- [./data_collection.py ](./data_collection.py) > Main script for recording data
- [./camera.py ](./camera.py) > Starts up RealSenseCamera, starts camera feed
- [./check_pickle.py ](./check_pickle.py) > Inspects trajectory pickle files, saves videos from RGB image observations 

## Collecting data
0. Make sure the "Starting SPARK-Remote" steps have been done in [../README.md](../README.md)
1. Inside data_collection.py, change the "data collection settings" variables as needed
2. Run data_collection.py
3. Press 's' to record, 'e' to stop, etc.

## Troubleshooting data collection
- [./test_camera.py ](./test_camera.py) > Script for testing your cameras
- If you run into the error below during data collection, it might be because the sudden motion when you start SPARK jostled the camera cable out of its socket. You can confirm this with [./test_camera.py ](./test_camera.py).
```
File "/home/andrewliao/SPARK-Remote/TeleopSoftware/data_collection/camera.py", line 17, in get_color_frame
    frames = self.pipeline.wait_for_frames()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Frame didn't arrive within 5000
```

## Convert to LeRobot: 
Change necessary fields in [./convert_to_lerobot.py ](./convert_to_lerobot.py) and run.  
**Note:** Make sure datasets==3.6.0!

## Deployment
In the TeleopSoftware/data_collection folder:
1. ```pip install Pyro5```
2. ```python -m Pyro5.nameserver```
3. In another terminal, start a nameserver: ```pyro5-ns```
4. Run ```python deploy.py```
  
In pi0_deploy.ipynb in openpi/examples/:
1. Run the steps to setup, load model, initialize cameras, and run actual deployment.