## Scripts for data saving
- [./data_collection.py ](./data_collection.py) > Main script for recording data
- [./camera.py ](./camera.py) > Starts up RealSenseCamera, starts camera feed
- [./check_pickle.py ](./check_pickle.py) > Inspects trajectory pickle files, saves videos from RGB image observations 

## Collecting data
1. Inside data_collection.py, change the "data collection settings" variables as needed
2. Run data_collection.py
3. Press 'Run Spark' and get the arm into control mode
4. Press 's' to record, 'e' to stop, etc.

## Convert to LeRobot: 
Change necessary fields in [./check_pickle.py ](./check_pickle.py) and run.

## Troubleshooting
- [./test_camera.py ](./test_camera.py) > Script for testing your cameras
- If you run into the error below during data collection, it might be because the sudden motion when you start SPARK jostled the camera cable out of its socket. You can confirm this with test_camera.py.
```
File "/home/andrewliao/SPARK-Remote/TeleopSoftware/data_collection/camera.py", line 17, in get_color_frame
    frames = self.pipeline.wait_for_frames()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Frame didn't arrive within 5000
```