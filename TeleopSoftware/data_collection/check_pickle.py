import pickle
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

# ====== 1. Load the .pkl file ======
def load_trajectory(pkl_path):
    with open(pkl_path, 'rb') as f:
        traj = pickle.load(f)
    return traj

# ====== 2. Print summary of the trajectory ======
def print_summary(traj):
    meta = traj.get('meta', {})
    frames = traj.get('frames', [])

    print("\n===== Trajectory Summary =====")
    print(f"Trajectory ID: {meta.get('traj_id', 'N/A')}")
    print(f"Date: {meta.get('date', 'N/A')}")
    print(f"Total frames: {len(frames)}")

    if frames:
        for i in range(len(frames)-1, len(frames)):
            sample_frame = frames[i]
            print("\nExample frame keys:", list(sample_frame.keys()))
            print("RGB image shape:", sample_frame['rgb'].shape)
            print("Joint positions:", sample_frame['joint_positions'])
            print("EE pose:", sample_frame['eef_pose'])
            if 'gripper_state' in sample_frame:
                print("Gripper state:", sample_frame['gripper_state'])
            if 'action' in sample_frame:
                print("Action:", sample_frame['action'])
            if 'lang_instruction' in sample_frame:
                print("Language instruction:", sample_frame['lang_instruction'])

# ====== 3. Play and save the RGB sequence as a video ======
def play_and_save_video(traj, save_path='output_video.mp4', fps=10):
    frames = traj.get('frames', [])
    if not frames:
        print("No frames to visualize.")
        return

    # Determine image dimensions
    h, w, _ = frames[0]['rgb'].shape
    print(f"Video size: {w}x{h}")

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    for idx, frame in enumerate(frames):
        rgb = frame['rgb']  # (H, W, 3), RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for cv2 display
        video_writer.write(bgr)

        # playback
        cv2.imshow('Video', bgr)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {save_path}")

# ====== Main ======
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        pkl_path = input("Please enter the path to the .pkl file: ")

    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        exit(1)

    traj = load_trajectory(pkl_path)
    print_summary(traj)
    play_and_save_video(traj, save_path='trajectory_video.mp4', fps=10)

