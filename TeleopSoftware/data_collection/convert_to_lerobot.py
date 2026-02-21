import pickle
import json
from pathlib import Path

import numpy as np
import torch
import datasets
import imageio.v2 as imageio
import cv2

# --- User-configurable parameters ---
# Adjust these to match your specific data.
CONFIG = {
    "robot_type": "UR5e with Robotiq 2F-85 Gripper",
    "fps": 30,
    "chunk_size": 100, # Number of episodes per chunk; may need to adjust for large datasets
    "image_height": 304, # DOUBLE CHECK THIS
    "image_width": 224, # DOUBLE CHECK THIS
    "state_names": [
        "joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5", "joint_pos_6",
        # "eef_x", "eef_y", "eef_z", "eef_roll", "eef_pitch", "eef_yaw",
        "gripper_state"
    ],
    "action_names": [
        "cmd_joint_1", "cmd_joint_2", "cmd_joint_3", "cmd_joint_4", "cmd_joint_5", "cmd_joint_6",
        "cmd_gripper"
    ]
}

# Define a custom feature for the VideoFrame dictionary
# VideoFrame = datasets.Features({"path": datasets.Value("string"), "timestamp": datasets.Value("float32")})

def process_and_convert_to_lerobot_format(
    data_dir: str,
    output_dir: str,
    hf_repo_id: str = None,
    push_to_hub: bool = False,
):
    """
    Loads trajectory data, processes it into the full LeRobot v2.1 format,
    saves it locally and/or pushes to hub.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    videos_path = output_path / "videos"
    meta_path = output_path / "meta"
    videos_path.mkdir(exist_ok=True)
    meta_path.mkdir(exist_ok=True)
    
    stats_file = meta_path / "episodes_stats.jsonl"
    if stats_file.exists():
        stats_file.unlink()

    episode_metadata = []
    tasks_metadata = []
    task_to_id = {}
    global_idx = 0
    
    pkl_files = sorted(Path(data_dir).glob("*.pkl"))
    print(f"Found {len(pkl_files)} .pkl files to process.")

    # --- Main data processing loop ---
    # Loop through each episode (.pkl file)
    for episode_idx, pkl_file in enumerate(pkl_files):
        print(f"Processing episode {episode_idx}: {pkl_file.name}...")
        pkl_file = pkl_file.resolve()  # in case it's a symbolic link
        with open(pkl_file, "rb") as f:
            all_data = pickle.load(f)

        meta, episode_data = all_data['meta'], all_data['frames']
        
        # --- Setting up directories and paths for videos ---
        # Calculate the chunk index for the current episode
        episode_chunk = episode_idx // CONFIG["chunk_size"]

        # Define the new camera keys and create their directories
        wrist_key = "observation.image_wrist"
        scene_key = "observation.image_scene"
        wrist_video_dir = videos_path / f"chunk-{episode_chunk:03d}" / wrist_key
        scene_video_dir = videos_path / f"chunk-{episode_chunk:03d}" / scene_key
        wrist_video_dir.mkdir(parents=True, exist_ok=True)
        scene_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the full path for the final video files
        wrist_mp4_path = wrist_video_dir / f"episode_{episode_idx:06d}.mp4"
        scene_mp4_path = scene_video_dir / f"episode_{episode_idx:06d}.mp4"

        wrist_frames, scene_frames = [], []
        episode_states, episode_actions = [], []
        episode_tasks = set()

        episode_steps = []

        # --- Loop through timesteps to create transitions and collect data ---
        for frame_idx, step in enumerate(episode_data):
            # TO DO: Remove once timestamp is fixed in data_collection.py script
            step['timestamp'] = frame_idx / CONFIG["fps"] # generates perfect, evenly spaced timestamps
            task = step['lang_instruction']
            episode_tasks.add(task)
            if task not in task_to_id:
                task_id = len(task_to_id)
                task_to_id[task] = task_id
                tasks_metadata.append({"task_index": task_id, "task": task})
            # Make sure images are uint8
            wrist_frame = step['rgb_wrist'].astype(np.uint8)
            scene_frame = step['rgb_scene'].astype(np.uint8)
            # Resize to CONFIG dimensions
            target_shape = (CONFIG["image_height"], CONFIG["image_width"])
            if (wrist_frame.shape[0], wrist_frame.shape[1]) != target_shape:
                wrist_frame = cv2.resize(wrist_frame, (CONFIG["image_width"], CONFIG["image_height"]))
            if (scene_frame.shape[0], scene_frame.shape[1]) != target_shape:
                scene_frame = cv2.resize(scene_frame, (CONFIG["image_width"], CONFIG["image_height"]))

            wrist_frames.append(wrist_frame)
            scene_frames.append(scene_frame)

            if frame_idx < len(episode_data) - 1:
                step_t = episode_data[frame_idx]
                # eef_pose = np.concatenate(
                #     [step_t['eef_pose']['position'], step_t['eef_pose']['orientation_rpy']]
                #     ).astype(np.float32)

                # TO DO: in the future, change to 'actions' as it is more friendly with openpi
                action = np.concatenate([
                    np.array(step_t['spark_command_angles'], dtype=np.float32),
                    np.array([step_t['spark_command_gripper']], dtype=np.float32)
                ])
                state_t = np.concatenate([
                    np.array(step_t['joint_positions'], dtype=np.float32),
                    # np.array(eef_pose, dtype=np.float32),
                    np.array([step_t['gripper_state'] / 255.0], dtype=np.float32)
                ])
                
                episode_states.append(state_t)
                episode_actions.append(action)
                
                episode_steps.append({
                    'index': global_idx,
                    'episode_index': episode_idx,
                    'frame_index': frame_idx,
                    'timestamp': step_t['timestamp'],
                    # LeRobot gets these images from video files
                    # 'observation.image_wrist': wrist_frame,
                    # 'observation.image_scene': scene_frame,
                    'observation.state': state_t,
                    'action': action,
                    # 'next.done': frame_idx == len(episode_data) - 2,
                    'task': task,
                    'task_index': task_id,
                })
                global_idx += 1

        # --- Save this episode's data to a Parquet file ---
        if episode_steps: # Ensure the episode has steps
            # Create the specific chunk directory for the data
            data_chunk_dir = output_path/"data"/f"chunk-{episode_chunk:03d}"
            data_chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Define the output path for this episode's parquet file
            parquet_path = data_chunk_dir/f"episode_{episode_idx:06d}.parquet"
            
            # Create a Dataset for this episode only and save it
            episode_dataset = datasets.Dataset.from_list(episode_steps)
            episode_dataset.to_parquet(parquet_path)
        
        imageio.mimsave(wrist_mp4_path, wrist_frames, fps=CONFIG["fps"])
        imageio.mimsave(scene_mp4_path, scene_frames, fps=CONFIG["fps"])

        states_tensor = torch.from_numpy(np.array(episode_states))
        actions_tensor = torch.from_numpy(np.array(episode_actions))
        
        # --- Save episode statistics ---
        episode_stats = {
            "episode_index": episode_idx,
            "stats": {
                "observation.state": {
                    "mean": states_tensor.mean(axis=0).tolist(), "std": states_tensor.std(axis=0).tolist(),
                    "min": states_tensor.min(axis=0).values.tolist(), "max": states_tensor.max(axis=0).values.tolist(),
                    "count": [states_tensor.shape[0]],
                },
                "action": {
                    "mean": actions_tensor.mean(axis=0).tolist(), "std": actions_tensor.std(axis=0).tolist(),
                    "min": actions_tensor.min(axis=0).values.tolist(), "max": actions_tensor.max(axis=0).values.tolist(),
                    "count": [actions_tensor.shape[0]],
                }
            }
        }
        with stats_file.open("a") as f:
            f.write(json.dumps(episode_stats) + '\n')
        
        episode_metadata.append({
            "episode_index": episode_idx,
            "tasks": list(episode_tasks), # List of unique tasks for this episode
            "length": len(episode_steps),
        })

    # --- Assemble and save all metadata files ---
    print("\nAssembling and saving metadata...")
    
    # 1. info.json
    total_episodes = len(pkl_files)
    state_shape = len(CONFIG["state_names"])
    action_shape = len(CONFIG["action_names"])
    features_dict = {
        "index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "observation.state": {
            "dtype": "float32", "shape": [state_shape],
            "names": CONFIG["state_names"]
        },
        "action": {
            "dtype": "float32", "shape": [action_shape],
            "names": CONFIG["action_names"]
        },
        # "next.done": {"dtype": "bool", "shape": [1]},
        "observation.image_wrist": {
            "dtype": "video", "shape": [CONFIG["image_height"], CONFIG["image_width"], 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": CONFIG["fps"]}
        },
        "observation.image_scene": {
            "dtype": "video", "shape": [CONFIG["image_height"], CONFIG["image_width"], 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": CONFIG["fps"]}
        },
    }

    info = {
        "codebase_version": "v2.1",
        "robot_type": CONFIG["robot_type"],
        "fps": CONFIG["fps"],
        "total_episodes": total_episodes,
        "total_frames": global_idx, # Total transitions (# frames - 1 per episode)
        "total_tasks": len(tasks_metadata),
        "total_videos": total_episodes * 2,
        "total_chunks": (total_episodes + CONFIG["chunk_size"] - 1) // CONFIG["chunk_size"],
        "chunks_size": CONFIG["chunk_size"],
        "splits": {"train": f"0:{total_episodes}"}, # Default split uses all data for training
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features_dict,
    }
    with (meta_path / "info.json").open("w") as f:
        json.dump(info, f, indent=4)
        
    # 2. episodes.jsonl
    with (meta_path / "episodes.jsonl").open("w") as f:
        for item in episode_metadata:
            f.write(json.dumps(item) + "\n")

    # 3. tasks.jsonl
    with (meta_path / "tasks.jsonl").open("w") as f:
        for item in tasks_metadata:
            f.write(json.dumps(item) + "\n")
            
    # 4. episodes_stats.jsonl was already created in the loop.


    # --- Push to Hugging Face Hub ---
    if push_to_hub:
        from huggingface_hub import HfApi
        if hf_repo_id is None:
            raise ValueError("hf_repo_id must be provided to push to the Hub.")
        print(f"\nPushing dataset to the Hub at '{hf_repo_id}'...")
        
        api = HfApi()
        api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=hf_repo_id,
            repo_type="dataset",
            delete_patterns="*"
        )
        api.create_tag(hf_repo_id, tag=info["codebase_version"], repo_type="dataset", exist_ok=True)
        print("\n✅ Dataset pushed to the Hub successfully.")


if __name__ == '__main__':
    MY_DATA_DIR = "/data/shared_data/real_world_data/pickblueblock/all_bottomleft_topright"
    OUTPUT_DIR = "/home/liao0241/.cache/huggingface/lerobot/iamandrewliao/pickblueblock_bottomleft_topright"
    MY_HF_REPO_ID = "iamandrewliao/pickblueblock_bottomleft_topright"

    process_and_convert_to_lerobot_format(
        data_dir = MY_DATA_DIR, 
        output_dir = OUTPUT_DIR,
        push_to_hub=True,
        hf_repo_id=MY_HF_REPO_ID,
    )