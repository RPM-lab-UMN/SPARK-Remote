import pickle
import json
from pathlib import Path

import numpy as np
import torch
import datasets
import imageio.v2 as imageio

# --- User-configurable parameters ---
# You can adjust these to match your specific data.
CONFIG = {
    "robot_type": "UR5e with Robotiq 2F-85 Gripper",
    "fps": 30,
    "image_height": 240,
    "image_width": 320,
    "state_names": [
        "joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5", "joint_pos_6",
        "eef_x", "eef_y", "eef_z", "eef_r", "eef_p", "eef_y",
        "gripper_state"
    ],
    "action_names": [
        "cmd_joint_1", "cmd_joint_2", "cmd_joint_3", "cmd_joint_4", "cmd_joint_5", "cmd_joint_6",
        "cmd_gripper"
    ]
}

# Define a custom feature for the VideoFrame dictionary
VideoFrame = datasets.Features({"path": datasets.Value("string"), "timestamp": datasets.Value("float32")})

def process_and_convert_to_lerobot_format(
    data_dir: str,
    output_dir: str,
    hf_repo_id: str = None,
    save_locally: bool = True,
    push_to_hub: bool = False,
):
    """
    Loads trajectory data, processes it into the full LeRobot v2.1 format,
    saves it locally and/or pushes to hub.
    """
    if save_locally:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    videos_path = output_path / "videos"
    meta_path = output_path / "meta"
    videos_path.mkdir(exist_ok=True)
    meta_path.mkdir(exist_ok=True)
    
    stats_file = meta_path / "episodes_stats.jsonl"
    if stats_file.exists():
        stats_file.unlink()

    all_steps = []
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
        with open(pkl_file, "rb") as f:
            all_data = pickle.load(f)

        meta, episode_data = all_data['meta'], all_data['frames']
        
        wrist_frames, scene_frames = [], []
        episode_states, episode_actions = [], []
        episode_tasks = set()

        # Loop through timesteps to create transitions and collect data
        for frame_idx, step in enumerate(episode_data):
            task = step['lang_instruction']
            episode_tasks.add(task)
            if task not in task_to_id:
                task_id = len(task_to_id)
                task_to_id[task] = task_id
                tasks_metadata.append({"task_index": task_id, "task": task})
            wrist_frames.append(step['rgb_wrist'])
            scene_frames.append(step['rgb_scene'])

            if frame_idx < len(episode_data) - 1:
                step_t = episode_data[frame_idx]
                # step_t_plus_1 = episode_data[frame_idx + 1]
                eef_pose = np.concatenate(
                    [step_t['eef_pose']['position'], step_t['eef_pose']['orientation_rpy']]
                    ).astype(np.float32)

                actions = np.concatenate([
                    np.array(step_t['spark_command_angles'], dtype=np.float32),
                    np.array([step_t['spark_command_gripper']], dtype=np.float32)
                ])
                state_t = np.concatenate([
                    np.array(step_t['joint_positions'], dtype=np.float32),
                    np.array(eef_pose, dtype=np.float32),
                    np.array([step_t['gripper_state']], dtype=np.float32)
                ])
                # state_t_plus_1 = np.concatenate([
                #     np.array(step_t_plus_1['joint_positions'], dtype=np.float32),
                #     np.array(step_t_plus_1['eef_pose'], dtype=np.float32),
                #     np.array([step_t_plus_1['gripper_state']], dtype=np.float32)
                # ])
                
                episode_states.append(state_t)
                episode_actions.append(actions)
                
                video_filename = f"episode_{episode_idx}.mp4"
                wrist_video_path = f"videos/wrist_{video_filename}"
                scene_video_path = f"videos/scene_{video_filename}"

                all_steps.append({
                    'index': global_idx,
                    'episode_index': episode_idx,
                    'frame_index': frame_idx,
                    'timestamp': step_t['timestamp'],
                    'observation': {
                        'image_wrist': {'path': wrist_video_path, 'timestamp': step_t['timestamp']},
                        'image_scene': {'path': scene_video_path, 'timestamp': step_t['timestamp']},
                        'state': state_t,
                    },
                    'actions': actions,
                    'next': {
                        # 'observation': {
                        #     'image_wrist': {'path': wrist_video_path, 'timestamp': step_t_plus_1['timestamp']},
                        #     'image_scene': {'path': scene_video_path, 'timestamp': step_t_plus_1['timestamp']},
                        #     'state': state_t_plus_1,
                        # },
                        'done': frame_idx == len(episode_data) - 2,
                    },
                    'task': task,
                })
                global_idx += 1
        
        imageio.mimsave(videos_path / f"wrist_episode_{episode_idx}.mp4", wrist_frames, fps=CONFIG["fps"])
        imageio.mimsave(videos_path / f"scene_episode_{episode_idx}.mp4", scene_frames, fps=CONFIG["fps"])

        states_tensor = torch.from_numpy(np.array(episode_states))
        actions_tensor = torch.from_numpy(np.array(episode_actions))
        
        # Save episode statistics
        episode_stats = {
            "episode_index": episode_idx,
            "stats": {
                "observation.state": {
                    "mean": states_tensor.mean(axis=0).tolist(), "std": states_tensor.std(axis=0).tolist(),
                    "min": states_tensor.min(axis=0).values.tolist(), "max": states_tensor.max(axis=0).values.tolist(),
                },
                "actions": {
                    "mean": actions_tensor.mean(axis=0).tolist(), "std": actions_tensor.std(axis=0).tolist(),
                    "min": actions_tensor.min(axis=0).values.tolist(), "max": actions_tensor.max(axis=0).values.tolist(),
                }
            }
        }
        with stats_file.open("a") as f:
            f.write(json.dumps(episode_stats) + '\n')
        
        episode_metadata.append({
            "episode_index": episode_idx,
            "tasks": list(episode_tasks), # List of unique tasks for this episode
            "length": len(episode_data),
        })

    # --- Create the Hugging Face Dataset from the collected steps ---
    hf_dataset = datasets.Dataset.from_list(all_steps)
    print("\nDataset created successfully!")
    print(hf_dataset)

    # --- Assemble and save all metadata files ---
    print("\nAssembling and saving metadata...")
    
    # 1. info.json
    total_episodes = len(pkl_files)
    features_dict = {
        "index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "observation.state": {
            "dtype": "float32", "shape": list(np.array(hf_dataset[0]['observation']['state']).shape),
            "names": CONFIG["state_names"]
        },
        "actions": {
            "dtype": "float32", "shape": list(np.array(hf_dataset[0]['actions']).shape),
            "names": CONFIG["action_names"]
        },
        "next.done": {"dtype": "bool", "shape": [1]},
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
        "total_frames": hf_dataset.num_rows, # Total transitions, not raw frames
        "total_tasks": len(tasks_metadata),
        "total_videos": total_episodes * 2,
        "splits": {"train": f"0:{total_episodes}"}, # Default split uses all data for training
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

    # --- Save main dataset to disk ---
    if save_locally:
        print(f"\nSaving main dataset to disk at '{output_path}'...")
        hf_dataset.save_to_disk(str(output_path))
        print("\n✅ Dataset saved locally in LeRobot v2.1 format.")

    # --- Push to Hugging Face Hub ---
    if push_to_hub:
        if hf_repo_id is None:
            raise ValueError("hf_repo_id must be provided to push to the Hub.")
        print(f"\nPushing dataset to the Hub at '{hf_repo_id}'...")
        hf_dataset.push_to_hub(hf_repo_id)
        print("\n✅ Dataset pushed to the Hub successfully.")


if __name__ == '__main__':
    MY_DATA_DIR = "/data/shared_data/real_world_data/pickblueblock_blackbowl"
    OUTPUT_DIR = "./lerobot_datasets_v2_1/pickblueblock_blackbowl"
    MY_HF_REPO_ID = "iamandrewliao/pickblueblock_blackbowl"

    process_and_convert_to_lerobot_format(
        data_dir = MY_DATA_DIR, 
        output_dir = OUTPUT_DIR,
        save_locally=True,
        push_to_hub=False,
        hf_repo_id=MY_HF_REPO_ID,
    )