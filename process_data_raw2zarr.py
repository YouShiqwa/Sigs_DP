import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from diffusion_policy.common.replay_buffer import ReplayBuffer
import pathlib
import zarr
import imagecodecs
import cv2
import os
import numpy as np
from PIL import Image

def extract_data_from_episode(episode_path: str) -> np.ndarray:   
    step_dirs = sorted([
        d for d in os.listdir(episode_path)
        if os.path.isdir(os.path.join(episode_path, d)) and d.isdigit()
    ])
    robot_datas = {}
    actions=[]
    timestamps=[]
    force_datas = []
    for i in tqdm(range(len(step_dirs))):
        state_path = os.path.join(episode_path, step_dirs[i], "state.json")
        if not os.path.exists(state_path):
            continue
        with open(state_path, 'r') as f:
            data = json.load(f)
        current_robot_data = data['robot_data']
        if isinstance(current_robot_data, dict):
            # 处理字典形式的robot_data
            for key, value in current_robot_data.items():
                if key not in robot_datas:
                    robot_datas[key] = []
                robot_datas[key].append(value)
        else:
            # 处理单个数值形式的robot_data
            if 'robot_data' not in robot_datas:
                robot_datas['robot_data'] = []
            robot_datas['robot_data'].append(current_robot_data)
        if 'action' in data:
            action = data['action']
            actions.append(action)
        else:
            print(f"{state_path} 的 action 格式异常: {action}")
        if 'timestamp' in data:
            timestamp = data['timestamp']
            timestamps.append(timestamp)
        else:
            print(f"{state_path} 的 timestap 格式异常: {timestamp}")
        if 'force_data' in data:
            force_data = data['force_data']['force_torque']
            force_datas.append(force_data)
        else:
            print(f"{state_path} 的 force_data 格式异常: {force_data}")
        episode={}

    actions = np.stack(actions, axis=0)
    timestamps = np.stack(timestamps, axis=0)
    force_datas = np.stack(force_datas, axis=0)
    episode['action']=actions
    episode['timestamps']=timestamps
    episode['force_datas']=force_datas
    for key, value in robot_datas.items():
        episode[key] = np.stack(value,axis=0)
    return episode
if __name__ == "__main__":
    dataset_path = "/home/chengy/NewDisk/Data/SIGS/DP"
    output_dir="/home/chengy/NewDisk/Data/SIGS/DP"
    output_dir = pathlib.Path(output_dir)
    zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='a') #创建可以追加的zarr数据集存储
    episode_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith('episode')])
    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_path, episode_dir)
        episode=extract_data_from_episode(episode_path)
        episode_id = replay_buffer.n_episodes
        replay_buffer.add_episode(episode, compressors='disk')    #Low_dim数据加载完成,可以加载很多个episode 这里就追加完成了



