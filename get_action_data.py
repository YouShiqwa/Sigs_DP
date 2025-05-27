import os
import json
import shutil
from tqdm import tqdm

def merge_actions_in_episode(dataset_path: str):
    episode_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith('episode')])
    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_path, episode_dir)
        if not os.path.isdir(episode_path):
            continue
        print(f"处理 {episode_dir}...")
        step_dirs = sorted([
            d for d in os.listdir(episode_path)
            if os.path.isdir(os.path.join(episode_path, d)) and d.isdigit()
        ])

        for i in tqdm(range(len(step_dirs) - 1)):  # 不处理最后一个
            curr_step = step_dirs[i]
            next_step = step_dirs[i + 1]

            curr_json_path = os.path.join(episode_path, curr_step, "state.json")
            next_json_path = os.path.join(episode_path, next_step, "state.json")

            if not os.path.isfile(curr_json_path) or not os.path.isfile(next_json_path):

                print(f"缺失文件：{curr_json_path} 或 {next_json_path}，跳过")
                continue

            with open(curr_json_path, 'r') as f:
                curr_data = json.load(f)
            with open(next_json_path, 'r') as f:
                next_data = json.load(f)
            next_robot_data = next_data.get('robot_data', {})
            joint_angles = next_robot_data.get('joint_angles',[])

            gripper_angle = next_robot_data.get('gripper_angle',0)
            action = joint_angles + [gripper_angle]  
            if 'action' not in curr_data:
                curr_data['action'] = []

            curr_data['action'] = action
            with open(curr_json_path, 'w') as f:
                json.dump(curr_data, f, indent=4)
            # print(f"合成 action 到 {curr_step}/state.json(来自 {next_step})")
        last_dir = os.path.join(episode_path, step_dirs[-1])   # 删除最后一个文件夹
        shutil.rmtree(last_dir)

if __name__ == "__main__":
    dataset_path = "/home/chengy/NewDisk/Data/SIGS/DP"
    merge_actions_in_episode(dataset_path)
