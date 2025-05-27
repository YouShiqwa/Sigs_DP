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
# root = zarr.open('/home/chengy/NewDisk/Data/SIGS/DP/output.zarr', mode='r')
# print(zarr.tree(root))
# {
#     "robot_data": {
#         "joint_angles": [
#             0.6297346505496129,
#             0.7975020506207261,
#             0.6281331654727823,
#             0.09979689441980633,
#             0.33717034687121383,
#             0.24341240584086565
#         ],
#         "joint_speeds": [
#             0.7848431228028022,
#             0.6006770077961366,
#             0.29609024086874425,
#             0.32532853328974654,
#             0.5115908717371666,
#             0.7485381261639255
#         ],
#         "robot_pose": [
#             0.5727706425080172,
#             0.20862241856806962,
#             0.5953289206489875,
#             0.4478946028745461,
#             0.24274686531324607,
#             0.9008185980450505
#         ],
#         "tcp_speed": [
#             0.24032841457171872,
#             0.6270826865815509,
#             0.17248195191993143,
#             0.14049370349056922,
#             0.04953107415384661,
#             0.08163733862314748
#         ],
#         "gripper_angle": 0.08854392458523586
#     },
#     "force_data": {
#         "force_torque": [
#             0.5672405990479155,
#             0.42282414635829735,
#             0.2580229853325694,
#             0.40604739715391314,
#             0.6684102175792895,
#             0.870043765082138
#         ]
#     },
#     "timestamp": 1747126383.6484542
# }
# def extract_actions_from_episode(episode_path: str) -> np.ndarray:
#     step_dirs = sorted([
#         d for d in os.listdir(episode_path)
#         if os.path.isdir(os.path.join(episode_path, d)) and d.isdigit()
#     ])
#     actions = []

#     for i in tqdm(range(len(step_dirs))):
#         state_path = os.path.join(episode_path, step_dirs[i], "state.json")
#         if not os.path.exists(state_path):
#             continue
#         with open(state_path, 'r') as f:
#             data = json.load(f)
#         if 'action' in data:
#             action = data['action']
#             actions.append(action)
#         else:
#             print(f"{state_path} 的 action 格式异常: {action}")

#     return np.array(actions)

# def extract_data_from_episode(episode_path: str) -> np.ndarray:   
#     step_dirs = sorted([
#         d for d in os.listdir(episode_path)
#         if os.path.isdir(os.path.join(episode_path, d)) and d.isdigit()
#     ])
#     robot_datas = {}
#     actions=[]
#     timestamps=[]
#     force_datas = []
#     for i in tqdm(range(len(step_dirs))):
#         state_path = os.path.join(episode_path, step_dirs[i], "state.json")
#         if not os.path.exists(state_path):
#             continue
#         with open(state_path, 'r') as f:
#             data = json.load(f)
#         current_robot_data = data['robot_data']
#         if isinstance(current_robot_data, dict):
#             # 处理字典形式的robot_data
#             for key, value in current_robot_data.items():
#                 if key not in robot_datas:
#                     robot_datas[key] = []
#                 robot_datas[key].append(value)
#         else:
#             # 处理单个数值形式的robot_data
#             if 'robot_data' not in robot_datas:
#                 robot_datas['robot_data'] = []
#             robot_datas['robot_data'].append(current_robot_data)
#         if 'action' in data:
#             action = data['action']
#             actions.append(action)
#         else:
#             print(f"{state_path} 的 action 格式异常: {action}")
#         if 'timestamp' in data:
#             timestamp = data['timestamp']
#             timestamps.append(timestamp)
#         else:
#             print(f"{state_path} 的 timestap 格式异常: {timestap}")
#         if 'force_data' in data:
#             force_data = data['force_data']['force_torque']
#             force_datas.append(force_data)
#         else:
#             print(f"{state_path} 的 force_data 格式异常: {force_data}")

#     return robot_datas, np.array(actions), np.array(timestamps),np.array(force_datas)
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
    episode['actions']=actions
    episode['timestamps']=timestamps
    episode['force_datas']=force_datas
    for key, value in robot_datas.items():
        episode[key] = np.stack(value,axis=0)
    return episode


# def extract_image_data_from_episode(episode_path: str) -> np.ndarray:     #图片的压缩格式
#     expected_shape=(240, 320, 3)
#     images_1=[]
#     images_2=[]
#     images_1_dep=[]
#     images_2_dep=[]
#     images_tac=[]
#     step_dirs = sorted([
#         d for d in os.listdir(episode_path)
#         if os.path.isdir(os.path.join(episode_path, d)) and d.isdigit()
#     ])
#     for i in tqdm(range(len(step_dirs))):
#         d435_1_color_path = os.path.join(episode_path, step_dirs[i], "d435_1_color.png")
#         d435_1_depth_path = os.path.join(episode_path, step_dirs[i], "d435_1_depth.png")
#         d435_2_color_path = os.path.join(episode_path, step_dirs[i], "d435_2_color.png")
#         d435_2_depth_path = os.path.join(episode_path, step_dirs[i], "d435_2_depth.png")
#         usb_camera_path = os.path.join(episode_path, step_dirs[i], "usb_camera.png")
#         img_depth_1= cv2.imread(d435_1_depth_path, cv2.IMREAD_ANYDEPTH)
#         img_depth_1 = cv2.resize(img_depth_1, expected_shape[:2][::-1])  # OpenCV 需要 (宽,高)
#         images_1_dep.append(img_depth_1)
#         img_depth_2= cv2.imread(d435_2_depth_path, cv2.IMREAD_ANYDEPTH)
#         img_depth_2 = cv2.resize(img_depth_2, expected_shape[:2][::-1])  # OpenCV 需要 (宽,高)
#         images_2_dep.append(img_depth_2)
#         img_1= cv2.imread(d435_1_color_path)
#         # 转换通道顺序 BGR -> RGB
#         img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
#         # 调整尺寸
#         img_1 = cv2.resize(img_1, expected_shape[:2][::-1])  
#         images_1.append(img_1)
#         img_2= cv2.imread(d435_2_color_path)
#         # 转换通道顺序 BGR -> RGB
#         img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
#         # 调整尺寸
#         img_2 = cv2.resize(img_2, expected_shape[:2][::-1])  
#         images_2.append(img_2)
#         image_tac= cv2.imread(usb_camera_path)
#         # 转换通道顺序 BGR -> RGB
#         image_tac = cv2.cvtColor(image_tac, cv2.COLOR_BGR2RGB)
#         # 调整尺寸
#         image_tac = cv2.resize(image_tac, expected_shape[:2][::-1])  
#         images_tac.append(image_tac)
#     return images_1,images_2,images_1_dep,images_2_dep,images_tac






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

episode=extract_data_from_episode("/home/chengy/NewDisk/Data/SIGS/DP/episode_3")
output_dir="/home/chengy/NewDisk/Data/SIGS/DP"
output_dir = pathlib.Path(output_dir)
zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='a')      #创建可以追加的zarr数据集存储
episode_id = replay_buffer.n_episodes
replay_buffer.add_episode(episode, compressors='disk')    #Low_dim数据加载完成,可以加载很多个episode 这里就追加完成了


# def extract_forces_from_episode(episode_path: str) -> np.ndarray:
#     step_dirs = sorted([
#         d for d in os.listdir(episode_path)
#         if os.path.isdir(os.path.join(episode_path, d)) and d.isdigit()
#     ])
#     force_datas = []

#     for i in tqdm(range(len(step_dirs))):
#         state_path = os.path.join(episode_path, step_dirs[i], "state.json")
#         if not os.path.exists(state_path):
#             continue
#         with open(state_path, 'r') as f:
#             data = json.load(f)
#         if 'force_data' in data:
#             force_data = data['force_data']['force_torque']
#             force_datas.append(force_data)
#         else:
#             print(f"{state_path} 的 force_data 格式异常: {force_data}")

#     return np.array(force_datas)
# def convertPNG(pngfile, outdir):
#     """Convert 16-bit depth PNG to colorized 8-bit PNG without overwriting original"""
#     # 1. Read 16-bit depth image (0-65535)
#     uint16_img = cv2.imread(pngfile, cv2.IMREAD_UNCHANGED)  # -1 for unchanged bit depth
    
#     # 2. Normalize to [0,1] then scale to [0,255]
#     normalized = cv2.normalize(uint16_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
#     # 3. Invert values (near=red, far=blue)
#     # 4. Apply JET colormap (blue=far, red=near)
#     im_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
#     # 5. Generate new filename (add '_color' suffix)
#     basename = os.path.basename(pngfile)
#     filename, ext = os.path.splitext(basename)
#     color_filename = f"{filename}_color{ext}"
    
#     # 6. Save colorized version
#     output_path = os.path.join(outdir, color_filename)
#     cv2.imwrite(output_path, im_color)  # Directly save with OpenCV
    
    # print(f"Original: {pngfile}")
    # print(f"Colorized: {output_path}")
    # print(f"Depth range: {uint16_img.min()}\~{uint16_img.max()} (raw values)")

# def transform_depth_image_episode(episode_path: str) -> np.ndarray:     #图片的压缩格式 这里可以不改变大小
#     step_dirs = sorted([
#         d for d in os.listdir(episode_path)
#         if os.path.isdir(os.path.join(episode_path, d)) and d.isdigit()
#     ])
#     for i in tqdm(range(len(step_dirs))):
#         d435_1_depth_path = os.path.join(episode_path, step_dirs[i], "d435_1_depth.png")
#         d435_2_depth_path = os.path.join(episode_path, step_dirs[i], "d435_2_depth.png")
#         output_dir=os.path.join(episode_path, step_dirs[i])
#         convertPNG(d435_1_depth_path, output_dir)
#         convertPNG(d435_2_depth_path, output_dir)

# transform_depth_image_episode("/home/chengy/NewDisk/Data/SIGS/DP/episode_3")      #转换为3通道图片




    #     img_depth_1 = cv2.resize(img_depth_1, expected_shape[:2][::-1])  # OpenCV 需要 (宽,高)
    #     images_1_dep.append(img_depth_1)
    #     img_depth_2= cv2.imread(d435_2_depth_path, cv2.IMREAD_ANYDEPTH)
    #     img_depth_2 = cv2.resize(img_depth_2, expected_shape[:2][::-1])  # OpenCV 需要 (宽,高)
    #     images_2_dep.append(img_depth_2)
    #     img_1= cv2.imread(d435_1_color_path)
    #     # 转换通道顺序 BGR -> RGB
    #     img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    #     # 调整尺寸
    #     img_1 = cv2.resize(img_1, expected_shape[:2][::-1])  
    #     images_1.append(img_1)
    #     img_2= cv2.imread(d435_2_color_path)
    #     # 转换通道顺序 BGR -> RGB
    #     img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    #     # 调整尺寸
    #     img_2 = cv2.resize(img_2, expected_shape[:2][::-1])  
    #     images_2.append(img_2)
    #     image_tac= cv2.imread(usb_camera_path)
    #     # 转换通道顺序 BGR -> RGB
    #     image_tac = cv2.cvtColor(image_tac, cv2.COLOR_BGR2RGB)
    #     # 调整尺寸
    #     image_tac = cv2.resize(image_tac, expected_shape[:2][::-1])  
    #     images_tac.append(image_tac)
    # return images_1,images_2,images_1_dep,images_2_dep,images_tac

# import cv2
# import os
# import numpy as np
# from PIL import Image

# PNG_FILE = "/home/chengy/NewDisk/Data/SIGS/DP/episode_1/00001/d435_2_depth.png"
# TARGET_DIR = "/home/chengy/NewDisk/Data/SIGS/DP/episode_1/00001"

# def convertPNG(pngfile, outdir):
#     """Convert 16-bit depth PNG to colorized 8-bit PNG without overwriting original"""
#     # 1. Read 16-bit depth image (0-65535)
#     uint16_img = cv2.imread(pngfile, cv2.IMREAD_UNCHANGED)  # -1 for unchanged bit depth
    
#     # 2. Normalize to [0,1] then scale to [0,255]
#     normalized = cv2.normalize(uint16_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
#     # 3. Invert values (near=red, far=blue)
#     # 4. Apply JET colormap (blue=far, red=near)
#     im_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
#     # 5. Generate new filename (add '_color' suffix)
#     basename = os.path.basename(pngfile)
#     filename, ext = os.path.splitext(basename)
#     color_filename = f"{filename}_color{ext}"
    
#     # 6. Save colorized version
#     output_path = os.path.join(outdir, color_filename)
#     cv2.imwrite(output_path, im_color)  # Directly save with OpenCV
    
#     print(f"Original: {pngfile}")
#     print(f"Colorized: {output_path}")
#     print(f"Depth range: {uint16_img.min()}\~{uint16_img.max()} (raw values)")

# # Run conversion (original file remains untouched)
# convertPNG(PNG_FILE, TARGET_DIR)
# img=cv2.imread("/home/chengy/NewDisk/Data/SIGS/DP/episode_1/00001/d435_2_depth_color.png")
# cv2.imshow("img",img)
# cv2.waitKey(50000)






















# episode=extract_data_from_episode("/home/chengy/NewDisk/Data/SIGS/DP/episode_1")
# output_dir="/home/chengy/NewDisk/Data/SIGS/DP"
# output_dir = pathlib.Path(output_dir)
# zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
# replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='a')      #创建可以追加的zarr数据集存储
# episode_id = replay_buffer.n_episodes
# replay_buffer.add_episode(episode, compressors='disk')    #Low_dim数据加载完成,可以加载很多个episode








