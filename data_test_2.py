import cv2
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import av
import concurrent.futures
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from diffusion_policy.common.cv2_util import my_get_image_transform
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)
register_codecs()
max_inflight_tasks: int=multiprocessing.cpu_count()*5
n_encoding_threads = multiprocessing.cpu_count()
dataset_path='/home/chengy/NewDisk/Data/SIGS/DP'
input = pathlib.Path(os.path.expanduser(dataset_path)) #输入目录
in_zarr_path = input.joinpath('replay_buffer.zarr')
assert in_zarr_path.is_dir()
out_store = zarr.DirectoryStore("/home/chengy/NewDisk/Data/SIGS/DP/output.zarr")
lowdim_compressor=None
lowdim_keys = None
verify_read = True
in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r')
image_compressor = Jpeg2k(level=50)  #压缩器
# save lowdim data to single chunk  !!!!!
chunks_map = dict()
compressor_map = dict()
futures = set()
for key, value in in_replay_buffer.data.items():
    chunks_map[key] = value.shape
    compressor_map[key] = lowdim_compressor

print('Loading lowdim data')
out_replay_buffer = ReplayBuffer.copy_from_store(
    src_store=in_replay_buffer.root.store,
    store=out_store,
    keys=lowdim_keys,
    chunks=chunks_map,
    compressors=compressor_map
    )



episode_path=dataset_path
expected_shape=(240, 320, 3)

def put_img(zarr_arr, zarr_idx, img):
    try:
        zarr_arr[zarr_idx] = img
        # make sure we can successfully decode
        if verify_read:
            _ = zarr_arr[zarr_idx]
        return True
    except Exception as e:
        return False
    
n_steps = in_replay_buffer.n_steps      #一共有多少data 296                   #一定要注意都是从0开始
episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]    #start 0 148 结束的位置 148 296 每一个episode的长度148 148
episode_lengths = in_replay_buffer.episode_lengths #148 148
timestamps = in_replay_buffer['timestamps'][:]
dt = timestamps[1] - timestamps[0]
in_img_res=(640,480)    #输入图片的分辨率
out_img_res=(320,240)   #(W,H)
n_cameras=5
with tqdm(total=n_steps*n_cameras, desc="Loading image data", mininterval=0.1) as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
        futures = set()
    #     step_dirs = sorted([d for d in os.listdir(dataset_path)
    #     if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()
    # ])
        step_dirs_src=os.path.join(dataset_path,"episode_3")
        step_dirs = sorted([d for d in os.listdir(step_dirs_src)
        if os.path.isdir(os.path.join(step_dirs_src, d)) and d.isdigit()
    ])
        for episode_idx, episode_length in enumerate(episode_lengths):    #0 148 1 148 找到每一个episode    现在就是进入了每一个episode中进行添加图片
            episode_start = episode_starts[episode_idx]  #从0开始寻找压缩
            for step_idx in tqdm(range(len(step_dirs))): #读取图片  i->frame 
                arr_names={'d435_1_color','d435_1_depth','d435_2_color','d435_2_depth'}
                for arr_name in arr_names:
                    if arr_name=='d435_1_depth' or arr_name=='d435_2_depth':
                        if arr_name not in out_replay_buffer:
                                ow, oh = out_img_res
                                _ = out_replay_buffer.data.require_dataset(
                                    name=arr_name,
                                    shape=(n_steps,oh,ow,1),
                                    chunks=(1,oh,ow,1),
                                    compressor=image_compressor,
                                    dtype=np.uint8
                                )
                    else:
                        if arr_name not in out_replay_buffer:
                                ow, oh = out_img_res
                                _ = out_replay_buffer.data.require_dataset(
                                    name=arr_name,
                                    shape=(n_steps,oh,ow,3),
                                    chunks=(1,oh,ow,3),
                                    compressor=image_compressor,
                                    dtype=np.uint8
                                )
                    arr = out_replay_buffer[arr_name]#找到对应的buffer    下面应该寻找相机
                    img_name = arr_name+".png"
                    # d435_1_color_path = os.path.join(episode_path, step_dirs[i], "d435_1_color.png")
                    # d435_1_depth_path = os.path.join(episode_path, step_dirs[i], "d435_1_depth.png")
                    # d435_2_color_path = os.path.join(episode_path, step_dirs[i], "d435_2_color.png")
                    # d435_2_depth_path = os.path.join(episode_path, step_dirs[i], "d435_2_depth.png")
                    # usb_camera_path = os.path.join(episode_path, step_dirs[i], "usb_camera.png")
                    img_path=os.path.join(step_dirs_src, step_dirs[step_idx],img_name)
                    if arr_name=='d435_1_depth' or arr_name=='d435_2_depth':
                        img=cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
                        # img = np.transpose(img, (1, 0))  
                        is_depth=True
                    else:
                        img=cv2.imread(img_path)
                        # img = np.transpose(img, (1, 0, 2)) 
                        is_depth=False
                    image_tf = my_get_image_transform(
                    input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False,is_depth=is_depth)  #输入和输出的分辨率
                    img_tf_out=image_tf(img)
                    if len(futures) >= max_inflight_tasks:
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        for f in completed:
                            if not f.result():
                                raise RuntimeError('Failed to encode image!')
                        pbar.update(len(completed))
                                
                    global_idx = episode_start + step_idx
                    futures.add(executor.submit(put_img, arr, global_idx, img_tf_out))

                    if step_idx == (episode_length - 1):
                        break
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))
                # img_depth_1= cv2.imread(d435_1_depth_path, cv2.IMREAD_ANYDEPTH)
                # img_depth_1 = cv2.resize(img_depth_1, expected_shape[:2][::-1])  # OpenCV 需要 (宽,高)

                # img_depth_2= cv2.imread(d435_2_depth_path, cv2.IMREAD_ANYDEPTH)
                # img_depth_2 = cv2.resize(img_depth_2, expected_shape[:2][::-1])  # OpenCV 需要 (宽,高)

                # img_1= cv2.imread(d435_1_color_path)
                # # 转换通道顺序 BGR -> RGB
                # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
                # # 调整尺寸
                # img_1 = cv2.resize(img_1, expected_shape[:2][::-1])  

                # img_2= cv2.imread(d435_2_color_path)
                # # 转换通道顺序 BGR -> RGB
                # img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
                # # 调整尺寸
                # img_2 = cv2.resize(img_2, expected_shape[:2][::-1])  

                # image_tac= cv2.imread(usb_camera_path)
                # # 转换通道顺序 BGR -> RGB
                # image_tac = cv2.cvtColor(image_tac, cv2.COLOR_BGR2RGB)
                # # 调整尺寸
                # image_tac = cv2.resize(image_tac, expected_shape[:2][::-1])  