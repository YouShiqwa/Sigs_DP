from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import av
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
import cv2
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)
register_codecs()


def real_data_to_replay_buffer(
        dataset_path: str, 
        out_store: Optional[zarr.ABSStore]=None, 
        out_resolutions: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
        lowdim_keys: Optional[Sequence[str]]=None,
        image_keys: Optional[Sequence[str]]=None,
        lowdim_compressor: Optional[numcodecs.abc.Codec]=None,
        image_compressor: Optional[numcodecs.abc.Codec]=None,
        n_decoding_threads: int=multiprocessing.cpu_count(),
        n_encoding_threads: int=multiprocessing.cpu_count(),
        max_inflight_tasks: int=multiprocessing.cpu_count()*5,
        verify_read: bool=True
        ) -> ReplayBuffer:
    """
    It is recommended to use before calling this function
    to avoid CPU oversubscription
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)

    out_resolution:
        if None:
            use video resolution
        if (width, height) e.g. (1280, 720)
        if dict:
            camera_0: (1280, 720)
    image_keys: ['camera_0', 'camera_1']
    """
    if out_store is None:
        out_store = zarr.MemoryStore()
    if n_decoding_threads <= 0:
        n_decoding_threads = multiprocessing.cpu_count()
    if n_encoding_threads <= 0:
        n_encoding_threads = multiprocessing.cpu_count()
    if image_compressor is None:
        image_compressor = Jpeg2k(level=50)

    # verify input
    input = pathlib.Path(os.path.expanduser(dataset_path)) #输入目录
    in_zarr_path = input.joinpath('replay_buffer.zarr')     #在这里有输入的zarr数据
    assert in_zarr_path.is_dir()

    in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r')

    # save lowdim data to single chunk  !!!!!
    chunks_map = dict()
    compressor_map = dict()
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
    in_img_res=dict() #这里是确定输入的分辨率，后面有进行裁剪操作'd435_1_color','d435_1_depth','d435_2_color','d435_2_depth','usb_camera'
    in_img_res['d435_1_color']=[640,480]
    in_img_res['d435_1_depth_color']=[640,480]
    in_img_res['d435_2_color']=[640,480]
    in_img_res['d435_2_depth_color']=[640,480]
    in_img_res['usb_camera']=[3280,2464] #w,h

    
    # worker function
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            # make sure we can successfully decode
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    
    #在压缩low_dim数据的时候，从0开始压缩，那么选择压缩image数据的时候，也从episode0开始压缩

    n_steps = in_replay_buffer.n_steps
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    episode_lengths = in_replay_buffer.episode_lengths
    timestamps = in_replay_buffer['timestamps'][:]
    dt = timestamps[1] - timestamps[0]
    n_cameras=5    #一共要压缩多少图片
    with tqdm(total=n_steps*n_cameras, desc="Loading image data", mininterval=0.1) as pbar:     #n_steps就是一共有多少时间步需要压缩 每个时间步到底有几个相机
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
            futures = set()
        #     step_dirs = sorted([d for d in os.listdir(dataset_path)
        #     if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()
        # ])
            for episode_idx, episode_length in enumerate(episode_lengths):    #0 148 1 148 找到每一个episode    现在就是进入了每一个episode中进行添加图片
                episode_start = episode_starts[episode_idx]  #从0开始寻找压缩
                episode_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith('episode')])[episode_idx]
                step_dirs_src=os.path.join(dataset_path,episode_dirs)
                step_dirs = sorted([d for d in os.listdir(step_dirs_src)
            if os.path.isdir(os.path.join(step_dirs_src, d)) and d.isdigit()
            ])
                for step_idx in tqdm(range(len(step_dirs))): #读取图片  step_idx->frame 
                    arr_names={'d435_1_color','d435_1_depth_color','d435_2_color','d435_2_depth_color','usb_camera'}   #输入的都是三通道数据，在此之前要先进行转换
                    for arr_name in arr_names:
                        out_img_res = tuple(out_resolutions[arr_name])
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
                        img=cv2.imread(img_path)
                        image_tf = get_image_transform(          
                        input_res=in_img_res[arr_name], output_res=out_img_res, bgr_to_rgb=False)  #输入和输出的分辨率
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
    return out_replay_buffer

