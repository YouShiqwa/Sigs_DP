from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import av
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
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
    
    # verify input
    input = pathlib.Path(os.path.expanduser(dataset_path)) #输入目录
    in_zarr_path = input.joinpath('replay_buffer.zarr')
    in_video_dir = input.joinpath('videos')
    assert in_zarr_path.is_dir()
    assert in_video_dir.is_dir()
    
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