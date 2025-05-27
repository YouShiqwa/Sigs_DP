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
def convertPNG(pngfile, outdir):
    """Convert 16-bit depth PNG to colorized 8-bit PNG without overwriting original"""
    # 1. Read 16-bit depth image (0-65535)
    uint16_img = cv2.imread(pngfile, cv2.IMREAD_UNCHANGED)  # -1 for unchanged bit depth
    
    # 2. Normalize to [0,1] then scale to [0,255]
    normalized = cv2.normalize(uint16_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 3. Invert values (near=red, far=blue)
    # 4. Apply JET colormap (blue=far, red=near)
    im_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # 5. Generate new filename (add '_color' suffix)
    basename = os.path.basename(pngfile)
    filename, ext = os.path.splitext(basename)
    color_filename = f"{filename}_color{ext}"
    
    # 6. Save colorized version
    output_path = os.path.join(outdir, color_filename)
    cv2.imwrite(output_path, im_color)  # Directly save with OpenCV
    
    # print(f"Original: {pngfile}")
    # print(f"Colorized: {output_path}")
    # print(f"Depth range: {uint16_img.min()}\~{uint16_img.max()} (raw values)")

def transform_depth_image_episode(episode_path: str) -> np.ndarray:     #图片的压缩格式 这里可以不改变大小
    step_dirs = sorted([
        d for d in os.listdir(episode_path)
        if os.path.isdir(os.path.join(episode_path, d)) and d.isdigit()
    ])
    for i in tqdm(range(len(step_dirs))):
        d435_1_depth_path = os.path.join(episode_path, step_dirs[i], "d435_1_depth.png")
        d435_2_depth_path = os.path.join(episode_path, step_dirs[i], "d435_2_depth.png")
        output_dir=os.path.join(episode_path, step_dirs[i])
        convertPNG(d435_1_depth_path, output_dir)
        convertPNG(d435_2_depth_path, output_dir)
if __name__ == "__main__":
    dataset_path = "/home/chengy/NewDisk/Data/SIGS/DP"
    episode_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith('episode')])
    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_path, episode_dir)
        transform_depth_image_episode(episode_path)      #转换为3通道图片