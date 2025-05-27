import cv2
import cv2
import os.path
import glob
import numpy as np
from PIL import Image
# if __name__ == "__main__":
#     # 1.读取一张深度图
#     depth_img = cv2.imread("/home/chengy/NewDisk/Data/SIGS/DP/episode_1/00000/usb_camera.png", cv2.IMREAD_UNCHANGED)
#     cv2.imshow("depth", depth_img)
#     # 2.转换深度图 , 将深度图转换为[0-255] 范围更直观的表示形式显示
#     depth_normalized = cv2.convertScaleAbs(depth_img, alpha=255.0 / depth_img.max())
#     # 3.显示深度图
#     cv2.imshow("depth_normalized", depth_normalized)
#     cv2.waitKey(50000)
	
import cv2
import os
from PIL import Image

# def convertPNG(pngfile, outdir):
#     """将深度图转为伪彩色图，保留原始文件"""
#     # 读取原始深度图（16位/单通道）
#     im_depth = cv2.imread(pngfile, cv2.IMREAD_UNCHANGED)  # 保留原始位深
    
#     # 生成输出文件名（避免覆盖原图）
#     basename = os.path.basename(pngfile)
#     filename, ext = os.path.splitext(basename)
#     color_filename = f"{filename}_colorized{ext}"  # 添加_colorized后缀
    
#     # 将深度图转为8位并应用伪彩色
#     im_color = cv2.applyColorMap(
#         cv2.convertScaleAbs(im_depth, alpha=1),  # 调整alpha值控制对比度
#         cv2.COLORMAP_JET
#     )
    
#     # 保存伪彩色图（与原图同一目录）
#     output_path = os.path.join(outdir, color_filename)
#     cv2.imwrite(output_path, im_color)
#     print(f"伪彩色图已保存至: {output_path}")



 
import cv2
import os
import numpy as np
from PIL import Image

PNG_FILE = "/home/chengy/NewDisk/Data/SIGS/DP/episode_1/00001/d435_2_depth.png"
TARGET_DIR = "/home/chengy/NewDisk/Data/SIGS/DP/episode_1/00001"

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
    
    print(f"Original: {pngfile}")
    print(f"Colorized: {output_path}")
    print(f"Depth range: {uint16_img.min()}\~{uint16_img.max()} (raw values)")

# Run conversion (original file remains untouched)
convertPNG(PNG_FILE, TARGET_DIR)
img = cv2.imread("/home/chengy/NewDisk/Data/SIGS/DP/episode_1/00001/d435_2_depth_color.png")
cv2.imshow("img",img)
cv2.waitKey(50000)

