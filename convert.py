# 1. 取第一帧
# 2. 每个视频都提取所有帧，保存在同名文件夹下
import os
from pathlib import Path
import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./output/output_0_20250813_233500", help='folder of input videos')
    parser.add_argument("--first_frame_root", type=str, default='first_frame', help="Root directory for output.")

    args = parser.parse_args()
    input_dir = args.input_dir
    first_frame_dir = args.first_frame_root
    for video_path in Path(input_dir).glob("*.mp4"):
        print(f"Processing: {video_path}")

        # 创建同名目录
        frames_dir = video_path.with_suffix("")
        os.makedirs(frames_dir, exist_ok=True)
        # 提取第一帧
        input_dir_name = input_dir.split('/')[-1]
        first_frame_path = os.path.join(first_frame_dir, input_dir_name)
        os.makedirs(first_frame_path, exist_ok=True)
        first_frame_path = os.path.join(first_frame_path, frames_dir.name+ ".jpg")
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", "select=eq(n\\,0)", "-q:v", "2",
            str(first_frame_path),
            "-y"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 提取所有帧（%05d 格式）
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            str(frames_dir / "%05d.jpg")
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Done.")
