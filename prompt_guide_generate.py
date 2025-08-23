import subprocess
import time
import os
from datetime import datetime
import csv
import argparse


# 从CSV文件中读取prompts
def load_prompts_from_csv(filename="t2v_shadow_prompts.csv"):
    prompts = []
    nums = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                nums.append(row['num'])
                prompts.append(row['with'])  # 提取"with"列的内容
        print(f"✅ 成功从 {filename} 加载 {len(prompts)} 条提示词")
        return nums,prompts
    except FileNotFoundError:
        print(f"⚠️ 文件 {filename} 不存在，请先运行生成脚本")
        return [],[]
    except KeyError:
        print("⚠️ CSV文件格式错误，请确保包含'with'列")
        return [],[]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="ti2v-5B", help='The task to perform'
    )
    parser.add_argument("--size", type=str, default="1280*704", help="The size of the output video")
    parser.add_argument("--ckpt_dir", type=str, default="/storage/pmj/data_paul/Wan2.2/Wan2.2-TI2V-5B", help="The directory of the model checkpoint")
    parser.add_argument("--file_name", type=str, default='/storage/pmj/data_paul/Wan2.1/t2v_pair_prompts_55.csv', help="The file of the prompts")
    parser.add_argument("--output_root", type=str, default='/storage/pmj/data_paul/Wan2.2/output', help="The root directory for output videos")

    args = parser.parse_args()
    # 模型参数
    task = args.task
    size = args.size
    ckpt_dir = args.ckpt_dir

    # 多条 prompt 列表

    file_name = args.file_name
    nums,prompts=load_prompts_from_csv(file_name)
    # formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_root, exist_ok=True)
    output_file = file_name.split("_")[-1].split('.')[0]
    # 遍历 prompt 并执行命令
    # for i, prompt in enumerate(prompts):
        # print(f"正在生成第 {i+1} 个视频...")
    cmd = [
        "python", "generate_edited.py",
        "--task", task,
        "--size", size,
        "--ckpt_dir", ckpt_dir,
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        # "--prompt", prompt,
        "--base_seed", str(int(time.time())),  # 使用不同的种子以生成不同的视频
        # "--save_file", f"{args.output_root}/output_{output_file}_{formatted_time}/{nums[i]}.mp4",
        "--save_file", f"{args.output_root}",
        "--csv_file_name", file_name,

    ]
        # subprocess.run(cmd)

    try:
        result = subprocess.run(cmd, check=True, 
                                # capture_output=True, 
                                text=True)
        print(f"✅ 视频生成成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ 视频生成失败: {e}")
        print(f"错误输出: {e.stderr}")
    except Exception as e:
        print(f"❌ 意外错误: {e}")
