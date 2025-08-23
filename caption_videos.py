# codes from https://gist.github.com/a-r-r-o-w/4dee20250e82f4e44690a02351324a4a

import argparse
import os
import glob
import json
import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import transformers
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_data_root", type=str, required=True, help="Base folder where video files are located."
    )
    parser.add_argument("--cache_dir", type=str, default="~/.cache", help="Path to where models are stored.")
    parser.add_argument(
        "--output_path", type=str, default="captions.json", help="File path where dataset csv should be stored."
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=226, help="Maximum number of new tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args()


SYSTEM_PROMPT = """
You are part of a team that prepares prompts for a video generation model. 
The goal is to generate videos where the main subject has been removed, leaving only the environment and background. 

When analyzing the video:
- First, carefully describe both the main subject(s) and the background.
- Then, discard the subject(s) and only keep the description of the environment, background, lighting, textures, and atmosphere.
- Do not mention or imply the subject in any way. Pretend they never existed.
- Focus on what remains in the scene once the subject is removed.

You will only ever output a single video description per request.

""".strip()

USER_PROMPT = """
Could you generate a prompt for a video generation model for the following video summary:
```
{0}
```
Please limit the prompt to [{1}] words.
""".strip()

QUESTION = """
Describe the video. You should pay close attention to every detail in the video and describe it in as much detail as possible.
""".strip()

MAX_NUM_FRAMES = 49


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_video(video_path: str):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if os.path.isdir(os.path.join(video_path, 'bg')):
        frames = [Image.open(f).resize((256, 128), resample=Image.LANCZOS)
                  for f in sorted(glob.glob(os.path.join(video_path, 'bg', '*.png')))][:MAX_NUM_FRAMES]
    else:
        video_path = os.path.join(video_path, 'rgb_removed.mp4')
        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = 2
        frame_idx = [i for i in range(0, len(vr), sample_fps)][:MAX_NUM_FRAMES]
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")).resize((256, 128), resample=Image.LANCZOS) for v in frames]

    print("num frames:", len(frames))
    return frames


@torch.no_grad()
def main(args: Dict[str, Any]):
    set_seed(args.seed)

    video_files = [file for file in os.listdir(args.instance_data_root)]
    video_files = sorted([os.path.join(args.instance_data_root, file) for file in video_files])
    if args.end_index is not None:
        print(f'[INFO] processing {args.start_index} to {args.end_index}')
        video_files = video_files[args.start_index:args.end_index]
    video_descriptions = {}

    model_id = "openbmb/MiniCPM-V-2_6"
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    for filepath in video_files:
        print(f"Generating video summary for file: `{filepath}`")
        frames = encode_video(filepath)
        msgs = [{"role": "user", "content": frames + [QUESTION]}]

        params = {
            "use_image_id": False,
            "max_slice_nums": 10,
        }

        description = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)
        video_descriptions[filepath] = {"summary": description}

    del model
    del tokenizer
    model = None
    tokenizer = None
    torch.cuda.empty_cache()

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        model_kwargs={
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
        },
    )

    for filepath in video_files:
        print(f"Generating captions for file: `{filepath}`")

        for prompt_type, num_words in [("short_prompt", 25), ("prompt", 75), ("verbose_prompt", 125)]:
            user_prompt = USER_PROMPT.format(video_descriptions[filepath]["summary"], num_words)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=226,
            )
            video_descriptions[filepath][prompt_type] = outputs[0]["generated_text"][-1]["content"]

    df_data = []
    for filepath, description in video_descriptions.items():
        relative_path = os.path.relpath(filepath, args.instance_data_root)
        df_data.append(
            {
                "path": relative_path,
                "short_prompt": description.get("short_prompt", ""),
                "prompt": description.get("prompt", ""),
                "verbose_prompt": description.get("verbose_prompt", ""),
                "summary": description.get("summary", ""),
            }
        )

    json.dump(df_data, open(args.output_path, "w"), indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)

    # Example usage:
    # python3 workflows/caption_videos.py \
    #   --instance_data_root <PATH_TO_DATA> \
    #   --cache_dir <PATH_TO_MODEL_CACHE_DIR> \
    #   --output_path <PATH_TO_DATASET_SAVEDIR>
