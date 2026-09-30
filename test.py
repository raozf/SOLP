import os
import cv2
import json
import glob
import torch
import numpy as np
from PIL import Image
from datetime import datetime

from diffsynth import ModelManager, save_video, SVIVideoPipeline
import torch
from diffsynth import ModelManager, save_video, SVIVideoPipeline
import os
import pickle
import numpy as np
import random
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import cv2
import sys  
import subprocess
from tqdm import tqdm
import torch, os, argparse
from datetime import datetime
from utils.image_process import *
import glob
from utils.project_utils import print_args
from safetensors.torch import save_file, load_file
height = 480
width = 832
seed = None
# seed = 42
max_frames = 81
use_teacache = True

def load_prompts_from_file(prompt_file_path):
    """Load prompts from a prompt.txt file"""
    if not os.path.exists(prompt_file_path):
        print(f"Warning: prompt file not found at {prompt_file_path}")
        return ["Default prompt: the subject is moving naturally"]
    
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Extract prompts from the file content
        # The format is: prompts = ["prompt1", "prompt2", ...]
        if 'prompts = [' in content:
            # Find the prompts list and extract it
            start_idx = content.find('prompts = [')
            if start_idx != -1:
                # Execute the assignment to extract prompts
                local_vars = {}
                exec(content[start_idx:], {}, local_vars)
                return local_vars.get('prompts', ["Default prompt: the subject is moving naturally"])
        
        # If not in expected format, treat each line as a prompt
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
        if lines:
            return lines
        else:
            return ["Default prompt: the subject is moving naturally"]
            
    except Exception as e:
        print(f"Error reading prompts from {prompt_file_path}: {e}")
        return ["Default prompt: the subject is moving naturally"]


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--dit_root",
        default='weights/Wan2.1-I2V-14B-480P/',
        type=str,
        help="Root directory of the Wan2.1-I2 model.",
    )
    parser.add_argument(
        "--extra_module_root",
        default=None,
        type=str,
        help="Optional LoRA checkpoint or directory containing safetensors files.",
    )
    parser.add_argument(
        "--filter_path",
        default=None,
        type=str,
        help="Optional trained spectral-stabilizer checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="videos/",
        type=str,
    )
    parser.add_argument(
        "--cfg_scale_text",
        default=5.0,
        type=float,
    )
    parser.add_argument(
        "--lora_alpha",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--train_architecture",
        default='lora',
        type=str,
    )
    parser.add_argument(
        "--ref_pad_cfg",
        default=False,
        action="store_true",
        help="Whether to set mask with only 1-frame 1.",
    )
    parser.add_argument(
        "--num_motion_frames",
        type=int,
        default=5,
        help="Number of motion frames to generate."
    )
    parser.add_argument(
        "--num_clips",
        type=int,
        default=50,
        help="Number of clips to generate."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of steps to generate."
    )
    parser.add_argument(
        "--data_root",
        type=str,    
        default="data/test_samples",
        help="Root directory containing test samples."
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default=None,
        help="Direct path to reference image. If specified, will override data_root scanning."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Direct path to prompt file. If specified, will be used instead of prompt.txt in data_root."
    )
    parser.add_argument(
        "--test_samples",
        type=str,
        nargs='*',
        help="Specific test sample directories to process. If not specified, all samples will be processed."
    )
    parser.add_argument(
        "--max_prompts_per_sample",
        type=int,
        default=None,
        help="Maximum number of prompts to use per sample. If not specified, all prompts will be used."
    )
    parser.add_argument(
        "--ref_pad_num",
        type=int,
        default=0,  # 0 -> no padding k-> padding k , -1 -> full padding
        help="Number of reference frames to pad with",
    )
    parser.add_argument(
        "--use_first_prompt_only",
        default=False,
        action="store_true",
        help="Whether to use only the first prompt for all clips instead of cycling through different prompts.",
    )
    parser.add_argument(
        "--use_first_aug",
        default=False,
        action="store_true",
        help="Whether to use only the first prompt for all clips instead of cycling through different prompts.",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=832,
        help="Maximum width of the output video."
    )
    parser.add_argument(
        "--seed_times",
        type=int,
        default=42,
        help="Number of times to seed the random number generator."
    )
    parser.add_argument(
        "--repeat_first_clip",
        default=False,
        action="store_true",
        help="Whether to repeat the first frames."
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether to use tiled encoding.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs='+',
        default=[30, 52],
        help="Tile size.",
    )
    parser.add_argument(
        "--tile_stride",
        type=int,
        nargs='+',
        default=[15, 26],
        help="Tile stride.",
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        # default="Camera moves dynamically accordingly",
        # default="Camera slightly moves accordingly",
        default="none",
        help="Prefix to add before each prompt. Default is 'Camera moves dynamically accordingly'.",
    )
    parser.add_argument(
        "--prompt_repeat_times",
        type=int,
        default=50,
        help="Number of times to repeat each prompt before moving to the next. Default is 1 (no repeat).",
    )
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=6*10**9,
        help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
    )
    
    args = parser.parse_args()
    return args
import torch.distributed as dist

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


if __name__ == '__main__':

    # ============================
    # 初始化多卡
    # ============================
    rank, world_size, local_rank = init_distributed()
    is_main_process = (rank == 0)

    args = parse_args()
    print_args(args)

    common_negative_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    if args.ref_image_path and args.prompt_path:
        prompts = load_prompts_from_file(args.prompt_path)
        ref_image_dir = os.path.dirname(args.ref_image_path)
        ref_image_name = os.path.splitext(os.path.basename(args.ref_image_path))[0]

        test_list_path = [{
            "ref_image_root": ref_image_dir + '/',
            "prompt_name": ref_image_name,
            "prompts": prompts,
            "emotion_word": 'natural',
            "negative_prompt": common_negative_prompt,
            "direct_image_path": args.ref_image_path
        }]
    else:
        ref_image_base = args.data_root
        all_ref_image_dirs = [
            d for d in os.listdir(ref_image_base)
            if os.path.isdir(os.path.join(ref_image_base, d))
        ]
        ref_image_dirs = all_ref_image_dirs

        test_list_path = []

        for category in sorted(os.listdir(args.data_root)):
            category_path = os.path.join(args.data_root, category)
            if not os.path.isdir(category_path):
                continue

            for sample in sorted(os.listdir(category_path)):
                sample_path = os.path.join(category_path, sample)
                if not os.path.isdir(sample_path):
                    continue

                prompt_file = os.path.join(sample_path, "prompt.txt")
                if not os.path.exists(prompt_file):
                    continue

                prompts = load_prompts_from_file(prompt_file)

                test_list_path.append({
                    "ref_image_root": sample_path + "/",
                    "prompt_name": sample,          # car_00001
                    "prompts": prompts,
                    "emotion_word": "natural",
                    "negative_prompt": common_negative_prompt,
                })

    # ============================
    # 按 rank 切分任务
    # ============================
    test_list_path = [
        s for i, s in enumerate(test_list_path)
        if i % world_size == rank
    ]
    print(f"[Rank {rank}] Processing {len(test_list_path)} samples")

    root = os.path.abspath(args.dit_root) + os.sep
    model_manager = ModelManager(device="cuda", train_architecture=args.train_architecture)

    model_manager.load_models(
        [root + "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32,
    )

    os.makedirs(args.output, exist_ok=True)

    model_manager.load_models(
        [
            [
                root + "diffusion_pytorch_model-00001-of-00007.safetensors",
                root + "diffusion_pytorch_model-00002-of-00007.safetensors",
                root + "diffusion_pytorch_model-00003-of-00007.safetensors",
                root + "diffusion_pytorch_model-00004-of-00007.safetensors",
                root + "diffusion_pytorch_model-00005-of-00007.safetensors",
                root + "diffusion_pytorch_model-00006-of-00007.safetensors",
                root + "diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            root + "models_t5_umt5-xxl-enc-bf16.pth",
            root + "Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    extra_module_root = args.extra_module_root
    if extra_module_root:
        if extra_module_root.endswith('.safetensors'):
            safetensors_files = [extra_module_root]
        else:
            safetensors_files = sorted(glob.glob(os.path.join(extra_module_root, "*.safetensors")))
        if not safetensors_files:
            raise FileNotFoundError(f"No LoRA checkpoint found at {extra_module_root}")
        model_manager.load_lora_v2(safetensors_files, lora_alpha=args.lora_alpha)

    pipe = SVIVideoPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        is_test=True,
        filter_path=args.filter_path,
    )

    # pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit)

    # ============================
    #  video → prompt 映射
    # ============================
    video_prompt_map = {}

    for sample_idx, path_dir_per in enumerate(test_list_path):
        print(f"\n{'#'*100}")
        print(f"STARTING SAMPLE {sample_idx + 1}/{len(test_list_path)}: {path_dir_per['prompt_name']}")
        print(f"{'#'*100}")

        # ============================================================
        # 1. 找 reference image
        # ============================================================
        if "direct_image_path" in path_dir_per:
            ref_image_rgb_path = path_dir_per["direct_image_path"]
        else:
            ref_root = path_dir_per["ref_image_root"]

            image_files = sorted([
                f for f in os.listdir(ref_root)
                if f.lower().endswith((".jpg", ".png"))
                and os.path.isfile(os.path.join(ref_root, f))
            ])

            if len(image_files) == 0:
                sub_dirs = sorted([
                    d for d in os.listdir(ref_root)
                    if os.path.isdir(os.path.join(ref_root, d))
                ])

                found = False
                for sd in sub_dirs:
                    sd_path = os.path.join(ref_root, sd)
                    imgs = sorted([
                        f for f in os.listdir(sd_path)
                        if f.lower().endswith((".jpg", ".png"))
                        and os.path.isfile(os.path.join(sd_path, f))
                    ])
                    if len(imgs) > 0:
                        ref_root = sd_path
                        path_dir_per["ref_image_root"] = ref_root
                        image_files = imgs
                        found = True
                        break

                if not found:
                    raise FileNotFoundError(f"No reference image found under {ref_root}")

            ref_image_rgb_path = os.path.join(ref_root, image_files[0])

        emotion_word = path_dir_per.get("emotion_word", "neutral")

        print(f"Reference image: {ref_image_rgb_path}")
        print(f"Available prompts: {len(path_dir_per['prompts'])}")

        # ============================================================
        # 2. 分辨率
        # ============================================================
        height, width = calculate_dimensions(
            ref_image_rgb_path,
            max_width=args.max_width
        )
        print(f"Video dimensions: {width}x{height}")

        # ============================================================
        # 3. 读取 reference image
        # ============================================================
        rand_ref_frame = Image.fromarray(
            cv2.cvtColor(cv2.imread(ref_image_rgb_path), cv2.COLOR_BGR2RGB)
        )
        if rand_ref_frame.mode != 'RGB':
            rand_ref_frame = rand_ref_frame.convert('RGB')

        rand_ref_frame_final_single = rand_ref_frame.resize((width, height))

        num_motion_frames = args.num_motion_frames
        if args.repeat_first_clip:
            rand_ref_frame_final = [rand_ref_frame_final_single] * num_motion_frames
        else:
            rand_ref_frame_final = rand_ref_frame_final_single

        rand_ref_frame_final_gt = torch.from_numpy(
            np.array(rand_ref_frame_final_single)
        ).clone()

        # ============================================================
        # 4. prompt / clip 数量
        # ============================================================
        loaded_prompts = path_dir_per["prompts"]

        if args.use_first_prompt_only:
            num_clips = args.num_clips
        else:
            effective_prompts = len(loaded_prompts) * args.prompt_repeat_times
            num_clips = min(args.num_clips, effective_prompts)

        print(f"Generating {num_clips} clips")

        # ============================================================
        # 5. seeds / buffer
        # ============================================================
        if args.seed_times == -1:
            seeds = [None] * num_clips
        else:
            seeds = list(range(10000))

        video_list = []

        # ============================================================
        # 6. 输出路径
        # ============================================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_name = path_dir_per["prompt_name"]
        # base_filename = (
        #     f"i2v_{ref_name}_{emotion_word}_"
        #     f"cfgt{args.cfg_scale_text}_step{args.num_steps}_{timestamp}"
        # )
        base_filename = f"i2v_{ref_name}"
        
        sample_output_dir = os.path.join(args.output, f"{ref_name}")
        os.makedirs(sample_output_dir, exist_ok=True)
        print(f"Output dir: {sample_output_dir}")

        # ============================================================
        # 7. clip 连载生成
        # ============================================================
        last_prompt_used = None

        for chunk_idx in range(num_clips):
            seed = seeds[chunk_idx]
            if seed is not None:
                seed = int(seed * args.seed_times)

            if args.use_first_prompt_only:
                current_prompt = loaded_prompts[0]
            else:
                prompt_index = (chunk_idx // args.prompt_repeat_times) % len(loaded_prompts)
                current_prompt = loaded_prompts[prompt_index]

            if args.prompt_prefix != "none":
                current_prompt = f"{args.prompt_prefix}, {current_prompt}"

            last_prompt_used = current_prompt

            print(f"\n{'='*80}")
            print(f"SAMPLE: {ref_name}")
            print(f"CHUNK: {chunk_idx+1}/{num_clips}")
            print(f"PROMPT: {current_prompt}")
            print(f"{'='*80}")

            video = pipe(
                prompt=current_prompt,
                negative_prompt=path_dir_per["negative_prompt"],
                input_image=rand_ref_frame_final,
                num_inference_steps=args.num_steps,
                cfg_scale=dict(text=args.cfg_scale_text),
                seed=seed,
                tiled=args.tiled,
                random_ref_frame=rand_ref_frame_final_gt,
                height=height,
                width=width,
                tea_cache_l1_thresh=0.2 if use_teacache else None,
                tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,
                args=args
            )

            rand_ref_frame_final = video[-num_motion_frames:]
            if chunk_idx < num_clips - 1:
                video_list += video[:-num_motion_frames]
            else:
                video_list += video

            chunk_filename_vis = os.path.join(sample_output_dir, f"{base_filename}.mp4")
    
            save_video(video_list, chunk_filename_vis, fps=16, quality=8)

        # ============================================================
        #  记录 JSON 映射
        # ============================================================
        final_video_path = chunk_filename_vis
        final_video_name = os.path.basename(final_video_path)
        video_prompt_map[final_video_name] = last_prompt_used

        print(f"\n{'#'*100}")
        print(f"COMPLETED SAMPLE {sample_idx + 1}/{len(test_list_path)}: {ref_name}")
        print(f"{'#'*100}")

    print(f"\nCompleted processing all {len(test_list_path)} scenarios!")

    use_dist = dist.is_available() and dist.is_initialized()

    rank = dist.get_rank() if use_dist else 0
    is_main_process = (not use_dist) or rank == 0
    world_size = dist.get_world_size() if use_dist else 1

    rank_json_path = os.path.join(args.output, f"video2prompt_rank{rank}.json")
    with open(rank_json_path, "w", encoding="utf-8") as f:
        json.dump(video_prompt_map, f, indent=2, ensure_ascii=False)

    print(f"[Rank {rank}] Saved local json: {rank_json_path}")

    try:
        if use_dist:
            gathered = [None for _ in range(world_size)] if is_main_process else None
            dist.gather_object(video_prompt_map, gathered, dst=0)
        else:
            gathered = [video_prompt_map]
    except Exception as e:
        print(f"[Rank {rank}] gather_object failed, fallback to file merge: {e}")
        gathered = None

    if is_main_process:
        final_map = {}

        if gathered is not None:
            for d in gathered:
                if d is not None:
                    final_map.update(d)
        else:
            for r in range(world_size):
                p = os.path.join(args.output, f"video2prompt_rank{r}.json")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        final_map.update(json.load(f))

        json_path = os.path.join(args.output, "video2prompt.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_map, f, indent=2, ensure_ascii=False)

        print(f"[Rank 0] Saved video2prompt.json with {len(final_map)} entries")