import os
import numpy as np
import argparse
import imageio
import torch

from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
# from annotator.canny import CannyDetector
# from annotator.openpose import OpenposeDetector
# from annotator.midas import MidasDetector
# import sys
# sys.path.insert(0, ".")
import controlnet_aux
from controlnet_aux import OpenposeDetector, CannyDetector, MidasDetector

from models.pipeline_controlvideo import ControlVideoPipeline
from models.util import save_videos_grid, read_video, get_annotation
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D
from models.RIFE.IFNet_HDv3 import IFNet


device = "cuda"
sd_path = "checkpoints/stable-diffusion-v1-5"
inter_path = "checkpoints/flownet.pkl"
controlnet_dict = {
    "pose": "checkpoints/sd-controlnet-openpose",
    "depth": "checkpoints/sd-controlnet-depth",
    "canny": "checkpoints/sd-controlnet-canny",
}

controlnet_parser_dict = {
    "pose": OpenposeDetector,
    "depth": MidasDetector,
    "canny": CannyDetector,
}
  
POS_PROMPT = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text description of target video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to a source video")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Directory of output")
    parser.add_argument("--condition", type=str, default="depth", help="Condition of structure sequence")
    parser.add_argument("--video_length", type=int, default=15, help="Length of synthesized video")
    parser.add_argument("--smoother_steps", nargs='+', default=[19, 20], type=int, help="Timesteps at which using interleaved-frame smoother")
    parser.add_argument("--is_long_video", action='store_true', help="Whether to use hierarchical sampler to produce long video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of generator")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    # if args.condition == "canny":
    annotator = controlnet_parser_dict[args.condition]()
    # else:
    #     annotator = controlnet_parser_dict[args.condition].from_pretrained("lllyasviel/ControlNet")

    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(controlnet_dict[args.condition]).to(dtype=torch.float16)
    interpolater = IFNet(ckpt_path=inter_path).to(dtype=torch.float16)
    scheduler=DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = ControlVideoPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet, interpolater=interpolater, scheduler=scheduler,
        )
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    # Step 1. Read a video
    video = read_video(video_path=args.video_path, video_length=args.video_length)

    # Save source video
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    save_videos_grid(original_pixels, os.path.join(args.output_path, "source_video.mp4"), rescale=True)


    # Step 2. Parse a video to conditional frames
    pil_annotation = get_annotation(video, annotator)
    if args.condition == "depth" and controlnet_aux.__version__ == '0.0.1':
        pil_annotation = [pil_annot[0] for pil_annot in pil_annotation]

    # Save condition video
    video_cond = [np.array(p).astype(np.uint8) for p in pil_annotation]
    imageio.mimsave(os.path.join(args.output_path, f"{args.condition}_condition.mp4"), video_cond, fps=8)

    # Reduce memory (optional)
    del annotator; torch.cuda.empty_cache()

    # Step 3. inference

    if args.is_long_video:
        window_size = int(np.sqrt(args.video_length))
        sample = pipe.generate_long_video(args.prompt + POS_PROMPT, video_length=args.video_length, frames=pil_annotation, 
                    num_inference_steps=50, smooth_steps=args.smoother_steps, window_size=window_size,
                    generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT
                ).videos
    else:
        sample = pipe(args.prompt + POS_PROMPT, video_length=args.video_length, frames=pil_annotation, 
                    num_inference_steps=50, smooth_steps=args.smoother_steps,
                    generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT
                ).videos
    save_videos_grid(sample, f"{args.output_path}/{args.prompt}.mp4")