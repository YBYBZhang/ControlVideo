import os
import numpy as np
import argparse
import imageio
import torch

from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

import torchvision
from controlnet_aux.processor import Processor

from models.pipeline_controlvideo import ControlVideoPipeline
from models.util import save_videos_grid, read_video
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D
from models.RIFE.IFNet_HDv3 import IFNet


device = "cuda"
sd_path = "checkpoints/stable-diffusion-v1-5"
inter_path = "checkpoints/flownet.pkl"
controlnet_dict_version = {
    "v10":{
        "openpose": "checkpoints/sd-controlnet-openpose",
        "depth_midas": "checkpoints/sd-controlnet-depth",
        "canny": "checkpoints/sd-controlnet-canny",
    },
    "v11": {
    "softedge_pidinet": "checkpoints/control_v11p_sd15_softedge",
    "softedge_pidsafe": "checkpoints/control_v11p_sd15_softedge",
    "softedge_hed": "checkpoints/control_v11p_sd15_softedge",
    "softedge_hedsafe": "checkpoints/control_v11p_sd15_softedge",
    "scribble_hed": "checkpoints/control_v11p_sd15_scribble",
    "scribble_pidinet": "checkpoints/control_v11p_sd15_scribble",
    "lineart_anime": "checkpoints/control_v11p_sd15_lineart_anime",
    "lineart_coarse": "checkpoints/control_v11p_sd15_lineart",
    "lineart_realistic": "checkpoints/control_v11p_sd15_lineart",
    "depth_midas": "checkpoints/control_v11f1p_sd15_depth",
    "depth_leres": "checkpoints/control_v11f1p_sd15_depth",
    "depth_leres++": "checkpoints/control_v11f1p_sd15_depth",
    "depth_zoe": "checkpoints/control_v11f1p_sd15_depth",
    "canny": "checkpoints/control_v11p_sd15_canny",
    "openpose": "checkpoints/control_v11p_sd15_openpose",
    "openpose_face": "checkpoints/control_v11p_sd15_openpose",
    "openpose_faceonly": "checkpoints/control_v11p_sd15_openpose",
    "openpose_full": "checkpoints/control_v11p_sd15_openpose",
    "openpose_hand": "checkpoints/control_v11p_sd15_openpose",
    "normal_bae": "checkpoints/control_v11p_sd15_normalbae"
    }
}
# load processor from processor_id
# options are:
# ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
#  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
#  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
#  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
#  "softedge_pidinet", "softedge_pidsafe"]

POS_PROMPT = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text description of target video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to a source video")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Directory of output")
    parser.add_argument("--condition", type=str, default="depth", help="Condition of structure sequence")
    parser.add_argument("--video_length", type=int, default=15, help="Length of synthesized video")
    parser.add_argument("--height", type=int, default=512, help="Height of synthesized video, and should be a multiple of 32")
    parser.add_argument("--width", type=int, default=512, help="Width of synthesized video, and should be a multiple of 32")
    parser.add_argument("--smoother_steps", nargs='+', default=[19, 20], type=int, help="Timesteps at which using interleaved-frame smoother")
    parser.add_argument("--is_long_video", action='store_true', help="Whether to use hierarchical sampler to produce long video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of generator")
    parser.add_argument("--version", type=str, default='v10', choices=["v10", "v11"], help="Version of ControlNet")
    parser.add_argument("--frame_rate", type=int, default=None, help="The frame rate of loading input video. Default rate is computed according to video length.")
    parser.add_argument("--temp_video_name", type=str, default=None, help="Default video name")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    # Height and width should be a multiple of 32
    args.height = (args.height // 32) * 32    
    args.width = (args.width // 32) * 32    

    processor = Processor(args.condition)
    controlnet_dict = controlnet_dict_version[args.version]
    
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
    video = read_video(video_path=args.video_path, video_length=args.video_length, width=args.width, height=args.height, frame_rate=args.frame_rate)

    # Save source video
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    save_videos_grid(original_pixels, os.path.join(args.output_path, "source_video.mp4"), rescale=True)


    # Step 2. Parse a video to conditional frames
    t2i_transform = torchvision.transforms.ToPILImage()
    pil_annotation = []
    for frame in video:
        pil_frame = t2i_transform(frame)
        pil_annotation.append(processor(pil_frame, to_pil=True))

    # Save condition video
    video_cond = [np.array(p).astype(np.uint8) for p in pil_annotation]
    imageio.mimsave(os.path.join(args.output_path, f"{args.condition}_condition.mp4"), video_cond, fps=8)

    # Reduce memory (optional)
    del processor; torch.cuda.empty_cache()

    # Step 3. inference

    if args.is_long_video:
        window_size = int(np.sqrt(args.video_length))
        sample = pipe.generate_long_video(args.prompt + POS_PROMPT, video_length=args.video_length, frames=pil_annotation, 
                    num_inference_steps=50, smooth_steps=args.smoother_steps, window_size=window_size,
                    generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                    width=args.width, height=args.height
                ).videos
    else:
        sample = pipe(args.prompt + POS_PROMPT, video_length=args.video_length, frames=pil_annotation, 
                    num_inference_steps=50, smooth_steps=args.smoother_steps,
                    generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                    width=args.width, height=args.height
                ).videos
    args.temp_video_name = args.prompt if args.temp_video_name is None else args.temp_video_name
    save_videos_grid(sample, f"{args.output_path}/{args.temp_video_name}.mp4")