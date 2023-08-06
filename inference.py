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
from params_proto import PrefixProto, Flag

device = "cuda"
sd_path = "checkpoints/stable-diffusion-v1-5"
inter_path = "checkpoints/flownet.pkl"
controlnet_dict = {
    "openpose": "checkpoints/sd-controlnet-openpose",
    "depth_midas": "checkpoints/sd-controlnet-depth",
    "canny": "checkpoints/sd-controlnet-canny",
}

POS_PROMPT = "best quality, extremely detailed, HD, realistic, 8K, masterpiece, trending on artstation, art, smooth"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"


class Lucid(PrefixProto):
    """
    prompt: Text description of target video
    condition: Condition of structure sequence
    video_path: Path to a source video
    output_path: Directory of output
    video_length: Length of synthesized video [IN FRAMES]
    smoother_steps: Timesteps at which using interleaved-frame smoother
    width: Width of synthesized video, and should be a multiple of 32
    height: Height of synthesized video, and should be a multiple of 32
    frame_rate: The frame rate of loading input video. [DEFAULT RATE IS COMPUTED ACCORDING TO VIDEO LENGTH.]
    is_long_video: Whether to use hierarchical sampler to produce long videoRandom seed of generator
    seed: Random seed of generator
    """

    prompt: str = ""
    video_path: str = ""
    output_path: str = ""
    sample_vid_name: str = ""
    condition: str = "canny"
    video_length: int = 50
    smoother_steps: list = [19, 20]
    width: int = 512
    height: int = 512
    frame_rate: int = None
    is_long_video: bool = False
    seed: int = 101

# def get_Lucid():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--prompt", Text description of target video
#     parser.add_argument("--video_path", Path to a source video
#     parser.add_argument("--output_path", Directory of output
#     parser.add_argument("--condition", Condition of structure sequence
#     parser.add_argument("--video_length", Length of synthesized video
#     parser.add_argument("--height", type=Height of synthesized video, and should be a multiple of 32
#     parser.add_argument("--width", type=Width of synthesized video, and should be a multiple of 32
#     parser.add_argument("--smoother_steps", nLucid='+', default=[19, Timesteps at which using interleaved-frame smoother
#     parser.add_argument("--Whether to use hierarchical sampler to produce long video
#     parser.add_argument("--seed", Random seed of generator
#     parser.add_argument("--frame_rate", type=The frame rate of loading input video. Default rate is computed according to video length.
#     parser.add_argument("--sample_vid_name", Default video name
#
#     Lucid = parser.parse_Lucid()
#     return Lucid


if __name__ == "__main__":
    Lucid.prompt = "Walking over stairs, first-person view, no background, no tree, blue sky and sun."
    Lucid.video_path = "data/log2.mp4"
    Lucid.output_path = "Lucid_sim"
    Lucid.sample_vid_name = "sample001"

    os.makedirs(Lucid.output_path, exist_ok=True)

    # Height and width should be a multiple of 32
    Lucid.height = (Lucid.height // 32) * 32
    Lucid.width = (Lucid.width // 32) * 32

    processor = Processor(Lucid.condition)
    # controlnet_dict = controlnet_dict_version[Lucid.version]

    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(controlnet_dict[Lucid.condition]).to(dtype=torch.float16)
    interpolater = IFNet(ckpt_path=inter_path).to(dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = ControlVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet, interpolater=interpolater, scheduler=scheduler,
    )
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(Lucid.seed)

    # Step 1. Read a video
    video = read_video(video_path=Lucid.video_path, video_length=Lucid.video_length, width=Lucid.width, height=Lucid.height,
                       frame_rate=Lucid.frame_rate)

    # Save source video
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    save_videos_grid(original_pixels, os.path.join(Lucid.output_path, "source_video.mp4"), rescale=True)

    # Step 2. Parse a video to conditional frames
    t2i_transform = torchvision.transforms.ToPILImage()
    pil_annotation = []
    for frame in video:
        pil_frame = t2i_transform(frame)
        pil_annotation.append(processor(pil_frame, to_pil=True))

    # Save condition video
    video_cond = [np.array(p).astype(np.uint8) for p in pil_annotation]
    imageio.mimsave(os.path.join(Lucid.output_path, f"{Lucid.condition}_condition.mp4"), video_cond, fps=8)

    # Reduce memory (optional)
    del processor;
    torch.cuda.empty_cache()

    # Step 3. inference

    if Lucid.is_long_video:
        window_size = int(np.sqrt(Lucid.video_length))
        sample = pipe.generate_long_video(Lucid.prompt + POS_PROMPT, video_length=Lucid.video_length,
                                          frames=pil_annotation,
                                          num_inference_steps=50, smooth_steps=Lucid.smoother_steps,
                                          window_size=window_size,
                                          generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                                          width=Lucid.width, height=Lucid.height
                                          ).videos
    else:
        sample = pipe(Lucid.prompt + POS_PROMPT, video_length=Lucid.video_length, frames=pil_annotation,
                      num_inference_steps=50, smooth_steps=Lucid.smoother_steps,
                      generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                      width=Lucid.width, height=Lucid.height
                      ).videos
    save_videos_grid(sample, f"{Lucid.output_path}/{Lucid.sample_vid_name}.mp4")
