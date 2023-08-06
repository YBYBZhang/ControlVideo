# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import numpy as np
import argparse
import imageio
import torch

from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import controlnet_aux
from controlnet_aux import OpenposeDetector, CannyDetector, MidasDetector

from models.pipeline_controlvideo import ControlVideoPipeline
from models.util import save_videos_grid, read_video, get_annotation
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D
from models.RIFE.IFNet_HDv3 import IFNet
from cog import BasePredictor, Input, Path

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

POS_PROMPT = "best quality, extremely detailed, HD, ultra-realistic, 8K, high quality, masterpiece, trending on artstation, art"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder"
        ).to(dtype=torch.float16)
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(
            dtype=torch.float16
        )
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            sd_path, subfolder="unet"
        ).to(dtype=torch.float16)
        self.interpolater = IFNet(ckpt_path=inter_path).to(dtype=torch.float16)
        self.scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.controlnet = {
            k: ControlNetModel3D.from_pretrained_2d(controlnet_dict[k]).to(
                dtype=torch.float16
            )
            for k in ["depth", "canny", "pose"]
        }
        self.annotator = {k: controlnet_parser_dict[k]() for k in ["depth", "canny"]}
        self.annotator["pose"] = controlnet_parser_dict["pose"].from_pretrained(
            "lllyasviel/ControlNet", cache_dir="checkpoints"
        )

    def predict(
        self,
        prompt: str = Input(
            description="Text description of target video",
            default="A striking mallard floats effortlessly on the sparkling pond.",
        ),
        video_path: Path = Input(description="source video"),
        condition: str = Input(
            default="depth",
            choices=["depth", "canny", "pose"],
            description="Condition of structure sequence",
        ),
        video_length: int = Input(
            default=15, description="Length of synthesized video"
        ),
        smoother_steps: str = Input(
            default="19, 20",
            description="Timesteps at which using interleaved-frame smoother, separate with comma",
        ),
        is_long_video: bool = Input(
            default=False,
            description="Whether to use hierarchical sampler to produce long video",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=12.5
        ),
        seed: str = Input(
            default=None, description="Random seed. Leave blank to randomize the seed"
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        else:
            seed = int(seed)
        print(f"Using seed: {seed}")

        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)

        pipe = ControlVideoPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet[condition],
            interpolater=self.interpolater,
            scheduler=self.scheduler,
        )

        pipe.enable_vae_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to("cuda")

        # Step 1. Read a video
        video = read_video(video_path=str(video_path), video_length=video_length)

        # Step 2. Parse a video to conditional frames
        pil_annotation = get_annotation(video, self.annotator[condition])

        # Step 3. inference
        smoother_steps = [int(s) for s in smoother_steps.split(",")]

        if is_long_video:
            window_size = int(np.sqrt(video_length))
            sample = pipe.generate_long_video(
                prompt + POS_PROMPT,
                video_length=video_length,
                frames=pil_annotation,
                num_inference_steps=num_inference_steps,
                smooth_steps=smoother_steps,
                window_size=window_size,
                generator=generator,
                guidance_scale=guidance_scale,
                negative_prompt=NEG_PROMPT,
            ).videos
        else:
            sample = pipe(
                prompt + POS_PROMPT,
                video_length=video_length,
                frames=pil_annotation,
                num_inference_steps=num_inference_steps,
                smooth_steps=smoother_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                negative_prompt=NEG_PROMPT,
            ).videos

        out_path = "/tmp/out.mp4"
        save_videos_grid(sample, out_path)
        del pipe
        torch.cuda.empty_cache()

        return Path(out_path)
