# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys

import time
import subprocess
from cog import BasePredictor, Input, Path

import cv2
import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

# for `ip-adaper`, `ControlNetModel`, and `stable-diffusion-xl-base-1.0`
CHECKPOINTS_CACHE = "./checkpoints"
CHECKPOINTS_URL = (
    "https://weights.replicate.delivery/default/InstantID/checkpoints.tar"
)

# for `models/antelopev2`
MODELS_CACHE = "./models"
MODELS_URL = "https://weights.replicate.delivery/default/InstantID/models.tar"


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
            offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(CHECKPOINTS_CACHE):
            download_weights(CHECKPOINTS_URL, CHECKPOINTS_CACHE)

        if not os.path.exists(MODELS_CACHE):
            download_weights(MODELS_URL, MODELS_CACHE)

        self.width, self.height = 640, 640
        self.app = FaceAnalysis(
            name="antelopev2",
            root="./",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(self.width, self.height))

        # Path to InstantID models
        face_adapter = f"./checkpoints/ip-adapter.bin"
        controlnet_path = f"./checkpoints/ControlNetModel"

        # Load pipeline
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        )

        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        )
        self.pipe.cuda()
        self.pipe.load_ip_adapter_instantid(face_adapter)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Input prompt",
            default="analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=640,
            ge=512,
            le=2048,
        ),
        height: int = Input(
            description="Height of output image",
            default=640,
            ge=512,
            le=2048,
        ),
        ip_adapter_scale: float = Input(
            description="Scale for IP adapter",
            default=0.8,
            ge=0,
            le=1,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Scale for ControlNet conditioning",
            default=0.8,
            ge=0,
            le=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=30,
            ge=1,
            le=500,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=5,
            ge=1,
            le=50,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if self.width != width or self.height != height:
            print(f"[!] Resizing output to {width}x{height}")
            self.width = width
            self.height = height
            self.app.prepare(ctx_id=0, det_size=(self.width, self.height))

        face_image = load_image(str(image))
        face_image = resize_img(face_image)

        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            reverse=True,
        )[
            0
        ]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(face_image, face_info["kps"])

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        output_path = "result.jpg"
        image.save(output_path)
        return Path(output_path)
