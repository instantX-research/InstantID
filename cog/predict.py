# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys

import time
import subprocess
from typing import List

from cog import BasePredictor, Input, Path

import cv2
import PIL
import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor


# for `ip-adapter`, `ControlNetModel`, and `stable-diffusion-xl-base-1.0`
CHECKPOINTS_CACHE = "./checkpoints"
CHECKPOINTS_URL = "https://weights.replicate.delivery/default/InstantID/checkpoints.tar"

# for `models/antelopev2`
MODELS_CACHE = "./models"
MODELS_URL = "https://weights.replicate.delivery/default/InstantID/models.tar"

# for the safety checker
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/playgroundai/safety-cache.tar"

SDXL_NAME_TO_PATHLIKE = {
    # `stable-diffusion-xl-base-1.0` is the default model, it's speical since it's always on disk (downloaded in setup)
    "stable-diffusion-xl-base-1.0": {
        "slug": "stabilityai/stable-diffusion-xl-base-1.0",
    },
    # These are all huggingface models that we host via gcp + pget
    "afrodite-xl-v2": {
        "slug": "stablediffusionapi/afrodite-xl-v2",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--afrodite-xl-v2.tar",
        "path": "checkpoints/models--stablediffusionapi--afrodite-xl-v2",
    },
    "albedobase-xl-20": {
        "slug": "stablediffusionapi/albedobase-xl-20",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--albedobase-xl-20.tar",
        "path": "checkpoints/models--stablediffusionapi--albedobase-xl-20",
    },
    "albedobase-xl-v13": {
        "slug": "stablediffusionapi/albedobase-xl-v13",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--albedobase-xl-v13.tar",
        "path": "checkpoints/models--stablediffusionapi--albedobase-xl-v13",
    },
    "animagine-xl-30": {
        "slug": "stablediffusionapi/animagine-xl-30",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--animagine-xl-30.tar",
        "path": "checkpoints/models--stablediffusionapi--animagine-xl-30",
    },
    "anime-art-diffusion-xl": {
        "slug": "stablediffusionapi/anime-art-diffusion-xl",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--anime-art-diffusion-xl.tar",
        "path": "checkpoints/models--stablediffusionapi--anime-art-diffusion-xl",
    },
    "anime-illust-diffusion-xl": {
        "slug": "stablediffusionapi/anime-illust-diffusion-xl",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--anime-illust-diffusion-xl.tar",
        "path": "checkpoints/models--stablediffusionapi--anime-illust-diffusion-xl",
    },
    "dreamshaper-xl": {
        "slug": "stablediffusionapi/dreamshaper-xl",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--dreamshaper-xl.tar",
        "path": "checkpoints/models--stablediffusionapi--dreamshaper-xl",
    },
    "dynavision-xl-v0610": {
        "slug": "stablediffusionapi/dynavision-xl-v0610",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--dynavision-xl-v0610.tar",
        "path": "checkpoints/models--stablediffusionapi--dynavision-xl-v0610",
    },
    "guofeng4-xl": {
        "slug": "stablediffusionapi/guofeng4-xl",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--guofeng4-xl.tar",
        "path": "checkpoints/models--stablediffusionapi--guofeng4-xl",
    },
    "juggernaut-xl-v8": {
        "slug": "stablediffusionapi/juggernaut-xl-v8",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--juggernaut-xl-v8.tar",
        "path": "checkpoints/models--stablediffusionapi--juggernaut-xl-v8",
    },
    "nightvision-xl-0791": {
        "slug": "stablediffusionapi/nightvision-xl-0791",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--nightvision-xl-0791.tar",
        "path": "checkpoints/models--stablediffusionapi--nightvision-xl-0791",
    },
    "omnigen-xl": {
        "slug": "stablediffusionapi/omnigen-xl",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--omnigen-xl.tar",
        "path": "checkpoints/models--stablediffusionapi--omnigen-xl",
    },
    "pony-diffusion-v6-xl": {
        "slug": "stablediffusionapi/pony-diffusion-v6-xl",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--pony-diffusion-v6-xl.tar",
        "path": "checkpoints/models--stablediffusionapi--pony-diffusion-v6-xl",
    },
    "protovision-xl-high-fidel": {
        "slug": "stablediffusionapi/protovision-xl-high-fidel",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stablediffusionapi--protovision-xl-high-fidel.tar",
        "path": "checkpoints/models--stablediffusionapi--protovision-xl-high-fidel",
    },
    # These are non-huggingface models, e.g. .safetensors files
    "RealVisXL_V3.0": {
        "url": "https://weights.replicate.delivery/default/comfy-ui/checkpoints/RealVisXL_V3.0.safetensors.tar",
        "path": "checkpoints/RealVisXL_V3.0",
        "file": "RealVisXL_V3.0.safetensors",
    },
}


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
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
        self.face_adapter = f"./checkpoints/ip-adapter.bin"
        controlnet_path = f"./checkpoints/ControlNetModel"

        # Load pipeline
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        )

        self.base_weights = "stable-diffusion-xl-base-1.0"
        weights_info = SDXL_NAME_TO_PATHLIKE[self.base_weights]
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            weights_info["slug"],
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        )

        self.pipe.cuda()
        self.pipe.load_ip_adapter_instantid(self.face_adapter)
        self.setup_safety_checker()

    def setup_safety_checker(self):
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.safety_checker.to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = np.array(image)  # Convert the single image to a NumPy array
        image, has_nsfw_concept = self.safety_checker(
            images=[np_image],  # Pass the NumPy array inside a list
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def load_weights(self, sdxl_weights):
        self.base_weights = sdxl_weights
        weights_info = SDXL_NAME_TO_PATHLIKE[self.base_weights]

        if sdxl_weights == "stable-diffusion-xl-base-1.0":  # Default, it's always there
            self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                weights_info["slug"],
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                cache_dir=CHECKPOINTS_CACHE,
                local_files_only=True,
            )
            self.pipe.cuda()
            self.pipe.load_ip_adapter_instantid(self.face_adapter)
            self.setup_safety_checker()
            return

        download_url = weights_info["url"]
        path_to_weights_dir = weights_info["path"]
        if not os.path.exists(path_to_weights_dir):
            download_weights(download_url, path_to_weights_dir)

        is_hugging_face_model = "slug" in weights_info.keys()
        path_to_weights_file = os.path.join(
            path_to_weights_dir,
            weights_info.get("file", ""),
        )

        print(f"[~] Loading new SDXL weights: {path_to_weights_file}")
        if is_hugging_face_model:
            self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                weights_info["slug"],
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                cache_dir=CHECKPOINTS_CACHE,
                local_files_only=True,
            )
        else:  # e.g. .safetensors
            self.pipe.from_single_file(
                path_to_weights_file,
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                cache_dir=CHECKPOINTS_CACHE,
            )
        self.pipe.cuda()
        self.pipe.load_ip_adapter_instantid(self.face_adapter)

    def predict(
        self,
        image: Path = Input(
            description="Input face image",
        ),
        pose_image: Path = Input(
            description="(Optional) reference pose image",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="a person",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        sdxl_weights: str = Input(
            description="Pick which base weights you want to use",
            default="stable-diffusion-xl-base-1.0",
            choices=[
                "stable-diffusion-xl-base-1.0",
                "juggernaut-xl-v8",
                "afrodite-xl-v2",
                "albedobase-xl-20",
                "albedobase-xl-v13",
                "animagine-xl-30",
                "anime-art-diffusion-xl",
                "anime-illust-diffusion-xl",
                "dreamshaper-xl",
                "dynavision-xl-v0610",
                "guofeng4-xl",
                "nightvision-xl-0791",
                "omnigen-xl",
                "pony-diffusion-v6-xl",
                "protovision-xl-high-fidel",
                "RealVisXL_V3.0",
            ],
        ),
        width: int = Input(
            description="Width of output image",
            default=640,
            ge=512,
            le=4096,
        ),
        height: int = Input(
            description="Height of output image",
            default=640,
            ge=512,
            le=4096,
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
        ip_adapter_scale: float = Input(
            description="Scale for image adapter strength (for detail)",
            default=0.8,
            ge=0,
            le=1.5,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Scale for IdentityNet strength (for fidelity)",
            default=0.8,
            ge=0,
            le=1.5,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # If no seed is provided, generate a random seed
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Load the weights if they are different from the base weights
        if sdxl_weights != self.base_weights:
            self.load_weights(sdxl_weights)

        # Run the prediction process
        # Resize the output if the provided dimensions are different from the current ones
        if self.width != width or self.height != height:
            print(f"[!] Resizing output to {width}x{height}")
            self.width = width
            self.height = height
            self.app.prepare(ctx_id=0, det_size=(self.width, self.height))

        # Load and resize the face image
        face_image = load_image(str(image))
        face_image = resize_img(face_image)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = self.app.get(face_image_cv2)

        # Raise an exception if no face is found in the image
        if len(face_info) == 0:
            raise Exception(
                "Cannot find any face in the image! Please upload another person image"
            )

        # Use the largest face found in the image
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
        )[-1]
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

        # If a pose image is provided, use it to extract the pose
        if pose_image is not None:
            pose_image = load_image(str(pose_image))
            pose_image = resize_img(pose_image)
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)
            face_info = self.app.get(pose_image_cv2)

            # Raise an exception if no face is found in the pose image
            if len(face_info) == 0:
                raise Exception(
                    "Cannot find any face in the reference image! Please upload another person image"
                )

            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info["kps"])

            width, height = face_kps.size

        # Set the seed for the random number generator
        generator = torch.Generator(device="cuda").manual_seed(seed)

        print("Start inference...")
        print(f"[!] Prompt: {prompt}, \n[!] Neg Prompt: {negative_prompt}")

        # Set the scale for the image adapter and run the pipeline
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images

        # Save the generated images and check for NSFW content
        output_paths = []
        for i, output_image in enumerate(images):
            if not disable_safety_checker:
                _, has_nsfw_content_list = self.run_safety_checker(output_image)
                has_nsfw_content = any(has_nsfw_content_list)
                print(f"NSFW content detected: {has_nsfw_content}")
                if has_nsfw_content:
                    raise Exception(
                        "NSFW content detected. Try running it again, or try a different prompt."
                    )
            output_path = f"/tmp/out_{i}.png"
            output_image.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths
