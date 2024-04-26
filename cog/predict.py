# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../gradio_demo"))

import cv2
import time
import torch
import mimetypes
import subprocess
import numpy as np
from typing import List
from cog import BasePredictor, Input, Path

import PIL
from PIL import Image

import diffusers
from diffusers import LCMScheduler
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from model_util import get_torch_device
from insightface.app import FaceAnalysis
from transformers import CLIPImageProcessor
from controlnet_util import openpose, get_depth_map, get_canny_image

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

mimetypes.add_type("image/webp", ".webp")

# GPU global variables
DEVICE = get_torch_device()
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32

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
    # These are all huggingface models that we host via gcp + pget
    "stable-diffusion-xl-base-1.0": {
        "slug": "stabilityai/stable-diffusion-xl-base-1.0",
        "url": "https://weights.replicate.delivery/default/InstantID/models--stabilityai--stable-diffusion-xl-base-1.0.tar",
        "path": "checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
    },
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
    "RealVisXL_V3.0_Turbo": {
        "slug": "SG161222/RealVisXL_V3.0_Turbo",
        "url": "https://weights.replicate.delivery/default/InstantID/models--SG161222--RealVisXL_V3.0_Turbo.tar",
        "path": "checkpoints/models--SG161222--RealVisXL_V3.0_Turbo",
    },
    "RealVisXL_V4.0_Lightning": {
        "slug": "SG161222/RealVisXL_V4.0_Lightning",
        "url": "https://weights.replicate.delivery/default/InstantID/models--SG161222--RealVisXL_V4.0_Lightning.tar",
        "path": "checkpoints/models--SG161222--RealVisXL_V4.0_Lightning",
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
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image


def download_weights(url, dest):
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    command = ["pget", "-vf", url, dest]
    if ".tar" in url:
        command.append("-x")
    try:
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(CHECKPOINTS_CACHE):
            download_weights(CHECKPOINTS_URL, CHECKPOINTS_CACHE)

        if not os.path.exists(MODELS_CACHE):
            download_weights(MODELS_URL, MODELS_CACHE)

        self.app = FaceAnalysis(
            name="antelopev2",
            root="./",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Path to InstantID models
        self.face_adapter = f"./checkpoints/ip-adapter.bin"
        controlnet_path = f"./checkpoints/ControlNetModel"

        # Load pipeline face ControlNetModel
        self.controlnet_identitynet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        )
        self.setup_extra_controlnets()

        self.load_weights("stable-diffusion-xl-base-1.0")
        self.setup_safety_checker()

    def setup_safety_checker(self):
        print(f"[~] Seting up safety checker")

        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE,
            torch_dtype=DTYPE,
            local_files_only=True,
        )
        self.safety_checker.to(DEVICE)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            DEVICE
        )
        np_image = np.array(image)
        image, has_nsfw_concept = self.safety_checker(
            images=[np_image],
            clip_input=safety_checker_input.pixel_values.to(DTYPE),
        )
        return image, has_nsfw_concept

    def load_weights(self, sdxl_weights):
        self.base_weights = sdxl_weights
        weights_info = SDXL_NAME_TO_PATHLIKE[self.base_weights]

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
                controlnet=[self.controlnet_identitynet],
                torch_dtype=DTYPE,
                cache_dir=CHECKPOINTS_CACHE,
                local_files_only=True,
                safety_checker=None,
                feature_extractor=None,
            )
            self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        else:  # e.g. .safetensors, NOTE: This functionality is not being used right now
            self.pipe.from_single_file(
                path_to_weights_file,
                controlnet=self.controlnet_identitynet,
                torch_dtype=DTYPE,
                cache_dir=CHECKPOINTS_CACHE,
            )

        self.pipe.load_ip_adapter_instantid(self.face_adapter)
        self.setup_lcm_lora()
        self.pipe.cuda()

    def setup_lcm_lora(self):
        print(f"[~] Seting up LCM (just in case)")

        lcm_lora_key = "models--latent-consistency--lcm-lora-sdxl"
        lcm_lora_path = f"checkpoints/{lcm_lora_key}"
        if not os.path.exists(lcm_lora_path):
            download_weights(
                f"https://weights.replicate.delivery/default/InstantID/{lcm_lora_key}.tar",
                lcm_lora_path,
            )
        self.pipe.load_lora_weights(
            "latent-consistency/lcm-lora-sdxl",
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
            weight_name="pytorch_lora_weights.safetensors",
        )
        self.pipe.disable_lora()

    def setup_extra_controlnets(self):
        print(f"[~] Seting up pose, canny, depth ControlNets")

        controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
        controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
        controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

        for controlnet_key in [
            "models--diffusers--controlnet-canny-sdxl-1.0",
            "models--diffusers--controlnet-depth-sdxl-1.0-small",
            "models--thibaud--controlnet-openpose-sdxl-1.0",
        ]:
            controlnet_path = f"checkpoints/{controlnet_key}"
            if not os.path.exists(controlnet_path):
                download_weights(
                    f"https://weights.replicate.delivery/default/InstantID/{controlnet_key}.tar",
                    controlnet_path,
                )

        controlnet_pose = ControlNetModel.from_pretrained(
            controlnet_pose_model,
            torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        ).to(DEVICE)
        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_canny_model,
            torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        ).to(DEVICE)
        controlnet_depth = ControlNetModel.from_pretrained(
            controlnet_depth_model,
            torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        ).to(DEVICE)

        self.controlnet_map = {
            "pose": controlnet_pose,
            "canny": controlnet_canny,
            "depth": controlnet_depth,
        }
        self.controlnet_map_fn = {
            "pose": openpose,
            "canny": get_canny_image,
            "depth": get_depth_map,
        }

    def generate_image(
        self,
        face_image_path,
        pose_image_path,
        prompt,
        negative_prompt,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        pose_strength,
        canny_strength,
        depth_strength,
        controlnet_selection,
        guidance_scale,
        seed,
        scheduler,
        enable_LCM,
        enhance_face_region,
        num_images_per_prompt,
    ):
        if enable_LCM:
            self.pipe.enable_lora()
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.disable_lora()
            scheduler_class_name = scheduler.split("-")[0]

            add_kwargs = {}
            if len(scheduler.split("-")) > 1:
                add_kwargs["use_karras_sigmas"] = True
            if len(scheduler.split("-")) > 2:
                add_kwargs["algorithm_type"] = "sde-dpmsolver++"
            scheduler = getattr(diffusers, scheduler_class_name)
            self.pipe.scheduler = scheduler.from_config(
                self.pipe.scheduler.config,
                **add_kwargs,
            )

        if face_image_path is None:
            raise Exception(
                f"Cannot find any input face `image`! Please upload the face `image`"
            )

        face_image = load_image(face_image_path)
        face_image = resize_img(face_image)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = self.app.get(face_image_cv2)

        if len(face_info) == 0:
            raise Exception(
                "Face detector could not find a face in the `image`. Please use a different `image` as input."
            )

        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
        )[
            -1
        ]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

        img_controlnet = face_image
        if pose_image_path is not None:
            pose_image = load_image(pose_image_path)
            pose_image = resize_img(pose_image, max_side=1024)
            img_controlnet = pose_image
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)

            face_info = self.app.get(pose_image_cv2)

            if len(face_info) == 0:
                raise Exception(
                    "Face detector could not find a face in the `pose_image`. Please use a different `pose_image` as input."
                )

            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info["kps"])

            width, height = face_kps.size

        if enhance_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None

        if len(controlnet_selection) > 0:
            controlnet_scales = {
                "pose": pose_strength,
                "canny": canny_strength,
                "depth": depth_strength,
            }
            self.pipe.controlnet = MultiControlNetModel(
                [self.controlnet_identitynet]
                + [self.controlnet_map[s] for s in controlnet_selection]
            )
            control_scales = [float(identitynet_strength_ratio)] + [
                controlnet_scales[s] for s in controlnet_selection
            ]
            control_images = [face_kps] + [
                self.controlnet_map_fn[s](img_controlnet).resize((width, height))
                for s in controlnet_selection
            ]
        else:
            self.pipe.controlnet = self.controlnet_identitynet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

        self.pipe.set_ip_adapter_scale(adapter_strength_ratio)
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=control_images,
            control_mask=control_mask,
            controlnet_conditioning_scale=control_scales,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        return images

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
                "RealVisXL_V3.0_Turbo",
                "RealVisXL_V4.0_Lightning",
            ],
        ),
        scheduler: str = Input(
            description="Scheduler",
            choices=[
                "DEISMultistepScheduler",
                "HeunDiscreteScheduler",
                "EulerDiscreteScheduler",
                "DPMSolverMultistepScheduler",
                "DPMSolverMultistepScheduler-Karras",
                "DPMSolverMultistepScheduler-Karras-SDE",
            ],
            default="EulerDiscreteScheduler",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=30,
            ge=1,
            le=500,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=1,
            le=50,
        ),
        ip_adapter_scale: float = Input(
            description="Scale for image adapter strength (for detail)",  # adapter_strength_ratio
            default=0.8,
            ge=0,
            le=1.5,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Scale for IdentityNet strength (for fidelity)",  # identitynet_strength_ratio
            default=0.8,
            ge=0,
            le=1.5,
        ),
        enable_pose_controlnet: bool = Input(
            description="Enable Openpose ControlNet, overrides strength if set to false",
            default=True,
        ),
        pose_strength: float = Input(
            description="Openpose ControlNet strength, effective only if `enable_pose_controlnet` is true",
            default=0.4,
            ge=0,
            le=1,
        ),
        enable_canny_controlnet: bool = Input(
            description="Enable Canny ControlNet, overrides strength if set to false",
            default=False,
        ),
        canny_strength: float = Input(
            description="Canny ControlNet strength, effective only if `enable_canny_controlnet` is true",
            default=0.3,
            ge=0,
            le=1,
        ),
        enable_depth_controlnet: bool = Input(
            description="Enable Depth ControlNet, overrides strength if set to false",
            default=False,
        ),
        depth_strength: float = Input(
            description="Depth ControlNet strength, effective only if `enable_depth_controlnet` is true",
            default=0.5,
            ge=0,
            le=1,
        ),
        enable_lcm: bool = Input(
            description="Enable Fast Inference with LCM (Latent Consistency Models) - speeds up inference steps, trade-off is the quality of the generated image. Performs better with close-up portrait face images",
            default=False,
        ),
        lcm_num_inference_steps: int = Input(
            description="Only used when `enable_lcm` is set to True, Number of denoising steps when using LCM",
            default=5,
            ge=1,
            le=10,
        ),
        lcm_guidance_scale: float = Input(
            description="Only used when `enable_lcm` is set to True, Scale for classifier-free guidance when using LCM",
            default=1.5,
            ge=1,
            le=20,
        ),
        enhance_nonface_region: bool = Input(
            description="Enhance non-face region", default=True
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            default=1,
            ge=1,
            le=8,
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

        # Set up ControlNet selection and their respective strength values (if any)
        controlnet_selection = []
        if pose_strength > 0 and enable_pose_controlnet:
            controlnet_selection.append("pose")
        if canny_strength > 0 and enable_canny_controlnet:
            controlnet_selection.append("canny")
        if depth_strength > 0 and enable_depth_controlnet:
            controlnet_selection.append("depth")

        # Switch to LCM inference steps and guidance scale if LCM is enabled
        if enable_lcm:
            num_inference_steps = lcm_num_inference_steps
            guidance_scale = lcm_guidance_scale

        # Generate
        images = self.generate_image(
            face_image_path=str(image),
            pose_image_path=str(pose_image) if pose_image else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_inference_steps,
            identitynet_strength_ratio=controlnet_conditioning_scale,
            adapter_strength_ratio=ip_adapter_scale,
            pose_strength=pose_strength,
            canny_strength=canny_strength,
            depth_strength=depth_strength,
            controlnet_selection=controlnet_selection,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            seed=seed,
            enable_LCM=enable_lcm,
            enhance_face_region=enhance_nonface_region,
            num_images_per_prompt=num_outputs,
        )

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

            extension = output_format.lower()
            extension = "jpeg" if extension == "jpg" else extension
            output_path = f"/tmp/out_{i}.{extension}"

            print(f"[~] Saving to {output_path}...")
            print(f"[~] Output format: {extension.upper()}")
            if output_format != "png":
                print(f"[~] Output quality: {output_quality}")

            save_params = {"format": extension.upper()}
            if output_format != "png":
                save_params["quality"] = output_quality
                save_params["optimize"] = True

            output_image.save(output_path, **save_params)
            output_paths.append(Path(output_path))
        return output_paths
