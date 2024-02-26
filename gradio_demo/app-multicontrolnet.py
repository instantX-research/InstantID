import sys
sys.path.append("./")

from typing import Tuple

import os
import cv2
import math
import torch
import random
import numpy as np
import argparse

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from style_template import styles
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device, torch_gc
from controlnet_util import openpose, get_depth_map, get_canny_image

import gradio as gr


# global variable
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"

# Load face encoder
app = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f"./checkpoints/ip-adapter.bin"
controlnet_path = f"./checkpoints/ControlNetModel"

# Load pipeline face ControlNetModel
controlnet_identitynet = ControlNetModel.from_pretrained(
    controlnet_path, torch_dtype=dtype
)

# controlnet-pose
controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

controlnet_pose = ControlNetModel.from_pretrained(
    controlnet_pose_model, torch_dtype=dtype
).to(device)
controlnet_canny = ControlNetModel.from_pretrained(
    controlnet_canny_model, torch_dtype=dtype
).to(device)
controlnet_depth = ControlNetModel.from_pretrained(
    controlnet_depth_model, torch_dtype=dtype
).to(device)

controlnet_map = {
    "pose": controlnet_pose,
    "canny": controlnet_canny,
    "depth": controlnet_depth,
}
controlnet_map_fn = {
    "pose": openpose,
    "canny": get_canny_image,
    "depth": get_depth_map,
}


def main(pretrained_model_name_or_path="wangqixun/YamerMIX_v8", enable_lcm_arg=False):
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        scheduler_kwargs = hf_hub_download(
            repo_id="wangqixun/YamerMIX_v8",
            subfolder="scheduler",
            filename="scheduler_config.json",
        )

        (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            scheduler_name=None,
            weight_dtype=dtype,
        )

        scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
        pipe = StableDiffusionXLInstantIDPipeline(
            vae=vae,
            text_encoder=text_encoders[0],
            text_encoder_2=text_encoders[1],
            tokenizer=tokenizers[0],
            tokenizer_2=tokenizers[1],
            unet=unet,
            scheduler=scheduler,
            controlnet=[controlnet_identitynet],
        ).to(device)

    else:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            pretrained_model_name_or_path,
            controlnet=[controlnet_identitynet],
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

    pipe.load_ip_adapter_instantid(face_adapter)
    # load and disable LCM
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.disable_lora()

    def toggle_lcm_ui(value):
        if value:
            return (
                gr.update(minimum=0, maximum=100, step=1, value=5),
                gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5),
            )
        else:
            return (
                gr.update(minimum=5, maximum=100, step=1, value=30),
                gr.update(minimum=0.1, maximum=20.0, step=0.1, value=5),
            )

    def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        return seed

    def remove_tips():
        return gr.update(visible=False)

    def get_example():
        case = [
            [
                "./examples/yann-lecun_resize.jpg",
                None,
                "a man",
                "Snow",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
            [
                "./examples/musk_resize.jpeg",
                "./examples/poses/pose2.jpg",
                "a man flying in the sky in Mars",
                "Mars",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
            [
                "./examples/sam_resize.png",
                "./examples/poses/pose4.jpg",
                "a man doing a silly pose wearing a suite",
                "Jungle",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
            ],
            [
                "./examples/schmidhuber_resize.png",
                "./examples/poses/pose3.jpg",
                "a man sit on a chair",
                "Neon",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
            [
                "./examples/kaifu_resize.png",
                "./examples/poses/pose.jpg",
                "a man",
                "Vibrant Color",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
        ]
        return case

    def run_for_examples(face_file, pose_file, prompt, style, negative_prompt):
        return generate_image(
            face_file,
            pose_file,
            prompt,
            negative_prompt,
            style,
            20,  # num_steps
            0.8,  # identitynet_strength_ratio
            0.8,  # adapter_strength_ratio
            0.4,  # pose_strength
            0.3,  # canny_strength
            0.5,  # depth_strength
            ["pose", "canny"],  # controlnet_selection
            5.0,  # guidance_scale
            42,  # seed
            "EulerDiscreteScheduler",  # scheduler
            False,  # enable_LCM
            True,  # enable_Face_Region
        )

    def convert_from_cv2_to_image(img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def convert_from_image_to_cv2(img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_kps(
        image_pil,
        kps,
        color_list=[
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ],
    ):
        stickwidth = 4
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
        kps = np.array(kps)

        w, h = image_pil.size
        out_img = np.zeros([h, w, 3])

        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[index[0]]

            x = kps[index][:, 0]
            y = kps[index][:, 1]
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly(
                (int(np.mean(x)), int(np.mean(y))),
                (int(length / 2), stickwidth),
                int(angle),
                0,
                360,
                1,
            )
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        out_img = (out_img * 0.6).astype(np.uint8)

        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

        out_img_pil = Image.fromarray(out_img.astype(np.uint8))
        return out_img_pil

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

    def apply_style(
        style_name: str, positive: str, negative: str = ""
    ) -> Tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + " " + negative

    def generate_image(
        face_image_path,
        pose_image_path,
        prompt,
        negative_prompt,
        style_name,
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
        progress=gr.Progress(track_tqdm=True),
    ):

        if enable_LCM:
            pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
            pipe.enable_lora()
        else:
            pipe.disable_lora()
            scheduler_class_name = scheduler.split("-")[0]

            add_kwargs = {}
            if len(scheduler.split("-")) > 1:
                add_kwargs["use_karras_sigmas"] = True
            if len(scheduler.split("-")) > 2:
                add_kwargs["algorithm_type"] = "sde-dpmsolver++"
            scheduler = getattr(diffusers, scheduler_class_name)
            pipe.scheduler = scheduler.from_config(pipe.scheduler.config, **add_kwargs)

        if face_image_path is None:
            raise gr.Error(
                f"Cannot find any input face image! Please upload the face image"
            )

        if prompt is None:
            prompt = "a person"

        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        face_image = load_image(face_image_path)
        face_image = resize_img(face_image, max_side=1024)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = app.get(face_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(
                f"Unable to detect a face in the image. Please upload a different photo with a clear face."
            )

        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
        img_controlnet = face_image
        if pose_image_path is not None:
            pose_image = load_image(pose_image_path)
            pose_image = resize_img(pose_image, max_side=1024)
            img_controlnet = pose_image
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)

            face_info = app.get(pose_image_cv2)

            if len(face_info) == 0:
                raise gr.Error(
                    f"Cannot find any face in the reference image! Please upload another person image"
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
            pipe.controlnet = MultiControlNetModel(
                [controlnet_identitynet]
                + [controlnet_map[s] for s in controlnet_selection]
            )
            control_scales = [float(identitynet_strength_ratio)] + [
                controlnet_scales[s] for s in controlnet_selection
            ]
            control_images = [face_kps] + [
                controlnet_map_fn[s](img_controlnet).resize((width, height))
                for s in controlnet_selection
            ]
        else:
            pipe.controlnet = controlnet_identitynet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps

        generator = torch.Generator(device=device).manual_seed(seed)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

        pipe.set_ip_adapter_scale(adapter_strength_ratio)
        images = pipe(
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
        ).images

        return images[0], gr.update(visible=True)

    # Description
    title = r"""
    <h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
    """

    description = r"""
    <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

    How to use:<br>
    1. Upload an image with a face. For images with multiple faces, we will only detect the largest face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
    2. (Optional) You can upload another image as a reference for the face pose. If you don't, we will use the first detected face image to extract facial landmarks. If you use a cropped face at step 1, it is recommended to upload it to define a new face pose.
    3. (Optional) You can select multiple ControlNet models to control the generation process. The default is to use the IdentityNet only. The ControlNet models include pose skeleton, canny, and depth. You can adjust the strength of each ControlNet model to control the generation process.
    4. Enter a text prompt, as done in normal text-to-image models.
    5. Click the <b>Submit</b> button to begin customization.
    6. Share your customized photo with your friends and enjoy! 😊"""

    article = r"""
    ---
    📝 **Citation**
    <br>
    If our work is helpful for your research or applications, please cite us via:
    ```bibtex
    @article{wang2024instantid,
    title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
    author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
    journal={arXiv preprint arXiv:2401.07519},
    year={2024}
    }
    ```
    📧 **Contact**
    <br>
    If you have any questions, please feel free to open an issue or directly reach us out at <b>haofanwang.ai@gmail.com</b>.
    """

    tips = r"""
    ### Usage tips of InstantID
    1. If you're not satisfied with the similarity, try increasing the weight of "IdentityNet Strength" and "Adapter Strength."    
    2. If you feel that the saturation is too high, first decrease the Adapter strength. If it remains too high, then decrease the IdentityNet strength.
    3. If you find that text control is not as expected, decrease Adapter strength.
    4. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
    """

    css = """
    .gradio-container {width: 85% !important}
    """
    with gr.Blocks(css=css) as demo:
        # description
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                with gr.Row(equal_height=True):
                    # upload face image
                    face_file = gr.Image(
                        label="Upload a photo of your face", type="filepath"
                    )
                    # optional: upload a reference pose image
                    pose_file = gr.Image(
                        label="Upload a reference pose image (Optional)",
                        type="filepath",
                    )

                # prompt
                prompt = gr.Textbox(
                    label="Prompt",
                    info="Give simple prompt is enough to achieve good face fidelity",
                    placeholder="A photo of a person",
                    value="",
                )

                submit = gr.Button("Submit", variant="primary")
                enable_LCM = gr.Checkbox(
                    label="Enable Fast Inference with LCM", value=enable_lcm_arg,
                    info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
                )
                style = gr.Dropdown(
                    label="Style template",
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                )

                # strength
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength (for detail)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                with gr.Accordion("Controlnet"):
                    controlnet_selection = gr.CheckboxGroup(
                        ["pose", "canny", "depth"], label="Controlnet", value=["pose"],
                        info="Use pose for skeleton inference, canny for edge detection, and depth for depth map estimation. You can try all three to control the generation process"
                    )
                    pose_strength = gr.Slider(
                        label="Pose strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
                    )
                    canny_strength = gr.Slider(
                        label="Canny strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
                    )
                    depth_strength = gr.Slider(
                        label="Depth strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
                    )
                with gr.Accordion(open=False, label="Advanced Options"):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="low quality",
                        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                    )
                    num_steps = gr.Slider(
                        label="Number of sample steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=5 if enable_lcm_arg else 30,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=20.0,
                        step=0.1,
                        value=0.0 if enable_lcm_arg else 5.0,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    schedulers = [
                        "DEISMultistepScheduler",
                        "HeunDiscreteScheduler",
                        "EulerDiscreteScheduler",
                        "DPMSolverMultistepScheduler",
                        "DPMSolverMultistepScheduler-Karras",
                        "DPMSolverMultistepScheduler-Karras-SDE",
                    ]
                    scheduler = gr.Dropdown(
                        label="Schedulers",
                        choices=schedulers,
                        value="EulerDiscreteScheduler",
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

            with gr.Column(scale=1):
                gallery = gr.Image(label="Generated Images")
                usage_tips = gr.Markdown(
                    label="InstantID Usage Tips", value=tips, visible=False
                )

            submit.click(
                fn=remove_tips,
                outputs=usage_tips,
            ).then(
                fn=randomize_seed_fn,
                inputs=[seed, randomize_seed],
                outputs=seed,
                queue=False,
                api_name=False,
            ).then(
                fn=generate_image,
                inputs=[
                    face_file,
                    pose_file,
                    prompt,
                    negative_prompt,
                    style,
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
                ],
                outputs=[gallery, usage_tips],
            )

            enable_LCM.input(
                fn=toggle_lcm_ui,
                inputs=[enable_LCM],
                outputs=[num_steps, guidance_scale],
                queue=False,
            )

        gr.Examples(
            examples=get_example(),
            inputs=[face_file, pose_file, prompt, style, negative_prompt],
            fn=run_for_examples,
            outputs=[gallery, usage_tips],
            cache_examples=True,
        )

        gr.Markdown(article)

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="wangqixun/YamerMIX_v8"
    )
    parser.add_argument(
        "--enable_LCM", type=bool, default=os.environ.get("ENABLE_LCM", False)
    )
    args = parser.parse_args()

    main(args.pretrained_model_name_or_path, args.enable_LCM)