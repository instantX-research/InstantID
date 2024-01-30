import sys
sys.path.append('./')

import os
import cv2
import math
import torch
import random
import numpy as np
import argparse
import logging

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from style_template import styles
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl

import gradio as gr

# Global variables
MAX_SEED = np.iinfo(np.int32).max
LOG_LEVEL = logging.INFO
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"
DEFAULT_MODEL = "wangqixun/YamerMIX_v8"
MODEL_DIRECTORY = "./models"
MAX_SIDE = 1280
MIN_SIDE = 1024

# Set device and torch_dtype
torch_dtype = torch.float16

if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Download ControlNet checkpoint from Hugging Face Hub
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="./checkpoints",
    local_dir_use_symlinks=False
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="./checkpoints",
    local_dir_use_symlinks=False
)

# Download IP-Adapter checkpoint from Hugging Face Hub
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ip-adapter.bin",
    local_dir="./checkpoints",
    local_dir_use_symlinks=False
)

# Load face encoder
app = FaceAnalysis(name="antelopev2", root='./', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f"./checkpoints/ip-adapter.bin"
controlnet_path = f"./checkpoints/ControlNetModel"

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)


def get_pipeline(model_path):
    if model_path.endswith(
        ".ckpt"
    ) or model_path.endswith(".safetensors"):
        scheduler_kwargs = hf_hub_download(
            repo_id="wangqixun/YamerMIX_v8",
            subfolder="scheduler",
            filename="scheduler_config.json",
        )

        (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
            pretrained_model_name_or_path=model_path,
            scheduler_name=None,
            weight_dtype=torch_dtype,
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
            controlnet=controlnet,
        )

    else:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        )

        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if device == "mps":
        pipe.to("mps", torch_dtype)
        pipe.enable_attention_slicing()
    elif device == "cuda":
        pipe.cuda()

    pipe.load_ip_adapter_instantid(face_adapter)

    if device == "mps" or device == "cuda":
        pipe.image_proj_model.to(device)
        pipe.unet.to(device)

    return pipe


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)


def upload_example_to_gallery(images, prompt, style, negative_prompt):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)


def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def remove_tips():
    return gr.update(visible=False)


def get_example():
    case = [
        [
            ['./examples/yann-lecun_resize.jpg'],
            "a man",
            "Snow",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            ['./examples/musk_resize.jpeg'],
            "a man",
            "Mars",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            ['./examples/sam_resize.png'],
            "a man",
            "Jungle",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
        ],
        [
            ['./examples/schmidhuber_resize.png'],
            "a man",
            "Neon",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            ['./examples/kaifu_resize.png'],
            "a man",
            "Vibrant Color",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
    ]
    return case


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
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
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):

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
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative


def generate_image(face_image, pose_image, prompt, negative_prompt, model_path, style_name, enhance_face_region,
                   num_steps, identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, seed,
                   progress=gr.Progress(track_tqdm=True)):
    global MAX_SIDE, MIN_SIDE
    if face_image is None:
        raise gr.Error(f"Cannot find any input face image! Please upload the face image")

    if prompt is None:
        prompt = "a person"

    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    face_image = load_image(face_image[0])
    face_image = resize_img(face_image, MAX_SIDE, MIN_SIDE)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Extract face features
    face_info = app.get(face_image_cv2)

    if len(face_info) == 0:
        raise gr.Error(f"Cannot find any face in the image! Please upload another person image")

    face_info = sorted(face_info, key=lambda x: (x["bbox"][2]-x["bbox"][0])*x["bbox"][3]-x["bbox"][1])[-1]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

    if pose_image is not None:
        pose_image = load_image(pose_image[0])
        pose_image = resize_img(pose_image, MAX_SIDE, MIN_SIDE)
        pose_image_cv2 = convert_from_image_to_cv2(pose_image)

        face_info = app.get(pose_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(f"Cannot find any face in the reference image! Please upload another person image")

        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info['kps'])

        width, height = face_kps.size

    if enhance_face_region:
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None

    generator = torch.Generator(device=device).manual_seed(seed)

    logging.info("Start inference...")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Negative Prompt: {negative_prompt}")

    pipe = get_pipeline(model_path)
    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        control_mask=control_mask,
        controlnet_conditioning_scale=float(identitynet_strength_ratio),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator
    ).images

    return images, gr.update(visible=True)


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_available_models():
    files = []
    extensions = ['.safetensors', '.ckpt']

    for file in os.listdir(MODEL_DIRECTORY):
        if any(file.endswith(ext) for ext in extensions):
            files.append(os.path.join(MODEL_DIRECTORY, file))

    return files


def refresh_models(selected_model):
    models = [DEFAULT_MODEL] + get_available_models()

    if selected_model in models:
        default_model = selected_model
    else:
        default_model = DEFAULT_MODEL

    return gr.Dropdown(
        label="Model path",
        choices=models,
        value=default_model
    )


def launch_ui(launch_kwargs):
    global DEFAULT_MODEL

    title = r"""
    <h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
    """

    description = r"""
    <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

    How to use:<br>
    1. Upload a person image. For multiple person images, we will only detect the biggest face. Make sure face is not too small and not significantly blocked or blurred.
    2. (Optionally) upload another person image as reference pose. If not uploaded, we will use the first person image to extract landmarks. If you use a cropped face at step1, it is recommeneded to upload it to extract a new pose.
    3. Enter a text prompt as done in normal text-to-image models.
    4. Click the <b>Submit</b> button to start customizing.
    5. Share your customizd photo with your friends, enjoy😊!
    """

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
    1. If you're not satisfied with the similarity, try to increase the weight of "IdentityNet Strength" and "Adapter Strength".
    2. If you feel that the saturation is too high, first decrease the Adapter strength. If it is still too high, then decrease the IdentityNet strength.
    3. If you find that text control is not as expected, decrease Adapter strength.
    4. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
    """

    css = '''
    .gradio-container {width: 85% !important}
    '''
    interface = gr.Blocks(
        css=css,
        title="InstantID: Zero-shot Identity-Preserving Generation in Seconds",
        theme=gr.themes.Default()
    )

    with interface:
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():

                # upload face image
                face_files = gr.Files(
                            label="Upload a photo of your face",
                            file_types=["image"]
                        )
                uploaded_faces = gr.Gallery(label="Your images", visible=False, columns=1, rows=1, height=512)
                with gr.Column(visible=False) as clear_button_face:
                    remove_and_reupload_faces = gr.ClearButton(
                        value="Remove and upload new ones",
                        components=face_files,
                        size="sm"
                    )

                # optional: upload a reference pose image
                pose_files = gr.Files(
                            label="Upload a reference pose image (optional)",
                            file_types=["image"]
                        )
                uploaded_poses = gr.Gallery(
                    label="Your images",
                    visible=False,
                    columns=1,
                    rows=1,
                    height=512
                )
                with gr.Column(visible=False) as clear_button_pose:
                    remove_and_reupload_poses = gr.ClearButton(
                        value="Remove and upload new ones",
                        components=pose_files,
                        size="sm"
                    )

                prompt = gr.Textbox(
                    label="Prompt",
                    info="Give simple prompt is enough to achieve good face fidelity",
                    placeholder="A photo of a person",
                    value=""
                )

                submit = gr.Button("Submit", variant="primary")

                # Allow a different model to be selected by loading models from disk
                # and displaying them in a dropdown
                model_choices = [DEFAULT_MODEL] + get_available_models()
                model = gr.Dropdown(
                    label="Model path",
                    choices=model_choices,
                    value=DEFAULT_MODEL
                )
                refresh_button = gr.Button("Refresh Models")
                refresh_button.click(fn=refresh_models, inputs=model, outputs=model)
                style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)

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

                with gr.Accordion(open=False, label="Advanced Options"):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="low quality",
                        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                    )
                    num_steps = gr.Slider(
                        label="Number of sample steps",
                        minimum=20,
                        maximum=100,
                        step=1,
                        value=30,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=5,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

            with gr.Column():
                gallery = gr.Gallery(label="Generated Images")
                usage_tips = gr.Markdown(label="Usage tips of InstantID", value=tips, visible=False)

            face_files.upload(fn=swap_to_gallery, inputs=face_files, outputs=[uploaded_faces, clear_button_face, face_files])
            pose_files.upload(fn=swap_to_gallery, inputs=pose_files, outputs=[uploaded_poses, clear_button_pose, pose_files])

            remove_and_reupload_faces.click(fn=remove_back_to_files, outputs=[uploaded_faces, clear_button_face, face_files])
            remove_and_reupload_poses.click(fn=remove_back_to_files, outputs=[uploaded_poses, clear_button_pose, pose_files])

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
                    face_files,
                    pose_files,
                    prompt,
                    negative_prompt,
                    model,
                    style,
                    enhance_face_region,
                    num_steps,
                    identitynet_strength_ratio,
                    adapter_strength_ratio,
                    guidance_scale,
                    seed
                ],
                outputs=[gallery, usage_tips]
            ).then(
                fn=clear_cuda_cache
            )

        gr.Examples(
            examples=get_example(),
            inputs=[face_files, prompt, style, negative_prompt],
            run_on_click=True,
            fn=upload_example_to_gallery,
            outputs=[uploaded_faces, clear_button_face, face_files],
        )

        gr.Markdown(article)

    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--listen", type=str, default="0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1", help="IP to listen on for connections to Gradio")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Share the Gradio UI")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--medvram", action="store_true", help="Medium VRAM settings")
    parser.add_argument("--lowvram", action="store_true", help="Low VRAM settings")
    parser.add_argument("--username", type=str, default="", help="Username for authentication")
    parser.add_argument("--password", type=str, default="", help="Password for authentication")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=LOG_LEVEL
    )

    logging.getLogger('httpx').setLevel(logging.WARNING)

    launch_kwargs = {
        "server_name": args.listen
    }

    if args.model_path != DEFAULT_MODEL:
        MODEL_PATH = args.model_path
    if args.medvram:
        MAX_SIDE, MIN_SIDE = 1024, 832
    elif args.lowvram:
        MAX_SIDE, MIN_SIDE = 832, 640
    if args.username and args.password:
        launch_kwargs["auth"] = (args.username, args.password)
    if args.server_port:
        launch_kwargs["server_port"] = args.server_port
    if args.inbrowser:
        launch_kwargs["inbrowser"] = args.inbrowser
    if args.share:
        launch_kwargs["share"] = args.share

    logging.info(f'MAX_SIDE: {MAX_SIDE}, MIN_SIDE: {MIN_SIDE}')
    launch_ui(launch_kwargs)
