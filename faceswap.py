# !pip install opencv-python transformers accelerate insightface
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler
import cv2
import torch
import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import draw_kps
from pipeline_stable_diffusion_xl_instantid_inpaint import StableDiffusionXLInstantIDInpaintPipeline

from PIL import Image

def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
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

def prepare_average_embeding(face_list):
    face_emebdings = []
    for face_path in face_list:
      face_image = load_image(face_path)
      face_image = resize_img(face_image)
      face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
      face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
      face_emb = face_info['embedding']
      face_emebdings.append(face_emb)

    return sum(face_emebdings) / len(face_emebdings)

def prepareMaskAndPoseAndControlImage(pose_image, face_info, padding = 50, mask_grow = 20, resize = True):
    if padding < mask_grow:
        raise ValueError('mask_grow cannot be greater than padding')

    kps = face_info['kps']
    width, height = pose_image.size

    x1, y1, x2, y2 = face_info['bbox']
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # check if image can contain padding
    p_x1 = padding if x1 - padding > 0 else x1
    p_y1 = padding if y1 - padding > 0 else y1
    p_x2 = padding if x2 + padding < width else width - x2
    p_y2 = padding if y2 + padding < height else height - y2

    p_x1, p_y1, p_x2, p_y2 = int(p_x1), int(p_y1), int(p_x2), int(p_y2)

    numpy_array = np.array(pose_image)
    index_y1 = y1 - p_y1; index_y2 = y2 + p_y2
    index_x1 = x1 - p_x1; index_x2 = x2 + p_x2

    # cut the face with paddings
    img = numpy_array[index_y1:index_y2, index_x1:index_x2]

    img = Image.fromarray(img.astype(np.uint8))
    original_width, original_height = img.size

    # mask
    mask = np.array(img)
    mask[:, :] = 0

    m_px1 =  p_x1 - mask_grow if p_x1 - mask_grow > 0 else 0
    m_py1 =  p_y1 - mask_grow if p_y1 - mask_grow > 0 else 0
    m_px2 =  mask_grow if original_width - p_x2 + mask_grow < original_width  else original_width - p_x2
    m_py2 =  mask_grow if original_height - p_y2 + mask_grow < original_height else original_height - p_y2

    mask[
        m_py1:(original_height - p_y2 + m_py2),
        m_px1:(original_width - p_x2 + m_px2)
    ] = 255

    mask = Image.fromarray(mask.astype(np.uint8))

    # resize image and KPS
    kps -= [index_x1, index_y1]
    if resize:
        mask = resize_img(mask)
        img = resize_img(img)
        new_width, new_height = img.size
        kps *= [new_width / original_width, new_height / original_height]
    control_image = draw_kps(img, kps)

    # (mask, pose, control), (original positon of face with padding: x, y, w, h)
    return (mask, img, control_image), (index_x1, index_y1, original_width, original_height)


if __name__ == '__main__':

    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    # LCM Lora path ( https://huggingface.co/latent-consistency/lcm-lora-sdxl )
    adapter_id = 'loras/pytorch_lora_weights.safetensors'

    # You can use any base XL model (do not use models for inpainting!)
    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

    pipe = StableDiffusionXLInstantIDInpaintPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load adapter
    pipe.load_ip_adapter_instantid(face_adapter)
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    # prepare images
    face_emb = prepare_average_embeding([
        'examples/kaifu_resize.png', # ..., ...
    ])

    #pose_image = load_image('examples/kaifu_resize.png')
    pose_image = load_image('examples/musk_resize.jpeg')
    face_info = app.get(cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face

    images, position = prepareMaskAndPoseAndControlImage(
        pose_image,
        face_info,
        60,  # padding
        40,  # grow mask
        True # resize
    )
    mask, pose_image_preprocessed, control_image = images

    prompt = ''
    negative_prompt = '(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured'

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        control_image=control_image,
        image=pose_image_preprocessed,
        mask_image=mask,
        controlnet_conditioning_scale=0.6,
        ip_adapter_scale=0.3, # keep it low
        num_inference_steps=11,
        guidance_scale=0.6
    ).images[0]

    image.save('face.jpg')

    # integrate cropped result into the image
    x, y, w, h = position

    image = image.resize((w, h))
    pose_image.paste(image, (x, y))
    pose_image.save('result.jpg')