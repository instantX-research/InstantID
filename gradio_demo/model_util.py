from typing import Literal, Union, Optional, Tuple, List

import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import (
    UNet2DConditionModel,
    SchedulerMixin,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
)
from safetensors.torch import load_file
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)

from omegaconf import OmegaConf

# DiffUsers版StableDiffusionのモデルパラメータ
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.0120

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 64  # fixed from old invalid value `32`
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8
# UNET_PARAMS_USE_LINEAR_PROJECTION = False

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

# V2
V2_UNET_PARAMS_ATTENTION_HEAD_DIM = [5, 10, 20, 20]
V2_UNET_PARAMS_CONTEXT_DIM = 1024
# V2_UNET_PARAMS_USE_LINEAR_PROJECTION = True

TOKENIZER_V1_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TOKENIZER_V2_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a", "euler", "uniPC"]

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

DIFFUSERS_CACHE_DIR = None  # if you want to change the cache dir, change this


def load_checkpoint_with_text_encoder_conversion(ckpt_path: str, device="cpu"):
    # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
    TEXT_ENCODER_KEY_REPLACEMENTS = [
        (
            "cond_stage_model.transformer.embeddings.",
            "cond_stage_model.transformer.text_model.embeddings.",
        ),
        (
            "cond_stage_model.transformer.encoder.",
            "cond_stage_model.transformer.text_model.encoder.",
        ),
        (
            "cond_stage_model.transformer.final_layer_norm.",
            "cond_stage_model.transformer.text_model.final_layer_norm.",
        ),
    ]

    if ckpt_path.endswith(".safetensors"):
        checkpoint = None
        state_dict = load_file(ckpt_path)  # , device) # may causes error
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            checkpoint = None

    key_reps = []
    for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
        for key in state_dict.keys():
            if key.startswith(rep_from):
                new_key = rep_to + key[len(rep_from) :]
                key_reps.append((key, new_key))

    for key, new_key in key_reps:
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return checkpoint, state_dict


def create_unet_diffusers_config(v2, use_linear_projection_in_v2=False):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    # unet_params = original_config.model.params.unet_config.params

    block_out_channels = [
        UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT
    ]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = (
            "CrossAttnDownBlock2D"
            if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS
            else "DownBlock2D"
        )
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = (
            "CrossAttnUpBlock2D"
            if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS
            else "UpBlock2D"
        )
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=UNET_PARAMS_IMAGE_SIZE,
        in_channels=UNET_PARAMS_IN_CHANNELS,
        out_channels=UNET_PARAMS_OUT_CHANNELS,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
        cross_attention_dim=UNET_PARAMS_CONTEXT_DIM
        if not v2
        else V2_UNET_PARAMS_CONTEXT_DIM,
        attention_head_dim=UNET_PARAMS_NUM_HEADS
        if not v2
        else V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
        # use_linear_projection=UNET_PARAMS_USE_LINEAR_PROJECTION if not v2 else V2_UNET_PARAMS_USE_LINEAR_PROJECTION,
    )
    if v2 and use_linear_projection_in_v2:
        config["use_linear_projection"] = True

    return config


def load_diffusers_model(
    pretrained_model_name_or_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
) -> Tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel,]:
    if v2:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V2_MODEL_NAME,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            # default is clip skip 2
            num_hidden_layers=24 - (clip_skip - 1) if clip_skip is not None else 23,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V1_MODEL_NAME,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            num_hidden_layers=12 - (clip_skip - 1) if clip_skip is not None else 12,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    return tokenizer, text_encoder, unet, vae


def load_checkpoint_model(
    checkpoint_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
) -> Tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel,]:
    pipe = StableDiffusionPipeline.from_single_file(
        checkpoint_path,
        upcast_attention=True if v2 else False,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    _, state_dict = load_checkpoint_with_text_encoder_conversion(checkpoint_path)
    unet_config = create_unet_diffusers_config(v2, use_linear_projection_in_v2=v2)
    unet_config["class_embed_type"] = None
    unet_config["addition_embed_type"] = None
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, unet_config)
    unet = UNet2DConditionModel(**unet_config)
    unet.load_state_dict(converted_unet_checkpoint)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    if clip_skip is not None:
        if v2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    del pipe

    return tokenizer, text_encoder, unet, vae


def load_models(
    pretrained_model_name_or_path: str,
    scheduler_name: str,
    v2: bool = False,
    v_pred: bool = False,
    weight_dtype: torch.dtype = torch.float32,
) -> Tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, SchedulerMixin,]:
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        tokenizer, text_encoder, unet, vae = load_checkpoint_model(
            pretrained_model_name_or_path, v2=v2, weight_dtype=weight_dtype
        )
    else:  # diffusers
        tokenizer, text_encoder, unet, vae = load_diffusers_model(
            pretrained_model_name_or_path, v2=v2, weight_dtype=weight_dtype
        )

    if scheduler_name:
        scheduler = create_noise_scheduler(
            scheduler_name,
            prediction_type="v_prediction" if v_pred else "epsilon",
        )
    else:
        scheduler = None

    return tokenizer, text_encoder, unet, scheduler, vae


def load_diffusers_model_xl(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> Tuple[List[CLIPTokenizer], List[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    # returns tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet

    tokenizers = [
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            pad_token_id=0,  # same as open clip
        ),
    ]

    text_encoders = [
        CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
    ]

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    return tokenizers, text_encoders, unet, vae


def load_checkpoint_model_xl(
    checkpoint_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> Tuple[List[CLIPTokenizer], List[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    unet = pipe.unet
    vae = pipe.vae
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    if len(text_encoders) == 2:
        text_encoders[1].pad_token_id = 0

    del pipe

    return tokenizers, text_encoders, unet, vae


def load_models_xl(
    pretrained_model_name_or_path: str,
    scheduler_name: str,
    weight_dtype: torch.dtype = torch.float32,
    noise_scheduler_kwargs=None,
) -> Tuple[
    List[CLIPTokenizer],
    List[SDXL_TEXT_ENCODER_TYPE],
    UNet2DConditionModel,
    SchedulerMixin,
]:
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        (tokenizers, text_encoders, unet, vae) = load_checkpoint_model_xl(
            pretrained_model_name_or_path, weight_dtype
        )
    else:  # diffusers
        (tokenizers, text_encoders, unet, vae) = load_diffusers_model_xl(
            pretrained_model_name_or_path, weight_dtype
        )
    if scheduler_name:
        scheduler = create_noise_scheduler(scheduler_name, noise_scheduler_kwargs)
    else:
        scheduler = None

    return tokenizers, text_encoders, unet, scheduler, vae

def create_noise_scheduler(
    scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
    noise_scheduler_kwargs=None,
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    name = scheduler_name.lower().replace(" ", "_")
    if name.lower() == "ddim":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
        scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    elif name.lower() == "ddpm":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
        scheduler = DDPMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    elif name.lower() == "lms":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/lms_discrete
        scheduler = LMSDiscreteScheduler(
            **OmegaConf.to_container(noise_scheduler_kwargs)
        )
    elif name.lower() == "euler_a":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
        scheduler = EulerAncestralDiscreteScheduler(
            **OmegaConf.to_container(noise_scheduler_kwargs)
        )
    elif name.lower() == "euler":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
        scheduler = EulerDiscreteScheduler(
            **OmegaConf.to_container(noise_scheduler_kwargs)
        )
    elif name.lower() == "unipc":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/unipc
        scheduler = UniPCMultistepScheduler(
            **OmegaConf.to_container(noise_scheduler_kwargs)
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

    return scheduler


def torch_gc():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


from enum import Enum


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


cpu_state = CPUState.GPU
xpu_available = False
directml_enabled = False


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        xpu_available = True
except:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass


def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu")
        else:
            return torch.device(torch.cuda.current_device())
