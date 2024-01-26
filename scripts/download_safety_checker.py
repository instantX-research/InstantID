# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPFeatureExtractor
from transformers import CLIPImageProcessor


safety = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    torch_dtype=torch.float16,
)
safety.save_pretrained("./safety-cache")

fe = feature_extractor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
)
fe.save_pretrained("./feature-extractor")
