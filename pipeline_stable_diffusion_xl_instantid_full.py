# Copyright 2024 The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import math

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from diffusers.image_processor import PipelineImageInput

from diffusers.models import ControlNetModel

from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils.import_utils import is_xformers_available

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from ip_adapter.attention_processor import region_control

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate insightface
        >>> import diffusers
        >>> from diffusers.utils import load_image
        >>> from diffusers.models import ControlNetModel

        >>> import cv2
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        
        >>> from insightface.app import FaceAnalysis
        >>> from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

        >>> # download 'antelopev2' under ./models
        >>> app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        >>> app.prepare(ctx_id=0, det_size=(640, 640))
        
        >>> # download models under ./checkpoints
        >>> face_adapter = f'./checkpoints/ip-adapter.bin'
        >>> controlnet_path = f'./checkpoints/ControlNetModel'
        
        >>> # load IdentityNet
        >>> controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        
        >>> pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
        ... )
        >>> pipe.cuda()
        
        >>> # load adapter
        >>> pipe.load_ip_adapter_instantid(face_adapter)

        >>> prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
        >>> negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

        >>> # load an image
        >>> image = load_image("your-example.jpg")
        
        >>> face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))[-1]
        >>> face_emb = face_info['embedding']
        >>> face_kps = draw_kps(face_image, face_info['kps'])
        
        >>> pipe.set_ip_adapter_scale(0.8)

        >>> # generate image
        >>> image = pipe(
        ...     prompt, image_embeds=face_emb, image=face_kps, controlnet_conditioning_scale=0.8
        ... ).images[0]
        ```
"""

from transformers import CLIPTokenizer
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
class LongPromptWeight(object):
    
    """
    Copied from https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion_xl.py
    """
    
    def __init__(self) -> None:
        pass

    def parse_prompt_attention(self, text):
        """
        Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
        Accepted tokens are:
        (abc) - increases attention to abc by a multiplier of 1.1
        (abc:3.12) - increases attention to abc by a multiplier of 3.12
        [abc] - decreases attention to abc by a multiplier of 1.1
        \( - literal character '('
        \[ - literal character '['
        \) - literal character ')'
        \] - literal character ']'
        \\ - literal character '\'
        anything else - just text

        >>> parse_prompt_attention('normal text')
        [['normal text', 1.0]]
        >>> parse_prompt_attention('an (important) word')
        [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
        >>> parse_prompt_attention('(unbalanced')
        [['unbalanced', 1.1]]
        >>> parse_prompt_attention('\(literal\]')
        [['(literal]', 1.0]]
        >>> parse_prompt_attention('(unnecessary)(parens)')
        [['unnecessaryparens', 1.1]]
        >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
        [['a ', 1.0],
        ['house', 1.5730000000000004],
        [' ', 1.1],
        ['on', 1.0],
        [' a ', 1.1],
        ['hill', 0.55],
        [', sun, ', 1.1],
        ['sky', 1.4641000000000006],
        ['.', 1.1]]
        """
        import re

        re_attention = re.compile(
            r"""
                \\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|
                \)|]|[^\\()\[\]:]+|:
            """,
            re.X,
        )

        re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

        res = []
        round_brackets = []
        square_brackets = []

        round_bracket_multiplier = 1.1
        square_bracket_multiplier = 1 / 1.1

        def multiply_range(start_position, multiplier):
            for p in range(start_position, len(res)):
                res[p][1] *= multiplier

        for m in re_attention.finditer(text):
            text = m.group(0)
            weight = m.group(1)

            if text.startswith("\\"):
                res.append([text[1:], 1.0])
            elif text == "(":
                round_brackets.append(len(res))
            elif text == "[":
                square_brackets.append(len(res))
            elif weight is not None and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), float(weight))
            elif text == ")" and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), round_bracket_multiplier)
            elif text == "]" and len(square_brackets) > 0:
                multiply_range(square_brackets.pop(), square_bracket_multiplier)
            else:
                parts = re.split(re_break, text)
                for i, part in enumerate(parts):
                    if i > 0:
                        res.append(["BREAK", -1])
                    res.append([part, 1.0])

        for pos in round_brackets:
            multiply_range(pos, round_bracket_multiplier)

        for pos in square_brackets:
            multiply_range(pos, square_bracket_multiplier)

        if len(res) == 0:
            res = [["", 1.0]]

        # merge runs of identical weights
        i = 0
        while i + 1 < len(res):
            if res[i][1] == res[i + 1][1]:
                res[i][0] += res[i + 1][0]
                res.pop(i + 1)
            else:
                i += 1

        return res

    def get_prompts_tokens_with_weights(self, clip_tokenizer: CLIPTokenizer, prompt: str):
        """
        Get prompt token ids and weights, this function works for both prompt and negative prompt

        Args:
            pipe (CLIPTokenizer)
                A CLIPTokenizer
            prompt (str)
                A prompt string with weights

        Returns:
            text_tokens (list)
                A list contains token ids
            text_weight (list)
                A list contains the correspodent weight of token ids

        Example:
            import torch
            from transformers import CLIPTokenizer

            clip_tokenizer = CLIPTokenizer.from_pretrained(
                "stablediffusionapi/deliberate-v2"
                , subfolder = "tokenizer"
                , dtype = torch.float16
            )

            token_id_list, token_weight_list = get_prompts_tokens_with_weights(
                clip_tokenizer = clip_tokenizer
                ,prompt = "a (red:1.5) cat"*70
            )
        """
        texts_and_weights = self.parse_prompt_attention(prompt)
        text_tokens, text_weights = [], []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = clip_tokenizer(word, truncation=False).input_ids[1:-1]  # so that tokenize whatever length prompt
            # the returned token is a 1d list: [320, 1125, 539, 320]

            # merge the new tokens to the all tokens holder: text_tokens
            text_tokens = [*text_tokens, *token]

            # each token chunk will come with one weight, like ['red cat', 2.0]
            # need to expand weight for each token.
            chunk_weights = [weight] * len(token)

            # append the weight back to the weight holder: text_weights
            text_weights = [*text_weights, *chunk_weights]
        return text_tokens, text_weights

    def group_tokens_and_weights(self, token_ids: list, weights: list, pad_last_block=False):
        """
        Produce tokens and weights in groups and pad the missing tokens

        Args:
            token_ids (list)
                The token ids from tokenizer
            weights (list)
                The weights list from function get_prompts_tokens_with_weights
            pad_last_block (bool)
                Control if fill the last token list to 75 tokens with eos
        Returns:
            new_token_ids (2d list)
            new_weights (2d list)

        Example:
            token_groups,weight_groups = group_tokens_and_weights(
                token_ids = token_id_list
                , weights = token_weight_list
            )
        """
        bos, eos = 49406, 49407

        # this will be a 2d list
        new_token_ids = []
        new_weights = []
        while len(token_ids) >= 75:
            # get the first 75 tokens
            head_75_tokens = [token_ids.pop(0) for _ in range(75)]
            head_75_weights = [weights.pop(0) for _ in range(75)]

            # extract token ids and weights
            temp_77_token_ids = [bos] + head_75_tokens + [eos]
            temp_77_weights = [1.0] + head_75_weights + [1.0]

            # add 77 token and weights chunk to the holder list
            new_token_ids.append(temp_77_token_ids)
            new_weights.append(temp_77_weights)

        # padding the left
        if len(token_ids) >= 0:
            padding_len = 75 - len(token_ids) if pad_last_block else 0

            temp_77_token_ids = [bos] + token_ids + [eos] * padding_len + [eos]
            new_token_ids.append(temp_77_token_ids)

            temp_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
            new_weights.append(temp_77_weights)

        return new_token_ids, new_weights

    def get_weighted_text_embeddings_sdxl(
        self,
        pipe: StableDiffusionXLPipeline,
        prompt: str = "",
        prompt_2: str = None,
        neg_prompt: str = "",
        neg_prompt_2: str = None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        extra_emb=None,
        extra_emb_alpha=0.6,
    ):
        """
        This function can process long prompt with weights, no length limitation
        for Stable Diffusion XL

        Args:
            pipe (StableDiffusionPipeline)
            prompt (str)
            prompt_2 (str)
            neg_prompt (str)
            neg_prompt_2 (str)
        Returns:
            prompt_embeds (torch.Tensor)
            neg_prompt_embeds (torch.Tensor)
        """
        # 
        if prompt_embeds is not None and \
            negative_prompt_embeds is not None and \
            pooled_prompt_embeds is not None and \
            negative_pooled_prompt_embeds is not None:
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

        if prompt_2:
            prompt = f"{prompt} {prompt_2}"

        if neg_prompt_2:
            neg_prompt = f"{neg_prompt} {neg_prompt_2}"

        eos = pipe.tokenizer.eos_token_id

        # tokenizer 1
        prompt_tokens, prompt_weights = self.get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
        neg_prompt_tokens, neg_prompt_weights = self.get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)

        # tokenizer 2
        # prompt_tokens_2, prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt)
        # neg_prompt_tokens_2, neg_prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt)
        # tokenizer 2 遇到 !! !!!! 等多感叹号和tokenizer 1的效果不一致
        prompt_tokens_2, prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
        neg_prompt_tokens_2, neg_prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)

        # padding the shorter one for prompt set 1
        prompt_token_len = len(prompt_tokens)
        neg_prompt_token_len = len(neg_prompt_tokens)

        if prompt_token_len > neg_prompt_token_len:
            # padding the neg_prompt with eos token
            neg_prompt_tokens = neg_prompt_tokens + [eos] * abs(prompt_token_len - neg_prompt_token_len)
            neg_prompt_weights = neg_prompt_weights + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        else:
            # padding the prompt
            prompt_tokens = prompt_tokens + [eos] * abs(prompt_token_len - neg_prompt_token_len)
            prompt_weights = prompt_weights + [1.0] * abs(prompt_token_len - neg_prompt_token_len)

        # padding the shorter one for token set 2
        prompt_token_len_2 = len(prompt_tokens_2)
        neg_prompt_token_len_2 = len(neg_prompt_tokens_2)

        if prompt_token_len_2 > neg_prompt_token_len_2:
            # padding the neg_prompt with eos token
            neg_prompt_tokens_2 = neg_prompt_tokens_2 + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
            neg_prompt_weights_2 = neg_prompt_weights_2 + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        else:
            # padding the prompt
            prompt_tokens_2 = prompt_tokens_2 + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
            prompt_weights_2 = prompt_weights + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)

        embeds = []
        neg_embeds = []

        prompt_token_groups, prompt_weight_groups = self.group_tokens_and_weights(prompt_tokens.copy(), prompt_weights.copy())

        neg_prompt_token_groups, neg_prompt_weight_groups = self.group_tokens_and_weights(
            neg_prompt_tokens.copy(), neg_prompt_weights.copy()
        )

        prompt_token_groups_2, prompt_weight_groups_2 = self.group_tokens_and_weights(
            prompt_tokens_2.copy(), prompt_weights_2.copy()
        )

        neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = self.group_tokens_and_weights(
            neg_prompt_tokens_2.copy(), neg_prompt_weights_2.copy()
        )

        # get prompt embeddings one by one is not working.
        for i in range(len(prompt_token_groups)):
            # get positive prompt embeddings with weights
            token_tensor = torch.tensor([prompt_token_groups[i]], dtype=torch.long, device=pipe.device)
            weight_tensor = torch.tensor(prompt_weight_groups[i], dtype=torch.float16, device=pipe.device)

            token_tensor_2 = torch.tensor([prompt_token_groups_2[i]], dtype=torch.long, device=pipe.device)

            # use first text encoder
            prompt_embeds_1 = pipe.text_encoder(token_tensor.to(pipe.device), output_hidden_states=True)
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]

            # use second text encoder
            prompt_embeds_2 = pipe.text_encoder_2(token_tensor_2.to(pipe.device), output_hidden_states=True)
            prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
            pooled_prompt_embeds = prompt_embeds_2[0]

            prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
            token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)

            for j in range(len(weight_tensor)):
                if weight_tensor[j] != 1.0:
                    token_embedding[j] = (
                        token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]
                    )

            token_embedding = token_embedding.unsqueeze(0)
            embeds.append(token_embedding)

            # get negative prompt embeddings with weights
            neg_token_tensor = torch.tensor([neg_prompt_token_groups[i]], dtype=torch.long, device=pipe.device)
            neg_token_tensor_2 = torch.tensor([neg_prompt_token_groups_2[i]], dtype=torch.long, device=pipe.device)
            neg_weight_tensor = torch.tensor(neg_prompt_weight_groups[i], dtype=torch.float16, device=pipe.device)

            # use first text encoder
            neg_prompt_embeds_1 = pipe.text_encoder(neg_token_tensor.to(pipe.device), output_hidden_states=True)
            neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]

            # use second text encoder
            neg_prompt_embeds_2 = pipe.text_encoder_2(neg_token_tensor_2.to(pipe.device), output_hidden_states=True)
            neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
            negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

            neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
            neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)

            for z in range(len(neg_weight_tensor)):
                if neg_weight_tensor[z] != 1.0:
                    neg_token_embedding[z] = (
                        neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor[z]
                    )

            neg_token_embedding = neg_token_embedding.unsqueeze(0)
            neg_embeds.append(neg_token_embedding)

        prompt_embeds = torch.cat(embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

        if extra_emb is not None:
            extra_emb = extra_emb.to(prompt_embeds.device, dtype=prompt_embeds.dtype) * extra_emb_alpha
            prompt_embeds = torch.cat([prompt_embeds, extra_emb], 1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, torch.zeros_like(extra_emb)], 1)
            print(f'fix prompt_embeds, extra_emb_alpha={extra_emb_alpha}')

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def get_prompt_embeds(self, *args, **kwargs):
        prompt_embeds, negative_prompt_embeds, _, _ = self.get_weighted_text_embeddings_sdxl(*args, **kwargs)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        return prompt_embeds

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

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil
    
class StableDiffusionXLInstantIDPipeline(StableDiffusionXLControlNetPipeline):
    
    def cuda(self, dtype=torch.float16, use_xformers=False):
        self.to('cuda', dtype)
        
        if hasattr(self, 'image_proj_model'):
            self.image_proj_model.to(self.unet.device).to(self.unet.dtype)
        
        if use_xformers:
            if is_xformers_available():
                import xformers
                from packaging import version

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    def load_ip_adapter_instantid(self, model_ckpt, image_emb_dim=512, num_tokens=16, scale=0.5):     
        self.set_image_proj_model(model_ckpt, image_emb_dim, num_tokens)
        self.set_ip_adapter(model_ckpt, num_tokens, scale)
        
    def set_image_proj_model(self, model_ckpt, image_emb_dim=512, num_tokens=16):
        
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.unet.config.cross_attention_dim,
            ff_mult=4,
        )

        image_proj_model.eval()
        
        self.image_proj_model = image_proj_model.to(self.device, dtype=self.dtype)
        state_dict = torch.load(model_ckpt, map_location="cpu")
        if 'image_proj' in state_dict:
            state_dict = state_dict["image_proj"]
        self.image_proj_model.load_state_dict(state_dict)
        
        self.image_proj_model_in_features = image_emb_dim
    
    def set_ip_adapter(self, model_ckpt, num_tokens, scale):
        
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                   cross_attention_dim=cross_attention_dim, 
                                                   scale=scale,
                                                   num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
        unet.set_attn_processor(attn_procs)
        
        state_dict = torch.load(model_ckpt, map_location="cpu")
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        if 'ip_adapter' in state_dict:
            state_dict = state_dict['ip_adapter']
        ip_layers.load_state_dict(state_dict)
    
    def set_ip_adapter_scale(self, scale):
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def _encode_prompt_image_emb(self, prompt_image_emb, device, num_images_per_prompt, dtype, do_classifier_free_guidance):
        
        if isinstance(prompt_image_emb, torch.Tensor):
            prompt_image_emb = prompt_image_emb.clone().detach()
        else:
            prompt_image_emb = torch.tensor(prompt_image_emb)
            
        prompt_image_emb = prompt_image_emb.reshape([1, -1, self.image_proj_model_in_features])
        
        if do_classifier_free_guidance:
            prompt_image_emb = torch.cat([torch.zeros_like(prompt_image_emb), prompt_image_emb], dim=0)
        else:
            prompt_image_emb = torch.cat([prompt_image_emb], dim=0)
        
        prompt_image_emb = prompt_image_emb.to(device=self.image_proj_model.latents.device, 
                                               dtype=self.image_proj_model.latents.dtype)
        prompt_image_emb = self.image_proj_model(prompt_image_emb)

        bs_embed, seq_len, _ = prompt_image_emb.shape
        prompt_image_emb = prompt_image_emb.repeat(1, num_images_per_prompt, 1)
        prompt_image_emb = prompt_image_emb.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        return prompt_image_emb.to(device=device, dtype=dtype)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        # IP adapter
        ip_adapter_scale=None,

        # Enhance Face Region
        control_mask = None,

        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                argument.
            image_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated image embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned containing the output images.
        """

        lpw = LongPromptWeight()

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
        
        # 0. set ip_adapter_scale
        if ip_adapter_scale is not None:
            self.set_ip_adapter_scale(ip_adapter_scale)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            prompt_2=prompt_2,
            image=image,
            callback_steps=callback_steps,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3.1 Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = lpw.get_weighted_text_embeddings_sdxl(
            pipe=self, 
            prompt=prompt, 
            neg_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        
        # 3.2 Encode image prompt
        prompt_image_emb = self._encode_prompt_image_emb(image_embeds, 
                                                         device,
                                                         num_images_per_prompt,
                                                         self.unet.dtype,
                                                         self.do_classifier_free_guidance)
        
        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 4.1 Region control
        if control_mask is not None:
            mask_weight_image = control_mask
            mask_weight_image = np.array(mask_weight_image)
            mask_weight_image_tensor = torch.from_numpy(mask_weight_image).to(device=device, dtype=prompt_embeds.dtype)
            mask_weight_image_tensor = mask_weight_image_tensor[:, :, 0] / 255.
            mask_weight_image_tensor = mask_weight_image_tensor[None, None]
            h, w = mask_weight_image_tensor.shape[-2:]
            control_mask_wight_image_list = []
            for scale in [8, 8, 8, 16, 16, 16, 32, 32, 32]:
                scale_mask_weight_image_tensor = F.interpolate(
                    mask_weight_image_tensor,(h // scale, w // scale), mode='bilinear')
                control_mask_wight_image_list.append(scale_mask_weight_image_tensor)
            region_mask = torch.from_numpy(np.array(control_mask)[:, :, 0]).to(self.unet.device, dtype=self.unet.dtype) / 255.
            region_control.prompt_image_conditioning = [dict(region_mask=region_mask)]
        else:
            control_mask_wight_image_list = None
            region_control.prompt_image_conditioning = [dict(region_mask=None)]

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        encoder_hidden_states = torch.cat([prompt_embeds, prompt_image_emb], dim=1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
                
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs
                
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                if isinstance(self.controlnet, MultiControlNetModel):
                    down_block_res_samples_list, mid_block_res_sample_list = [], []
                    for control_index in range(len(self.controlnet.nets)):
                        controlnet = self.controlnet.nets[control_index]
                        if control_index == 0:
                            # assume fhe first controlnet is IdentityNet
                            controlnet_prompt_embeds = prompt_image_emb
                        else:
                            controlnet_prompt_embeds = prompt_embeds
                        down_block_res_samples, mid_block_res_sample = controlnet(control_model_input,
                                                                                  t,
                                                                                  encoder_hidden_states=controlnet_prompt_embeds,
                                                                                  controlnet_cond=image[control_index],
                                                                                  conditioning_scale=cond_scale[control_index],
                                                                                  guess_mode=guess_mode,
                                                                                  added_cond_kwargs=controlnet_added_cond_kwargs,
                                                                                  return_dict=False)

                        # controlnet mask
                        if control_index == 0 and control_mask_wight_image_list is not None:
                            down_block_res_samples = [
                                down_block_res_sample * mask_weight
                                for down_block_res_sample, mask_weight in zip(down_block_res_samples, control_mask_wight_image_list)
                            ]
                            mid_block_res_sample *= control_mask_wight_image_list[-1]

                        down_block_res_samples_list.append(down_block_res_samples)
                        mid_block_res_sample_list.append(mid_block_res_sample)

                    mid_block_res_sample = torch.stack(mid_block_res_sample_list).sum(dim=0)
                    down_block_res_samples = [torch.stack(down_block_res_samples).sum(dim=0) for down_block_res_samples in
                                              zip(*down_block_res_samples_list)]
                else:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=prompt_image_emb,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                    )

                    # controlnet mask
                    if control_mask_wight_image_list is not None:
                        down_block_res_samples = [
                            down_block_res_sample * mask_weight
                            for down_block_res_sample, mask_weight in zip(down_block_res_samples, control_mask_wight_image_list)
                        ]
                        mid_block_res_sample *= control_mask_wight_image_list[-1]

                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
