import torch.nn.functional as F


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
