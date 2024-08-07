import torch
import transformers

from .auto_causal_model import AutoCausalModel

class MistralModel(AutoCausalModel):
    def __init__(self, version, use_prefix_space = False):
        super().__init__(
            version, 
            use_prefix_space = use_prefix_space)
