import torch
import transformers

from .auto_causal_model import AutoCausalModel

class LlamaModel(AutoCausalModel):
    def __init__(self, version, use_prefix_space = False, 
                halfPrecision=True):
        super().__init__(
            version, 
            use_prefix_space = use_prefix_space)
