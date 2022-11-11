import torch
import transformers

from .transformers_model import TransformersModel

class XGLMModel(TransformersModel):
    def __init__(self, version):
        super().__init__(
            version,
            tokenizer_cls=transformers.XGLMTokenizer,
            model_cls=transformers.XGLMForCausalLM,
            use_prefix_space=False,
            add_padding_token=True
        )

    def token_is_sep(self, token):
        return (token == self.tokenizer.eos_token_id or 
                token == self.tokenizer.bos_token_id or 
                token == self.tokenizer.cls_token_id or
                token == self.tokenizer.sep_token_id or
                token == self.tokenizer.pad_token_id)

    def get_slidding_window_output(self, texts):
        raise NotImplementedError

    @torch.no_grad()
    def get_output(self, texts, return_attn_mask=False):
        """See RTModel class for details.
        """
        #batchify whatever is coming in
        if type(texts) == str:
            texts = [texts]

        MAX_LENGTH = self.tokenizer.model_max_length

        #don't need prefix space in the full sentence case, only when 
        #later aligning (or checking for individual things)
        inputs_dict = self.tokenizer.batch_encode_plus(texts, padding=True, 
                return_tensors="pt").to(self.device)

        inputs = inputs_dict["input_ids"]
        attn_mask = inputs_dict["attention_mask"]

        #if the input is longer than the maximum allowed use sliding_window
        if inputs.shape[1] > MAX_LENGTH:
            self.get_slidding_window_output(texts)

        if return_attn_mask:
            return (inputs, attn_mask, self.model(**inputs_dict).logits)

        #Mark last position without padding
        #this works because transformers tokenizer flags 
        #padding with an attention mask value of 0
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
        return (inputs, last_non_masked_idx, self.model(**inputs_dict).logits)

    @torch.no_grad()
    def get_hidden_layers(self, texts):
        """See RTModel class for details.
        """
        #batchify whatever is coming in 
        if type(texts) == str:
            texts = [texts]

        MAX_LENGTH = self.tokenizer.model_max_length

        #don't need prefix space in the full sentence case, only when 
        #later aligning (or checking for individual things)
        inputs_dict = self.tokenizer.batch_encode_plus(texts, padding=True, 
                return_tensors="pt").to(self.device)

        inputs = inputs_dict["input_ids"]
        attn_mask = inputs_dict["attention_mask"]

        #if the input is longer than the maximum allowed use sliding_window
        if inputs.shape[1] > MAX_LENGTH:
            self.get_slidding_window_output(texts)

        #Mark last position without padding
        #this works because transformers tokenizer flags 
        #padding with an attention mask value of 0
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

        return (inputs, last_non_masked_idx, self.model(**inputs_dict).hidden_states)

    @torch.no_grad()
    def get_targeted_output(self, texts):
        """See RTModel class for details.
            Returns logits from last non-padded index
        """
        #get outputs
        _, last_non_masked_idx, logits = self.get_output(texts)

        #Correctly reshape indices for use in gather 
        indices = last_non_masked_idx.unsqueeze(-1).repeat(1, logits.shape[-1]).unsqueeze(1)
        #flatten out the inner dimension so its (batch size X vocab size)
        final_logits = logits.gather(1, indices).squeeze(1)

        return final_logits
