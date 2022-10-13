import torch
import transformers

from .transformers_model import TransformersModel

class ROBERTAModel(TransformersModel):
    def __init__(self, version):
        super().__init__(
            version,
            tokenizer_cls=transformers.RobertaTokenizer,
            model_cls=transformers.RobertaForMaskedLM,
            use_prefix_space=True,
            add_padding_token=False,
            bidirectional=True,
        )

    def token_is_sep(self, token):
        return (token == self.tokenizer.eos_token_id or 
                token == self.tokenizer.bos_token_id)

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

        inputs_dict = self.tokenizer.batch_encode_plus(texts, 
                        padding=True, add_special_tokens=True, 
                        return_tensors='pt').to(self.device)

        inputs = inputs_dict["input_ids"]
        attn_mask = inputs_dict["attention_mask"]

        #if the input is longer than the maximum allowed use sliding_window
        if inputs.shape[1] > MAX_LENGTH:
            self.get_slidding_window_output(texts)

        #For each batch, create version of input where each token (except, CLS and SEP) 
        #is MASK'd. This will generate conditional probabilities and follows, in spirit, 
        #Salazar et al. (2020) https://aclanthology.org/2020.acl-main.240.pdf
        #Kanishka Misra's minicons: https://github.com/kanishkamisra/minicons
        #Get each word masked inputs
        sent_input_dicts = self.tokens_to_masks(inputs_dict)
        #Get outputs
        logits = self.get_diagonaled_masked_output(sent_input_dicts)

        if return_attn_mask:
            return (inputs, attn_mask, self.model(**inputs_dict).logits)

        #Mark last position without padding
        #this works because transformers tokenizer flags 
        #padding with an attention mask value of 0
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

        return (inputs, last_non_masked_idx, logits) 

    @torch.no_grad()
    def get_targeted_output(self, texts):
        """See RTModel class for details.
            Returns logits from mask tokens, assumes
            one mask per element in batch.
        """
        #batchify whatever is coming in
        if type(texts) == str:
            texts = [texts]

        #Replace with correct mask
        TEXT_MASK = 'MASKTOKEN'
        texts = list(map(lambda x: x.replace(TEXT_MASK, self.tokenizer.mask_token), texts))

        MAX_LENGTH = self.tokenizer.model_max_length
        inputs_dict = self.tokenizer.batch_encode_plus(texts, 
                        padding=True, add_special_tokens=True, 
                        return_tensors='pt').to(self.device)

        inputs = inputs_dict["input_ids"]

        #Get the indices where token is the mask token
        mask_indices = torch.where(inputs==self.tokenizer.mask_token_id, 1, 0).nonzero(as_tuple=True)[1]

        #check that we have a mask for each input
        assert mask_indices.shape[0] == inputs.shape[0], "Likely some input does not have a mask token"

        logits = self.model(**inputs_dict).logits

        #Correctly reshape mask indices for use in gather 
        mask_indices = mask_indices.unsqueeze(-1).repeat(1, logits.shape[-1]).unsqueeze(1)
        #flatten out the inner dimension so its (batch size X vocab size)
        mask_logits = logits.gather(1, mask_indices).squeeze(1)

        return mask_logits
