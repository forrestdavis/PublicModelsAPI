import torch
import transformers

from .transformers_model import TransformersModel

class TFXLModel(TransformersModel):
    def __init__(self, version):
        super().__init__(
            version,
            tokenizer_cls=transformers.TransfoXLTokenizer,
            model_cls=transformers.TransfoXLLMHeadModel,
            use_prefix_space=False,
            add_padding_token=True
        )

    def token_is_sep(self, token):
        return (token == self.tokenizer.eos_token_id or 
                token == self.tokenizer.bos_token_id)

    def word_to_idx(self, text, isFirstWord=False, isLastWord=False):
        #sometimes words are only split when there's a word 
        #after for TFXL, for example "bob'" is different 
        #than "bob' loves". Trying to account for that here.
        
        #first we check left side 
        trim_right = False
        trim_left = False
        if "'" == text[-1] and not isLastWord:
            text = text + ' a'
            trim_right = True
        if "'" == text[0] and not isFirstWord:
            text = 'a ' + text
            trim_left = True

        if self.usePrefixSpace:
            if isFirstWord:
                encoding = self.tokenizer.encode(text)
            else:
                encoding = self.tokenizer.encode(text, add_prefix_space=True, add_special_tokens=False)
        else:
            encoding = self.tokenizer.encode(text)

        #Trim off trigger for '
        if trim_right:
            encoding = encoding[:-1]
        if trim_left:
            encoding = encoding[1:]

        return encoding 

    @torch.no_grad()
    def get_output(self, texts, return_attn_mask=False):
        #batchify whatever is coming in
        if type(texts) == str:
            texts = [texts]

        #need attention mask for locating padding
        inputs_dict = self.tokenizer.batch_encode_plus(texts, padding=True, 
                return_tensors="pt", return_attention_mask=True).to(self.device)

        inputs = inputs_dict["input_ids"]
        attn_mask = inputs_dict["attention_mask"]

        if return_attn_mask:
            return (inputs, attn_mask, self.model(**inputs_dict).logits)

        #Mark last position without padding
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
        #return inputs and logits
        return (inputs, last_non_masked_idx, self.model(input_ids=inputs)[0])

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
