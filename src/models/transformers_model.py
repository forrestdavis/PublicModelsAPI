import torch
from .RTModel import RTModel
import string
import sys

class TransformersModel(RTModel):
    """General form of a Transformer RTModel.
       Heavily inspired by refining-tse (https://github.com/bnewm0609/refining-tse).
        Args:
            tokenizer_cls: transformers tokenizer class
            model_cls: transformers model class
            use_prefix_space: whether a prefix space should be included
            add_padding_token: whether to add a padding_token for batches

        Attributes:
            device: GPU or CPU device.
            tokenizer: Transformer tokenizer.
            model: Transformer model to make predictions.
            usePrefixSpace: whether the tokenizer needs a prefix space (e.g., GPT-2)
            model_name: name of model
    """
    def __init__(self, version, tokenizer_cls, model_cls, use_prefix_space=False, 
            add_padding_token=False, 
            bidirectional=False, 
            halfPrecision=False, 
            useMPS=False):
        super().__init__(model_name=version, use_prefix_space=use_prefix_space, bidirectional=bidirectional)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.has_mps and useMPS:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        sys.stderr.write(f"Running on {self.device}\n")

        #Downcast if using something like GPT-J
        if halfPrecision:
            self.model = model_cls.from_pretrained(version, torch_dtype=torch.float16, low_cpu_mem_usage=True, output_hidden_states=True).to(self.device)

        else:
            self.model = model_cls.from_pretrained(version, output_hidden_states=True).to(self.device)
        self.model.eval()

        try:
            self._tokenizer = tokenizer_cls.from_pretrained(version)
        except:
            sys.stderr.write(f"Loading tokenizer with {tokenizer_cls} failed...\n")
            sys.stderr.write(f"Trying loading tokenizer with AutoTokenizer class...\n")
            from transformers import AutoTokenizer
            tokenizer_cls = AutoTokenizer
            self._tokenizer = tokenizer_cls.from_pretrained(version)

        if add_padding_token:
            if not self._tokenizer.pad_token:
                if self._tokenizer.eos_token_id is not None:
                    self._tokenizer.pad_token = self.tokenizer.eos_token

                #This is largely to catch some issue with mrm8488/spanish-gpt2
                else:
                    assert '<pad>' in self._tokenizer.get_vocab(), 'Error getting pad token likely a vocab issue'

                    self._tokenizer.pad_token = '<pad>'

            assert self._tokenizer.pad_token is not None, 'Error pad token not set'
            sys.stderr.write('Pad token was set\n')


    @property
    def tokenizer(self):
        return self._tokenizer

    def token_is_sep(self, token):
        """Returns whether token is sep token (e.g., <eos>, [CLS])
        """
        raise NotImplementedError

    def word_in_vocab(self, text, isFirstWord=False, isLastWord=False):
        if self.usePrefixSpace:
            if isFirstWord:
                indices = self.tokenizer.encode(text)
            else:
                indices = self.tokenizer.encode(text, add_prefix_space=True, add_special_tokens=False)
        else:
            indices = self.tokenizer.encode(text)

        #Filter out eos/bos/cls/sep
        indices = list(filter(lambda x: not self.token_is_sep(x), indices))
        if len(indices)>1 or self.token_is_unk(indices[0]):
            return False
        return True

    def word_to_idx(self, text, isFirstWord=False, isLastWord=False):
        if self.usePrefixSpace:
            if isFirstWord:
                indices = self.tokenizer.encode(text)
            else:
                indices = self.tokenizer.encode(text, add_prefix_space=True, add_special_tokens=False)
        else:
            indices = self.tokenizer.encode(text)

        #Filter out eos/bos/cls/sep
        return list(filter(lambda x: not self.token_is_sep(x), indices))

    def token_is_unk(self, token):
        return self.tokenizer.unk_token_id == token

    def token_is_punct(self, token):
        return self.tokenizer.convert_ids_to_tokens([token])[0] in string.punctuation

    @torch.no_grad()
    def get_output(self, tokenizer):
        pass

    def tokens_to_masks(self, inputs_dict):
        """Returns a list of expaned model inputs where 
        each word is masked out for use with masked language
        models.

        For example, given the output of a huggingface 
        tokenizer for the input 'the man is tall.'
        which for bert-based-uncased has input_ids
        [101, 1996, 2158, 2003, 4206, 102]
        this function will return a new batch where 
        each token is replaced with the mask token (103 for
        bert-base-uncased). In other words, 
        [[103, 1996, 2158, 2003, 4206, 102], 
         [101, 103, 2158, 2003, 4206, 102],
         [101, 1996, 103, 2003, 4206, 102],
         [101, 1996, 2158, 103, 4206, 102],
         [101, 1996, 2158, 2003, 103, 102],
         [101, 1996, 2158, 2003, 4206, 103]]
        The attention mask is broadcast accordingly. Masking  
        out the CLS and SEP tokens is not informative, but 
        it simplifies things on my end. 

        Args:
            inputs_dict dict: Output of huggingface tokenizer, 
                        a dict with input_ids and attention_mask

        Returns List[dict]: a list with an element for each batch in 
                        the inputs_dict, where each word is masked out 
                        for masked language models.
        """

        return_input_dicts = []

        input_ids = inputs_dict['input_ids']
        for sent_idx in range(input_ids.shape[0]):
            input_id = input_ids[sent_idx]

            #Create a matrix which has a row where each token 
            #is replaced with the mask token
            mask_token_diagonal = input_id.repeat(input_id.shape[0], 1)
            mask_token_diagonal.fill_diagonal_(self.tokenizer.mask_token_id)

            #Copy the attention mask for output
            attn_mask = inputs_dict['attention_mask'][sent_idx].repeat(input_id.shape[0], 1)

            return_dict = {'input_ids': mask_token_diagonal, 'attention_mask': attn_mask}
            return_input_dicts.append(return_dict)

        return return_input_dicts

    @torch.no_grad()
    def get_diagonaled_masked_output(self, input_dicts):
        """Returns the output logit values along the diagonal 
        of the inputted data. The use case is for 
        the output of tokens_to_masks for masked language model 
        pseudo-log likelihood approximation following 
        Salazar et al. (2020) https://aclanthology.org/2020.acl-main.240.pdf
        and Kanishka Misra's minicons: https://github.com/kanishkamisra/minicons

        For example, given the output of a huggingface 
        tokenizer for the input 'the man is tall.'
        which for bert-based-uncased has input_ids
        [101, 1996, 2158, 2003, 4206, 102], tokens_to_masks
        will return a new batch where 
        each token is replaced with the mask token (103 for
        bert-base-uncased). In other words, 
        [[103, 1996, 2158, 2003, 4206, 102], 
         [101, 103, 2158, 2003, 4206, 102],
         [101, 1996, 103, 2003, 4206, 102],
         [101, 1996, 2158, 103, 4206, 102],
         [101, 1996, 2158, 2003, 103, 102],
         [101, 1996, 2158, 2003, 4206, 103]]
        Masking out the CLS and SEP tokens is not informative, but 
        it simplifies things on my end. This function returns the logits 
        at the diagonal position. In other words, the predictions 
        at each of the masked tokens. The output mirrors the implicit batch 
        size, so that this returns the masked logits for each input token.

        Args:
            inputs_dicts List[dict]: a list with an element for each batch in 
                        the inputs_dict, where each word is masked out 
                        for masked language models.

        Returns Tensor: A tensor which contains the diagonal values 
                        of the model output logits. The shape 
                        is (batch size from len(input_dicts) X seq len X output dim)
        """


        outputs = None
        for input_dict in input_dicts:
            output = self.model(**input_dict).logits
            output = torch.diagonal(output, dim1=0, dim2=1).transpose(0,1)
            output = output.unsqueeze(0)
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
