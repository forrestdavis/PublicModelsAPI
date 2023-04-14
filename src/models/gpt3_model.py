import torch
import sys
import tiktoken
#from .RTModel import RTModel
from RTModel import RTModel

class GPT3Model(RTModel):
    def __init__(self, version, useMPS=True):

        super().__init__(model_name=version, 
                         use_prefix_space=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.has_mps and useMPS:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def token_is_sep(self, token):
        return (token == self.tokenizer.eos_token_id or 
                token == self.tokenizer.bos_token_id or 
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
                #add_prefix_space=False, 
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
        raise NotImplementedError

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

class GPT3Tokenizer:

    def __init__(self, version):
        self._enc_base = tiktoken.encoding_for_model(version)
        self._enc = tiktoken.Encoding(
                name=version+'_with_pad',
                pat_str=self._enc_base._pat_str,
                mergeable_ranks=self._enc_base._mergeable_ranks,
                special_tokens={
                    **self._enc_base._special_tokens,
                    " <|pad|>": 100264,
                }
            )
        self._vocab = self._enc._mergeable_ranks.copy()
        self._vocab.update(self._enc._special_tokens)

    @property
    def vocab(self):
        return self._vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def idx2word(self):
        return dict((v, k) for k, v in self.vocab.items())

    @property 
    def eos_token(self):
        return '<|endoftext|>'

    @property 
    def pad_token(self):
        """NOT USED BY TOKENIZER"""
        return " <|pad|>"

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def eos_token_id(self):
        if self.eos_token in self.vocab:
            return self.vocab[self.eos_token]
        else:
            return None

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, return_tensors=None):

        encodings = self.encode(text)
        encodings = self.batchify(encodings)

        if return_tensors=='pt':
            return {'input_ids': 
                         torch.tensor(encodings['input_ids'], 
                                      dtype=torch.int64), 
                         'attention_mask':
                         torch.tensor(encodings['attention_mask'], 
                                      dtype=torch.int64)}
        elif return_tensors is None:
            return encodings
        else:
            sys.stderr.write('I have not implemented a return_tensors type: '+str(return_tensors)+'\n')
            sys.exit(1)

    def encode(self, text, lower=False,
            remove_trailing_spaces=True):
        """ Returns a list of encoded text"""

        if type(text) == str:
            text = [text]

        if lower or remove_trailing_spaces:
            for idx, line in enumerate(text):
                if lower:
                    text[idx] = line.lower()
                if remove_trailing_spaces:
                    text[idx] = line.strip()

        return self._enc.encode_batch(text, allowed_special="all")

    def batchify(self, encodings):

        assert self.pad_token_id is not None, 'Attempting to PAD with no token'
        max_seq_len = max(len(encoding) for encoding in encodings)
        padded_batch_outputs = {'input_ids': [], 'attention_mask': []}

        for encoding in encodings:
            difference = max_seq_len - len(encoding)
            input_ids = encoding + [self.pad_token_id]*difference
            attn_ids = [1]*len(encoding) + [0]*difference

            padded_batch_outputs['input_ids'].append(input_ids)
            padded_batch_outputs['attention_mask'].append(attn_ids)

        return padded_batch_outputs

    def decode(self, input_dict):
        if type(input_dict) == dict:
            input_ids = input_dict['input_ids']
            attn_mask = input_dict['attention_mask']
        else:
            input_ids = input_dict

        if type(input_ids) != list:
            input_ids = input_ids.tolist()

        if type(input_ids[0]) != list:
            input_ids = [input_ids]

        decoded = []
        for encoding in input_ids:
            decoded.append(self._enc.decode_tokens_bytes(encoding))

        return decoded

    def convert_ids_to_tokens(self, ids):
        if type(ids) != list:
            ids = [ids]
        return self.decode(ids)

    def convert_tokens_to_ids(self, tokens):
        return self.encode(tokens)

        
if __name__ == '__main__':
    tokenizer = GPT3Tokenizer('gpt-3.5-turbo')

    encodings = tokenizer.encode(["hello world <|pad|>", "hell is a place"])
    encodings = tokenizer.batchify(encodings)
    print(encodings)
    print(tokenizer.decode(encodings))
    print(tokenizer.convert_ids_to_tokens(1000))
    print(tokenizer.convert_tokens_to_ids(['indow']))
    print()
    print(tokenizer(['hello world', 'goodbye today sleepy'], return_tensors=None))
    '''
    inputs = {'input_ids': [[1, 2, -100, -100], [1, 2, 3, 4]], 
              'attention_mask': torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
             }
    tokenizer.unbatchify(inputs)

    '''
