import torch
import string
import transformers
import json
import sys

from .RTModel import RTModel

class UnigramModel(RTModel):
    """General form of a unigram model (derived from some corpus counts) RTModel.
       Heavily inspired by refining-tse (https://github.com/bnewm0609/refining-tse).
        Args:
            model_file: Name of unigram file (should be vocab with counts in json)
            tokenizer_cls: class of transformers tokenizer class
            verison: version of tokenizer to use
            use_prefix_space: whether a prefix space should be included
            add_padding_token: whether to add a padding_token for batches

        Attributes:
            device: GPU or CPU device.
            tokenizer: neural-complexity tokenizer.
            model: neural-complexity model to make predictions.
            usePrefixSpace: whether the tokenizer needs a prefix space (e.g., GPT-2)
            model_name: name of model
    """
    def __init__(self, model_file, tokenizer_cls=transformers.GPT2Tokenizer, 
                 version='gpt2',
                 use_prefix_space=False, 
                 add_padding_token=True):

        super().__init__(model_name=model_file, use_prefix_space=use_prefix_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.create_model(model_file)

    @property
    def tokenizer(self):
        return self._tokenizer

    def token_is_sep(self, token):
        return False

    def token_is_unk(self, token):
        return self.tokenizer.unk_token_id == token

    def token_is_punct(self, token):
        return self.tokenizer.convert_ids_to_tokens([token])[0] in string.punctuation

    @torch.no_grad()
    def create_model(self, model_file):

        with open(model_file, 'r') as f:
            vocab2counts = json.load(f)

        probs = torch.zeros((len(self.tokenizer), 1))

        TOTAL = sum(vocab2counts.values())
        vocab = self.tokenizer.get_vocab()

        for word in vocab2counts:
            prob = vocab2counts[word]/TOTAL
            id = vocab[word]
            probs[id,0] = prob

        self.model = torch.nn.Embedding(len(self.tokenizer), 1)
        self.model.weight = torch.nn.Parameter(probs)

    @torch.no_grad()
    def get_sentence_likelihood(self, text):
        """Returns likelihood of each sentence in text.  
        For autoregressive models this is simply the joint probability
        of each word conditioned on the preceding context. For masked langauge
        models, we mask each token in the input to get its probability, then
        determine the joint probability across all tokens. No MASKTOKEN should
        be passed in for this use case. 

        Args: 
            text (List[str] | str ): A batch of strings or a string. Batches with 
                                    non equal length will be padded (though padding 
                                    will be ignored in return).

        Returns:
            List: predicted probabilites of each sentence in text
                                shape (batch_size, 1)
        """

        #batchify whatever is coming in
        if type(text) == str:
            text = [text]

        likelihoods = []
        for t in text:
            inputs = self.tokenizer.encode(t, return_tensors='pt')
            logprobs = torch.log(self.model(inputs).flatten())
            ll = sum(logprobs)
            ll = float(torch.exp(ll))
            likelihoods.append(ll)

        return likelihoods

