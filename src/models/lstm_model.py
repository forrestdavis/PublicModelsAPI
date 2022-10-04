import torch
import string

from .RTModel import RTModel


class LSTMModel(RTModel):
    """General form of a neural-complexity LSTM RTModel.
       Heavily inspired by refining-tse (https://github.com/bnewm0609/refining-tse).
        Args:
            tokenizer_cls: transformers tokenizer class
            model_cls: transformers model class
            use_prefix_space: whether a prefix space should be included
            add_padding_token: whether to add a padding_token for batches

        Attributes:
            device: GPU or CPU device.
            tokenizer: neural-complexity tokenizer.
            model: neural-complexity model to make predictions.
            usePrefixSpace: whether the tokenizer needs a prefix space (e.g., GPT-2)
            model_name: name of model
    """
    def __init__(self, model_file, vocab_file, path_to_neural_complexity, 
            use_prefix_space=False, add_padding_token=True):
        super().__init__(model_name=model_file, use_prefix_space=use_prefix_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        import sys
        sys.path.append(path_to_neural_complexity)
        from Tokenizer import Tokenizer

        self._tokenizer = Tokenizer(vocab_file)
        if add_padding_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        with open(model_file, 'rb') as f:
            self.model = torch.load(f).to(self.device)

        self.model.eval()

    @property
    def tokenizer(self):
        return self._tokenizer

    def token_is_sep(self, token):
        return token == self.tokenizer.eos_token_id

    def word_to_idx(self, text, isFirstWord=False, isLastWord=False):
        #first we check left side 
        trim_right = False
        trim_left = False
        if "'" == text[-1] and not isLastWord:
            text = text + ' a'
            trim_right = True
        if ("'" == text[0] or '"' == text[0] or '(' == text[0]) and not isFirstWord:
            text = 'a ' + text
            trim_left = True
        encoding = self.tokenizer.encode(text)
        #filter out eos
        encoding = list(filter(lambda x: not self.token_is_sep(x), encoding))

        #Trim off trigger for '
        if trim_right:
            encoding = encoding[:-1]
        if trim_left:
            encoding = encoding[1:]

        return encoding 


    def word_in_vocab(self, text, isFirstWord=False, isLastWord=False):
        #first we check left side 
        trim_right = False
        trim_left = False
        if "'" == text[-1] and not isLastWord:
            text = text + ' a'
            trim_right = True
        if ("'" == text[0] or '"' == text[0] or '(' == text[0]) and not isFirstWord:
            text = 'a ' + text
            trim_left = True
        encoding = self.tokenizer.encode(text)
        #filter out eos
        encoding = list(filter(lambda x: not self.token_is_sep(x), encoding))

        #Trim off trigger for '
        if trim_right:
            encoding = encoding[:-1]
        if trim_left:
            encoding = encoding[1:]

        if len(encoding) > 1 or self.token_is_unk(encoding[0]):
            return False
        return True

    def token_is_unk(self, token):
        return self.tokenizer.unk_token_id == token

    def token_is_punct(self, token):
        return self.tokenizer.convert_ids_to_tokens(token)[0] in string.punctuation

    @torch.no_grad()
    def get_output(self, texts):
        #batchify whatever is coming in
        if type(texts) == str:
            texts = [texts]

        #need attention mask for locating padding
        inputs_dict = self.tokenizer.batch_encode_plus(texts, padding=True, 
                return_tensors="pt")

        inputs = inputs_dict["input_ids"].to(self.device)
        attn_mask = inputs_dict["attention_mask"].to(self.device)

        #Mark last position without padding
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

        #Transpose for RNN class (not batch first)
        inputs = inputs.transpose(1, 0)
        hidden = self.model.init_hidden(inputs.shape[1])
        logits = self.model(inputs, hidden)[0]

        #Flip it back to batch first
        logits = logits.transpose(1, 0)
        inputs = inputs.transpose(1, 0)

        return (inputs, last_non_masked_idx, logits)

    @torch.no_grad()
    def get_hidden_layers(self, texts):
        #batchify whatever is coming in 
        if type(texts) == str:
            texts = [texts]

        #need attention mask for locating padding
        inputs_dict = self.tokenizer.batch_encode_plus(texts, padding=True, 
                return_tensors="pt")

        inputs = inputs_dict["input_ids"].to(self.device)
        attn_mask = inputs_dict["attention_mask"].to(self.device)

        #Mark last position without padding
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

        #Transpose for RNN class (not batch first)
        #sequence length X batch size 
        inputs = inputs.transpose(1, 0)
        hidden = self.model.init_hidden(inputs.shape[1])
        #We have to loop because intermediate steps are not recorded by NC
        return_list = []
        for seq_idx in range(inputs.shape[0]):

            intermediate_inputs = inputs[seq_idx,:].unsqueeze(0)
            embeddings = self.model.encoder(intermediate_inputs)
            output, hidden = self.model(intermediate_inputs, hidden)
            h, c = hidden
            #flip to batch first
            #batch size X num layers X hidden size
            embeddings = embeddings.transpose(1,0)
            h = h.transpose(1,0)

            #first time, seed 
            if seq_idx == 0:
                return_list.append(embeddings)
                for layer_idx in range(self.model.nlayers):
                    #print(f"item {seq_idx} layer {layer_idx}", h[:, layer_idx,:])
                    return_list.append(h[:,layer_idx,:].unsqueeze(1))

            else:
                return_list[0] = torch.cat((return_list[0], embeddings), dim=1)
                for layer_idx in range(self.model.nlayers):
                    #print(f"item {seq_idx} layer {layer_idx}", h[:, layer_idx,:])
                    return_list[layer_idx+1] = torch.cat((return_list[layer_idx+1], h[:,layer_idx,:].unsqueeze(1)), dim=1)

        return (inputs.transpose(1, 0), last_non_masked_idx, tuple(return_list))

    @torch.no_grad()
    def get_targeted_output(self, texts):
        """See RTModel class for details.
            Returns logits from last non-padded index
        """
        #get outputs
        inputs, last_non_masked_idx, logits = self.get_output(texts)

        #Correctly reshape indices for use in gather 
        indices = last_non_masked_idx.unsqueeze(-1).repeat(1, logits.shape[-1]).unsqueeze(1)
        #flatten out the inner dimension so its (batch size X vocab size)
        final_logits = logits.gather(1, indices).squeeze(1)

        return final_logits
