import torch
import transformers

from .transformers_model import TransformersModel

class Byt5Model(TransformersModel):
    def __init__(self, version):
        super().__init__(
            version,
            tokenizer_cls=transformers.AutoTokenizer,
            model_cls = transformers.T5ForConditionalGeneration,
            use_prefix_space='check',
            add_padding_token=True,
            halfPrecision=True,
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
        raise NotImplementedError

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
        raise NotImplementedError

    @torch.no_grad()
    def get_by_sentence_perplexity(self, texts):
        """Returns perplexity of each sentence for inputted text.

            The following code draws on the implementation of t5 perplexity from
            SLING
            (https://github.com/Yixiao-Song/SLING_Data_Code/blob/824aaeaf5fc1065916f1b2316bb68a12cfc98c41/SLING_Code/utils.py#L113).
            That is, we treat t5 based models like bert based models 
            calculating psueo-likelihoods. 

            PPL for bidirectional models requires by word 
            surprisal values, which are obtained using 
            psuedo-likelihoods (Salazar et al., 2020 
                https://aclanthology.org/2020.acl-main.240/)

        Args: 
            text (List[str] | str ): A batch of strings or a string.

        Returns:
            lists (sent, ppl): List of the perplexity of each string in the
            batch. 
        """
        sents_ppls = [] 
        MASK_IDX = 258       

        for sentence in texts:
            token_surps = [] 
            for cnum, char in enumerate(sentence):
                prefix = sentence[:cnum]
                suffix = sentence[cnum + 1:]
                inputs = list(prefix.encode("utf-8")) + [MASK_IDX] + list(suffix.encode("utf-8")) + [MASK_IDX - 1, 1]

                if cnum == 0:
                    # ignore this case since it wasn't seen in training
                    continue
                    # labels = list(char.encode("utf-8")) + [MASK_IDX] + [1]
                elif cnum == len(sentence) - 1:
                    labels = [MASK_IDX] + list(char.encode("utf-8")) + [1]
                else:
                    labels = [MASK_IDX] + list(char.encode("utf-8")) + [MASK_IDX - 1] + [1]

                inputs = torch.LongTensor([inputs]).to(self.device)
                labels = torch.LongTensor([labels]).to(self.device)

                outs = self.model.forward(input_ids=inputs, labels=labels)
                surp = (outs.loss.item()/torch.log(torch.tensor(2.0))).item()
                token_surps.append(surp)

            avg_surp = sum(token_surps)/len(token_surps)
            ppl = torch.exp2(torch.tensor(avg_surp)).item()
            sents_ppls.append((sentence, ppl))

        return sents_ppls

