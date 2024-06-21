import torch
import transformers

from .transformers_model import TransformersModel

class mT5Model(TransformersModel):
    def __init__(self, version):
        if 'google/mt5' in version:
            model_cls = transformers.MT5ForConditionalGeneration
        elif 'google/umt5' in version:
            model_cls = transformers.UMT5ForConditionalGeneration
        super().__init__(
            version,
            tokenizer_cls=transformers.AutoTokenizer,
            model_cls=model_cls,
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
        
        for sentence in texts:
            token_surps = [] 
            for cnum, char in enumerate(sentence):
                input_sent = list(sentence)
                input_sent[cnum] = ' <extra_id_0>'
                input_sent = ''.join(input_sent)

                if cnum == 0:
                    continue
                    # label_sent = f'{char} <extra_id_0>'
                elif cnum == len(sentence) - 1:
                    label_sent = f'<extra_id_0>{char}'
                else:
                    label_sent = f'<extra_id_0>{char} <extra_id_1>'

                inputs = self.tokenizer(input_sent,
                                        return_tensors='pt').to(self.device)
                labels = self.tokenizer(label_sent,
                                        return_tensors='pt').input_ids.to(self.device)
                outs = self.model.forward(**inputs, labels=labels)
                surp = (outs.loss.item()/torch.log(torch.tensor(2.0))).item()
                token_surps.append(surp)

            avg_surp = sum(token_surps)/len(token_surps)
            ppl = torch.exp2(torch.tensor(avg_surp)).item()
            sents_ppls.append((sentence, ppl))
        return sents_ppls

