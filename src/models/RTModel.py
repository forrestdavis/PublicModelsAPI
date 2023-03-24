#Basic model class inspired by 
#https://github.com/bnewm0609/refining-tse (which is great!)
import torch
from collections import namedtuple

"""Defines basic model for evaluting on corpora"""
class RTModel(object):
    def __init__(self, model_name = '', use_prefix_space=False, 
            bidirectional=False):

        """Initialize model.
        """
        self.model_name = model_name
        self.usePrefixSpace=use_prefix_space
        self.bidirectional=bidirectional

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return f"{self.model_name} on Device {self.device}"

    @property
    def tokenizer(self):
        """Maps words to model indices 
        """
        raise NotImplementedError

    def word_in_vocab(self, text, isFirstWord, isLastWord):
        """Returns whether a single token is in the vocab, 
        not subworded, non unk token.
        """
        raise NotImplementedError

    def word_to_idx(self, text, isFirstWord, isLastWord):
        """Maps a single word to its indices.
        """
        raise NotImplementedError

    def token_is_unk(self, token):
        """Returns whether single token is UNK.
        """
        raise NotImplementedError

    def token_is_punct(self, token):
        """Returns whether single token is punctuation.
        """
        raise NotImplementedError

    def token_is_sep(self, token):
        """Returns whether single token is eos, bos, cls, or sep tokens
        """
        raise NotImplementedError

    def tokens_to_ids(self, strings):
        """Returns ids from list of strings.
            
            NB: Treats all words as non-initial/final.
                Assert catches text that is split into 
                multiple tokens

            Returns: List[int] : List of model token ids 
        """
        string_ids = []
        for string in strings:
            assert self.word_in_vocab(string), f"{string} not in vocab"
            string_ids.extend(self.word_to_idx(string))

        return string_ids

    @torch.no_grad()
    def get_aligned_words_probabilities(self, texts, 
            include_punctuation=False):
        """Returns probabilities of each word for inputted text.
           Note that this requires that you've implemented
           a tokenizer, get_output, token_is_unk, token_is_punct, 
           and word_to_idx
           for the model instance.
           Note assumes that words are space seperated in text.
           Basically we assume the following things: 
           1) get_output works as specified 
           2) token_is_unk returns whether a token is unk
           3) token_is_punct returns whether a token is punctuation
           4) word_to_idx will tokenize a word as it would have in the sentence

           The most care should be taken with 4)

           Futher note, eos tokens are thrown out. Finally, calculations 
           are done in log space for numerical stability (i.e. 
           calls get_aligned_words_surprisals and maps result to prob space)

        Args: 
            texts (List[str] | str ): A batch of strings or a string.
            include_punctuation: Whether to include punctuation in surprisal


        Returns:
            list of lists Word: Word is a namedtuple containing
                                word: word in text
                                prob: probability of word
                                isSplit: whether the word was split by tokenizer
                                isUnk: whether the word is an unk
                                withPunct: whether the surprisal value includes punctuation
                                modelName: name of model
                            batch_size X len(split text)
        """
        #Tuple structure to hold word and other info
        ProbWord = namedtuple('SurpWord', 'word prob isSplit isUnk withPunct modelName')
        SurpWords = self.get_aligned_words_surprisals(texts, include_punctuation)

        ProbWords = []
        for sent in SurpWords:
            prob_words = []
            for surp_word in sent:
                w = ProbWord(surp_word.word, 2**(-surp_word.surp), surp_word.isSplit, 
                        surp_word.isUnk, surp_word.withPunct, surp_word.modelName)
                prob_words.append(w)
            ProbWords.append(prob_words)

        return ProbWords

    @torch.no_grad()
    def get_aligned_words_surprisals(self, texts, 
            include_punctuation=False, 
            language='en'):
        """Returns surprisal of each word for inputted text.
           Note that this requires that you've implemented
           a tokenizer, get_output, token_is_unk, token_is_punct, 
           and word_to_idx
           for the model instance.
           Note assumes that words are space seperated in text.
           Basically we assume the following things: 
           1) get_output works as specified 
           2) token_is_unk returns whether a token is unk
           3) token_is_punct returns whether a token is punctuation
           4) word_to_idx will tokenize a word as it would have in the sentence

           The most care should be taken with 4)

           Futher note, eos tokens are thrown out

        Args: 
            texts (List[str] | str ): A batch of strings or a string.
            include_punctuation: Whether to include punctuation in surprisal


        Returns:
            list of lists Word: Word is a namedtuple containing
                                word: word in text
                                surp: surprisal of word
                                isSplit: whether the word was split by tokenizer
                                isUnk: whether the word is an unk
                                withPunct: whether the surprisal value includes punctuation
                                modelName: name of model
                            batch_size X len(split text)
        """
        #Tuple structure to hold word and other info
        SurpWord = namedtuple('SurpWord', 'word surp isSplit isUnk withPunct modelName')
        #get by token surprisals
        token_surprisals = self.get_by_token_surprisals(texts)

        #Filter out eos/bos/sep/cls stuff
        filtered_token_surprisals = []
        for element in token_surprisals:
            filtered_token_surprisals.append(list(filter(lambda x: not self.token_is_sep(x[0]), 
                element)))
        token_surprisals = filtered_token_surprisals

        #batchify if necessary 
        if type(texts) == str:
            texts = [texts]

        return_data = []
        for text_idx, text in enumerate(texts):
            #add a holder for this text
            return_data.append([])
            if language in {'en', 'es', 'it'}:
                chunks = text.split(' ')
            elif language == 'zh':
                chunks = text
            else:
                import sys
                sys.stderr.write(f"The language code {lang} is not specified " \
                                 "for chunking")
                sys.exit(1)

            for word_pos, word in enumerate(chunks):
                surp = 0
                isUnk = 0
                isSplit = 0
                withPunct = 0

                #this is critical for gpt2 (or any that use prefix spacing)
                isFirstWord=False
                if word_pos == 0:
                    isFirstWord=True

                isLastWord = False
                if word_pos == len(chunks)-1:
                    isLastWord = True

                #Tokenize the word
                word_tokens = self.word_to_idx(word, isFirstWord, isLastWord)
                if len(word_tokens) > 1:
                    isSplit = 1

                for word_token in word_tokens:
                    token_surprisal = token_surprisals[text_idx].pop(0)
                    #here we ensure that things are working by 
                    #checking that the tokenizations match
                    assert word_token == token_surprisal[0]

                    #check unk and punct
                    if self.token_is_unk(word_token):
                        isUnk = 1

                    #NB: This skips over internal punctuation 
                    if self.token_is_punct(word_token):
                        if not include_punctuation:
                            continue
                        else:
                            withPunct = 1

                    #treating surprisal of subwords as joint probability
                    surp += token_surprisal[1]

                w = SurpWord(word, surp, isSplit, isUnk, withPunct, self.model_name)
                return_data[text_idx].append(w)

            #Make sure we got it all
            assert len(token_surprisals[text_idx]) == 0

        return return_data
    
    @torch.no_grad()
    def get_output(self, text):
        """Returns input token ids, last nonmasked idx, 
            and logits for inputted text.
            last nonmasked idx is as simple as the length of tokenized text if
            there is no padding needed for the batch.
        Args: 
            text (List[str] | str ): A batch of strings or a string. Batches with 
                                    non equal length will be padded.

        Returns:
            (torch.Tensor, torch.Tensor): (input token ids, last nonmasked idx, 
            predicted logits with shape 
            (batch_size, max len(tokenized text), vocab_size))
        """
        #Some notes on last nonmasked idx
        #meant to record positions that should be ignored
        #so if we have some input like 
        #[ [ 1 2 3]
        #  [ 1 2 ]]
        # to run a batch the model will pad the second row, and we 
        # will subsequently record that the last_non_masked_idx is 
        # [2, 1]
        raise NotImplementedError

    @torch.no_grad()
    def get_targeted_word_surprisals(self, context, target, 
            aggregate_funct = sum):
        """Returns the surprisals for target words given the context
        sentences. For more than one target per context, an 
        aggregate function is applied so as to ensure only 
        one value is returned per context. 
        For autoregressive models the context is treated 
        as the prefix to the target word. For masked language models
        the context should include the necessary masked token, with 
        the target replacing the masked location. MASKTOKEN 
        is the assumed mask in the text, which will be 
        replaced by the specific models mask token. Intended use
        is for targeted syntactic evaluations.

        NB: 
            Checks that there is a target for each context and 
            that the target is a single word for the model 
            in question. Non-single word tokens 
            are UNK's by tokens_to_ids so this 
            is caught and filtered. 
            This could easily be relaxed for 
            the autoregressive models but remains tricky 
            for masked models so it is ignored for now.

        Args: 
            context (List[str] | str): A batch of strings or a string. Batches with 
                                    non equal length will be padded (though padding 
                                    will be ignored in return).
            target (List[str] | str): A batch of targets, one per context, 
                                     where the targets are either stings or 
                                     list of strings (which are aggregated 
                                     over), to calculate 
                                     the probability of given the context. 
        Returns:
            List[float]: list of target surprisals, one per context
        """
        #batchify things 
        if type(context) == str:
            context = [context]
        if type(target) == str:
            target = [target]

        assert len(target) == len(context), "Mismatch number of targets and contexts"

        return_surps = []

        #assumes target is not the first word
        targeted_logits = self.get_targeted_surprisals(context)
        target_ids = []
        for batch_idx, t in enumerate(target):

            if type(t) == str:
                t = [t]

            t_ids = self.tokens_to_ids(t)

            #filter out UNKs
            t_ids = list(filter(lambda x: not self.token_is_unk(x), t_ids))
            assert len(t_ids) != 0

            return_surps.append(aggregate_funct(targeted_logits[batch_idx,t_ids].tolist()))

        assert len(return_surps) == len(context)

        return return_surps

    @torch.no_grad()
    def get_targeted_word_probabilities(self, context, target, 
            aggregate_funct = sum):
        """Returns the probability for target words given the context
        sentences. For more than one target per context, an 
        aggregate function is applied so as to ensure only 
        one value is returned per context. 
        For autoregressive models the context is treated 
        as the prefix to the target word. For masked language models
        the context should include the necessary masked token, with 
        the target replacing the masked location. MASKTOKEN 
        is the assumed mask in the text, which will be 
        replaced by the specific models mask token. Intended use
        is for targeted syntactic evaluations.

        NB: 
            Checks that there is a target for each context and 
            that the target is a single word for the model 
            in question. Non-single word tokens 
            are UNK's by tokens_to_ids so this 
            is caught and filtered. 
            This could easily be relaxed for 
            the autoregressive models but remains tricky 
            for masked models so it is ignored for now.

        Args: 
            context (List[str] | str): A batch of strings or a string. Batches with 
                                    non equal length will be padded (though padding 
                                    will be ignored in return).
            target (List[str] | str): A batch of targets, one per context, 
                                     where the targets are either stings or 
                                     list of strings (which are aggregated 
                                     over), to calculate 
                                     the probability of given the context. 
        Returns:
            List[float]: list of target probabilities, one per context
        """
        #batchify things 
        if type(context) == str:
            context = [context]
        if type(target) == str:
            target = [target]

        assert len(target) == len(context), "Mismatch number of targets and contexts"

        return_probs = []

        #assumes target is not the first word
        targeted_logits = self.get_targeted_probabilities(context)
        target_ids = []
        for batch_idx, t in enumerate(target):

            if type(t) == str:
                t = [t]

            t_ids = self.tokens_to_ids(t)

            #filter out UNKs
            t_ids = list(filter(lambda x: not self.token_is_unk(x), t_ids))
            assert len(t_ids) != 0

            return_probs.append(aggregate_funct(targeted_logits[batch_idx,t_ids].tolist()))

        assert len(return_probs) == len(context)

        return return_probs

    def get_targeted_output(self, text):
        """Returns targeted logits for use in targeted syntactic 
        analysis. For autoregressive models this is the final non-padded
        logit and for masked language models this is the logit 
        corresponding to the masked locations. MASKTOKEN 
        is the assumed mask in the text, which will be 
        replaced by the specific models mask token.
        Args: 
            text (List[str] | str ): A batch of strings or a string. Batches with 
                                    non equal length will be padded (though padding 
                                    will be ignored in return).

        Returns:
            torch.Tensor: predicted logits with shape (batch_size, vocab_size)
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_targeted_probabilities(self, text):
        """Returns targeted probabilities for use in targeted syntactic 
        analysis. For autoregressive models this is the final non-padded
        probabilities and for masked language models this is the probabilities
        corresponding to the masked locations. MASKTOKEN 
        is the assumed mask in the text, which will be 
        replaced by the specific models mask token.

        Args: 
            text (List[str] | str ): A batch of strings or a string. Batches with 
                                    non equal length will be padded (though padding 
                                    will be ignored in return).

        Returns:
            torch.Tensor: predicted probabilities with shape (batch_size, vocab_size)
        """
        logits = self.get_targeted_output(text)
        return self.convert_to_probabilities(logits)

    @torch.no_grad()
    def get_targeted_surprisals(self, text):
        """Returns targeted surprisals for use in targeted syntactic 
        analysis. For autoregressive models this is the final non-padded
        surprisal and for masked language models this is the surprisals
        corresponding to the masked locations. MASKTOKEN 
        is the assumed mask in the text, which will be 
        replaced by the specific models mask token.

        Args: 
            text (List[str] | str ): A batch of strings or a string. Batches with 
                                    non equal length will be padded (though padding 
                                    will be ignored in return).

        Returns:
            torch.Tensor: predicted surprisals with shape (batch_size, vocab_size)
        """
        logits = self.get_targeted_output(text)
        return self.convert_to_surprisal(logits)

    @torch.no_grad()
    def get_sentence_likelihood(self, text, startPOS=0, log=False):
        """Returns likelihood of each sentence in text.  
        For autoregressive models this is simply the joint probability
        of each word conditioned on the preceding context. For masked language
        models, we mask each token in the input to get its probability, then
        determine the joint probability across all tokens. No MASKTOKEN should
        be passed in for this use case. 

        Args: 
            text (List[str] | str ): A batch of strings or a string. Batches with 
                                    non equal length will be padded (though padding 
                                    will be ignored in return).
            startPOS (int | List[int]): The position to start summing from. Default
                            is 0 (the true beginning). If all sentences in
                            batch, should start from the same location use one
                            int. If you want varied starts, use a list.
            log (bool): Whether to return log likelihood (sum of log probs of
                            each token) default: false.

        Returns:
            List: predicted probabilites of each sentence in text
                                shape (batch_size, 1)
        """
        token_surps = self.get_by_token_surprisals(text)

        likelihoods = []

        if type(token_surps[0]) == tuple:
            ll = 0
            assert type(startPOS) == int
            token_surps = token_surps[startPOS:]
            for _, surp in token_surps:
                ll += surp
            if log:
                likelihoods.append(-ll)
            else:
                likelihoods.append(2**(-ll))
        else:
            assert type(startPOS) == int or len(startPOS) == len(token_surps)
            for idx, token_surp in enumerate(token_surps):

                if type(startPOS) == int:
                    sPOS = startPOS
                else:
                    sPOS = startPOS[idx]

                token_surp = token_surp[sPOS:]
                ll = 0
                for _, surp in token_surp:
                    ll += surp
                if log:
                    likelihoods.append(-ll)
                else:
                    likelihoods.append(2**(-ll))

        return likelihoods

    @torch.no_grad()
    def convert_to_surprisal(self, logits):
        """Returns surprisals from logits

        Args:
            logits torch.Tensor: logits with shape (batch_size,
            number of tokens, vocab_size),
            as in output of get_output()

        Returns:
            torch.Tensor: surprisals with shape (batch_size, number of tokens, vocab_size)
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        surprisals = -(log_probs/torch.log(torch.tensor(2.0)))
        return surprisals

    @torch.no_grad()
    def convert_to_probabilities(self, logits):
        """Returns probabilities from logits

        Args: 
            logits torch.Tensor: logits with shape (batch_size, number of tokens, vocab_size), as in output of get_output()

        Returns:
            torch.Tensor: probabilities with shape (batch_size, number of tokens, vocab_size)
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probs)
        return probabilities

    @torch.no_grad()
    def get_surprisals(self, text):
        """Returns (input_ids, last non masked idx, 
            surprisals by token for inputted text)
           Note that this requires that you've implemented
           get_output for the model instance.

        Args: 
            text (List[str] | str ): A batch of strings or a string.

        Returns:
            tuple(torch.Tensor, torch.Tensor): (input_ids, last non masked idx, 
            surprisals with shape (batch_size, max len(tokenized text), vocab_size))
            Meaning it includes padding
        """
        inputs, last_non_masked_idx, logits = self.get_output(text)
        return (inputs, last_non_masked_idx, self.convert_to_surprisal(logits))

    @torch.no_grad()
    def get_by_sentence_perplexity(self, text):
        """Returns perplexity of each sentence for inputted text.
           Note that this requires that you've implemented
           a tokenizer and get_output
           for the model instance.

           PPL for autoregressive models is defined as: 
            .. math::
                2^{-\\frac{1}{N} \\sum_{i}^{N} log p_{\\theta}(x_i|x_{<i})

            PPL for bidirectional models requires by word 
            surprisal values, which are obtained using 
            psuedo-likelihoods (Salazar et al., 2020 
                https://aclanthology.org/2020.acl-main.240/)

        Args: 
            text (List[str] | str ): A batch of strings or a string.

        Returns:
            lists (sent, ppl): List of the perplexity of each string in the
            batch. Padding is ignored in the calculation. 
        """
        inputs, last_non_masked_idx, surprisals = self.get_surprisals(text)

        # Now gather the target words (i.e. what words we should be checking)
        if not self.bidirectional:
            #we are predicting the next word, so the target 
            #for each word, target is in fact the next word
            target_ids = inputs[:, 1:]
            #We don't look at the predictions of the final word, as we 
            #don't have any need to see generation
            target_surprisals = surprisals[:,:-1, :]

            #We use gather to get the logprob for each target item in the batch
            target_surprisals = target_surprisals.gather(-1, target_ids.unsqueeze(2)).squeeze(1)
            # Prepend a zero vector for the first token which is assigned 
            # no logprob by autoregressive model (e.g., <bos>)
            target_surprisals = torch.cat((torch.zeros((target_surprisals.shape[0], 1,
                         target_surprisals.shape[-1])), target_surprisals), 
                          dim = 1)

        #This is non-autoregressive, so predictions are not offset
        else:
            target_ids = inputs.clone()
            target_surprisals = surprisals

            #We use gather to get the logprob for each target item in the batch
            target_surprisals = target_surprisals.gather(-1, target_ids.unsqueeze(2)).squeeze(1)

        # Adapted from the discussion here: 
        # https://stackoverflow.com/questions/57548180/
        #       filling-torch-tensor-with-zeros-after-certain-index
        # Set padded words in sequence to zero
        mask = torch.zeros(target_surprisals.shape[0],
                           target_surprisals.shape[1]+1, device=self.device)
        mask[(torch.arange(target_surprisals.shape[0]), last_non_masked_idx+1)] = 1
        mask = mask.cumsum(dim=1)[:,:-1]
        mask = 1. - mask
        mask = mask[...,None]
        target_surprisals = target_surprisals*mask

        # Remove redundant final dimension (which originally tracked 
        #           vocab size)
        target_surprisals = target_surprisals.squeeze(-1)

        # Actual length is 1 + last position
        # However with unidirectional model the first 
        # position will be meaningless, so we will ignore it
        # For example, 
        # [the, cat, eats] -> [0, X, Y] 
        # last position will say 2, which is the actual length 
        # of the values 

        # For bidirectional models we set the SEP and CLS 
        # tokens to 0 and set the length to the 
        # length of the string without these tokens 
        if self.bidirectional: 
            target_surprisals[(torch.arange(target_surprisals.shape[0]), 
                               last_non_masked_idx)] = 0
            target_surprisals[:,0] = 0

            last_non_masked_idx -= 1

        # Now get rowwise sum (i.e. the log prob of each batch) and
        # average
        log_avgs = torch.sum(target_surprisals, dim=1)/last_non_masked_idx

        ppl = torch.exp2(log_avgs)

        return list(zip(text, ppl.tolist()))

    @torch.no_grad()
    def get_by_token_surprisals(self, text):
        """Returns surprisal of each token for inputted text.
           Note that this requires that you've implemented
           a tokenizer and get_output
           for the model instance.

        Args: 
            text (List[str] | str ): A batch of strings or a string.

        Returns:
            lists (token, surp): Lists of (token id, surprisal) that are 
            batch_size X len(tokenized text). 
            Meaning that the padding from get_surprisals is removed.
        """
        input_ids, last_non_masked_idx, surprisals = self.get_surprisals(text)

        #This is autoregressive, so predictions are offset
        if not self.bidirectional:
            #we are predicting the next word, so the target 
            #for each word is in fact the next word
            target_ids = input_ids[:, 1:]
            #We don't look at the predictions of the final word, as we 
            #don't have any need to see generation
            target_surprisals = surprisals[:,:-1, :]
            #We use gather to get the surprisals for each target item in the batch
            target_surprisals = target_surprisals.gather(-1, target_ids.unsqueeze(2)).squeeze(1)

        #This is non-autoregressive, so predictions are not offset
        else:
            target_ids = input_ids.clone()
            target_surprisals = surprisals
            #We use gather to get the surprisals for each target item in the batch
            target_surprisals = target_surprisals.gather(-1, target_ids.unsqueeze(2)).squeeze(1)

        return_data = []
        for i in range(target_surprisals.shape[0]):
            #if unidirectional, first token is not 
            #included in target_surprisals
            #seed with the first input id per element in batch
            #it will have probability of zero
            if not self.bidirectional:
                return_data.append([(int(input_ids[i, 0].data), 0)])
            #else seed blank list
            else:
                return_data.append([])
            #couple steps here, 1) we zip together the token ids and their
            #surprisals 2) we map this to ints and floats (out of tensors)
            #3) dump it into a list
            batch_data = list(map(lambda x: (int(x[0].data), 
                float(x[1].data)), zip(target_ids[i], target_surprisals[i])))
            #Chop off the stuff which is padding
            batch_data = batch_data[:last_non_masked_idx[i]]
            return_data[i].extend(batch_data)

        return return_data

    @torch.no_grad()
    def get_hidden_layers(self, text):
        """Returns input token ids, last nonmasked idx, 
            and hidden representations for inputted text.
            last nonmasked idx is as simply as the length of tokenized text if
            there is no padding needed for the batch.
        Args: 
            text (List[str] | str ): A batch of strings or a string. Batches with 
                                    non equal length will be padded.

        Returns:
            (torch.Tensor, torch.Tensor, Tuple): input token ids, last nonmasked idx, 
            Tuple of (batch_size, max len(tokenized_text0, vocab size)) for embeddings
            and each hidden layer (like output of huggingface)
        """
        #Some notes on last nonmasked idx
        #meant to record positions that should be ignored
        #so if we have some input like 
        #[ [ 1 2 3]
        #  [ 1 2 ]]
        # to run a batch the model will pad the second row, and we 
        # will subsequently record that the last_non_masked_idx is 
        # [2, 1]
        raise NotImplementedError

    @torch.no_grad()
    def get_baseline_vectors(self, text, layer_idx, include_punctuation=False):
        """Returns final hidden representation of inputted text at the 
        layer specified in layer idx.
        Args:
            text (List[str] | str): A batch of strings or a string. Padding will 
                                     be removed.
            layer_idx (int): which layer to get representation from, with 
                             0 -> embedding, 1...N -> hidden layer N-1
            include_punct: whether to skip over punctuation (ie rep(word.) -> rep(word) or rep(.|word))

        Notes: eos/bos/cls/sep is removed

        Returns:
            (List, torch.Tensor): correspondings ids, 
                                hidden representations of text at layer idx, size 
                                (batch_size, dim of hidden layer)
        """

        input_ids, last_non_masked_idx, hidden = self.get_hidden_layers(text)
        hidden = hidden[layer_idx]

        return_data = torch.zeros((hidden.shape[0], hidden.shape[-1]), device=hidden.device)
        return_ids = [0]*hidden.shape[0]
        for batch_idx in range(hidden.shape[0]):
            for token_id, h in zip(input_ids[batch_idx], hidden[batch_idx,:, :]):
                if self.token_is_punct(token_id):
                    if not include_punctuation:
                        continue
                if self.token_is_sep(token_id):
                    continue
                return_data[batch_idx] = h
                return_ids[batch_idx] = int(token_id.data)

        return (return_ids, return_data)

    @torch.no_grad()
    def get_layerwise_similarity_to_baseline(self, text, baseline_vector,
            text_layer):
        """Returns input token ids, last nonmasked idx, 
            and similarity between input token and baseline for inputted text and 
            baseline vector.
            last nonmasked idx is as simply as the length of tokenized text if
            there is no padding needed for the batch.
        Args: 
            text (List[str] | str ): A batch of strings or a string. Batches with 
                                    non equal length will be padded.
            baseline_vector torch.Tensor: vector to compare to text representations (should have 
                                         one baseline per batch)
            text_layer (int): layer to compare baseline to 0 -> embedding, 1...N -> layerN

        Returns:
            lists (token, similarity): Lists of (token id, similarity to baseline) that are 
            batch_size X len(tokenized text). 
            Meaning that the padding from get_hidden_layers is removed.
        """

        input_ids, last_non_masked_idx, text_hidden_layers = self.get_hidden_layers(text)
        
        text_representations = text_hidden_layers[text_layer]

        #check dim
        assert baseline_vector.shape[0] == text_representations.shape[0]

        similarities = torch.nn.functional.cosine_similarity(baseline_vector.unsqueeze(1), text_representations, dim=-1)

        return_data = []
        for i in range(similarities.shape[0]):
            #couple steps here, 1) we zip together the token ids and their
            #similarities 2) we map this to ints and floats (out of tensors)
            #3) dump it into a list
            batch_data = list(map(lambda x: (int(x[0].data), 
                float(x[1].data)), zip(input_ids[i], similarities[i])))
            #batch_data = list(map(lambda x: (int(x[0].data), 
            #    float(x[1].data)), zip(target_ids[i], target_surprisals[i])))
            #Chop off the stuff which is padding
            batch_data = batch_data[:last_non_masked_idx[i]+1]
            return_data.append(batch_data)

        return return_data

    @torch.no_grad()
    def get_aligned_words_similarities(self, texts, baseline_texts, 
            layer_number, include_punctuation=False):
        """Returns similarities of each word for inputted text to baseline.
           Note that this requires that you've implemented
           a tokenizer, get_output, token_is_unk, token_is_punct, 
           and word_to_idx
           for the model instance.
           Note assumes that words are space seperated in text.
           Basically we assume the following things: 
           1) get_output works as specified 
           2) token_is_unk returns whether a token is unk
           3) token_is_punct returns whether a token is punctuation
           4) word_to_idx will tokenize a word as it would have in the sentence

           The most care should be taken with 4)

           Futher note, eos/bos/sep/cls tokens are thrown out. 
           Also for a word, the similarity 
           is to the last token from the word tokenizer, for both baselines 
           and texts. Say you have the baseline: the dog is happy.
           and texts the dog is happy but he loves Fig.

           If punctuation is included, the the baseline representation 
           will correspond to the period after happy, and Fig will correspond
           to the punct. If punctation is not included then it's just happy and 
           Fig (ignoring .)

        Args: 
            texts (List[str] | str ): A batch of strings or a string.
            baseline_texts (List[str] | str): A batch of strings or a string 
                                    to be the comparision for similarity
            layer_number (int): Layer to use 0 -> embedding 1...N -> hidden layer N-1
            include_punctuation: Whether to include punctuation in surprisal


        Returns:
            list of lists Word: Word is a namedtuple containing
                                word: word in text
                                sim: similarity of word to baseline at layer layer_number
                                layerN: layer number
                                isSplit: whether the word was split by tokenizer
                                isUnk: whether the word is an unk
                                withPunct: whether the sim value is punctuation
                                modelName: name of model
                            batch_size X len(split text)
        """
        #Tuple structure to hold word and other info
        SimWord = namedtuple('SimWord', 'word sim layerN isSplit isUnk baseUnk withPunct modelName')

        ##########################
        #    Get Baseline Vec    #
        ##########################
        baseline_ids, baseline_vectors = self.get_baseline_vectors(baseline_texts, layer_number, include_punctuation=False)

        ##########################
        #    Get Similarities    #
        ##########################
        token_similarities = self.get_layerwise_similarity_to_baseline(texts, baseline_vectors, layer_number)

        #Filter out eos
        filtered_token_similarities = []
        for element in token_similarities:
            filtered_token_similarities.append(list(filter(lambda x: not self.token_is_sep(x[0]), 
                element)))
        token_similarities = filtered_token_similarities

        ##########################
        #      Align Words       #
        ##########################
        
        #batchify if necessary
        if type(texts) == str:
            texts = [texts]

        return_data = []
        for text_idx, text in enumerate(texts):
            #add a holder for this text
            return_data.append([])
            for word_pos, word in enumerate(text.split(' ')):
                sim = 0
                isUnk = 0
                isSplit = 0
                withPunct = 0

                baseUnk = 0
                #check if baseline is unk
                if self.token_is_unk(baseline_ids[text_idx]):
                    baseUnk = 1

                #this is critical for gpt2 (or any that use prefix spacing)
                isFirstWord=False
                if word_pos == 0:
                    isFirstWord=True

                isLastWord = False
                if word_pos == len(text.split(' '))-1:
                    isLastWord = True

                #Tokenize the word
                word_tokens = self.word_to_idx(word, isFirstWord, isLastWord)
                if len(word_tokens) > 1:
                    isSplit = 1

                for word_token in word_tokens:
                    token_similarity = token_similarities[text_idx].pop(0)
                    #here we ensure that things are working by 
                    #checking that the tokenizations match
                    assert word_token == token_similarity[0]

                    #check unk and punct
                    if self.token_is_unk(word_token):
                        isUnk = 1

                    #NB: This skips over internal punctuation 
                    if self.token_is_punct(word_token):
                        if not include_punctuation:
                            continue
                        else:
                            withPunct = 1

                    #Setting to the final subword
                    sim = token_similarity[1]

                w = SimWord(word, sim, layer_number, isSplit, isUnk, baseUnk, withPunct, self.model_name)  
                return_data[text_idx].append(w)

            #Make sure we got it all
            assert len(token_similarities[text_idx]) == 0
        return return_data
