import pandas as pd
import dill
import os
import math
import sys
from ..models.models import load_models
from .Experiment import Experiment

try:
    import datasets
except:
    sys.stderr.write('Install datasets to run BLiMP: pip install datasets\n')
    sys.exit(1)

class Cumulative(Experiment):

    def __init__(self, fname):
        super().__init__()
        self.name = fname
        self.dataframe = None
        self.outputdata = None

        self.load_dataframe()
        self.special_strings = {}
        self.load_special_list()

    def load_dataframe(self):
        if '.csv' == self.name[-4:]:
            self.dataframe = pd.read_csv(self.name)
        elif '.tsv' == self.name[-4:]:
            self.dataframe = pd.read_csv(self.name, sep='\t')

        else:
            sys.stderr.write(f"The file {self.name} is not a recognized format\n")
            sys.exit(1)

    def load_special_list(self):
        """Loads the verbs from Newman et al. (2021) 
        which are contained in BERT/RoBERTa/LSTMs/GPT2/TFXL.
        """
        english_fname = 'stimuli/combined_verb_list_vocab_checked.tsv'
        spanish_fname = 'stimuli/spanish_verbs_vocab_checked.tsv'
        spanish_adj_fname = 'stimuli/spanish_adjectives_vocab_checked.tsv'

        en_verb_data = pd.read_csv(english_fname, sep='\t')
        addSpanish = False
        if os.path.exists(spanish_fname):
            es_verb_data = pd.read_csv(spanish_fname, sep='\t')
            es_adj_data = pd.read_csv(spanish_adj_fname, sep='\t')
            addSpanish = True

        #filter out of vocab verb pairs
        en_verb_data = en_verb_data[en_verb_data['inVocabs'] == 1]
        if addSpanish:
            es_verb_data = es_verb_data[es_verb_data['inVocabs'] == 1]
            es_adj_data = es_adj_data[es_adj_data['inVocabs'] == 1]

        #Holds special string by list of strings 
        #   (e.g., '$SG' -> ['runs', 'eats', ...]
        self.special_strings = {'$SG': [],
                                '$PL': []}
        if addSpanish:
            self.special_strings['$ES_SG'] = []
            self.special_strings['$ES_PL'] = []
            self.special_strings['$ES_Adj_m'] = []
            self.special_strings['$ES_Adj_f'] = []

        self.special_strings['$SG'].extend(en_verb_data['sing'].to_list())
        self.special_strings['$PL'].extend(en_verb_data['plur'].to_list())

        if addSpanish:
            self.special_strings['$ES_SG'].extend(es_verb_data['sing'].to_list())
            self.special_strings['$ES_PL'].extend(es_verb_data['plur'].to_list())
            self.special_strings['$ES_Adj_m'].extend(es_adj_data['m_sg'].to_list())
            self.special_strings['$ES_Adj_f'].extend(es_adj_data['f_sg'].to_list())

    def expand_strings(self, targets):
        """Returns targets with any special strings 
        expanded into their lists.

        Returns: 
            List[List[str] | str]: List of target strings and list of strings 
        """

        expanded_targets = []
        for target in targets:
            if target in self.special_strings:
                expanded_targets.append(self.special_strings[target])
            else:
                expanded_targets.append(target)

        return expanded_targets
        
    def get_likelihood_results(self, model, batch_size=40, log=False):
        """ This should be cleaned up... especially by encapsulating 
        the batching and expand_strings bit..."""

        if self.dataframe is None:
            self.load_dataframe()

        columns = self.dataframe.columns.tolist()

        LLs = []

        if 'prime' in columns:
            sents = self.dataframe['target'].tolist()
            contexts = self.dataframe['prime'].tolist()

            assert len(contexts) == len(sents)


            for idx in range(0, len(sents), batch_size):
                context_batch = contexts[idx:idx+batch_size]
                sent_batch = sents[idx:idx+batch_size]

                #Expand out special tokens
                sent_batch = self.expand_strings(sent_batch)

                # We are dealing with a wildcard, need care with batching
                if type(sent_batch[0]) == list:

                    # Rebatch around expanded set
                    for context, sent in zip(context_batch, sent_batch):

                        # Get length of context in encoding
                        encoding = model.tokenizer.batch_encode_plus(
                            [context])['input_ids'][0]

                        PROBS = []

                        # Loop through batches of expanded targets
                        for j in range(0, len(sent), batch_size):

                            # Create a batch and the end of context list
                            batch = []
                            startPOSs = []
                            for s in sent[j:j+batch_size]:
                                batch.append(context+' '+s)
                                startPOSs.append(len(encoding))

                            # Get probability of target conditioned on context
                            PROBS.extend(model.get_sentence_likelihood(batch, 
                                                        startPOS = startPOSs, 
                                                        log = False))

                        # The LL is the sum of the probs (over all the wildcard)
                        LL = sum(PROBS)
                        if log:
                            LL = math.log2(LL)
                        LLs.append(LL)

                else:
                    encodings = model.tokenizer.batch_encode_plus(context_batch)['input_ids']
                    startPOSs = []
                    for e in encodings:
                        startPOSs.append(len(e))

                    batch = []
                    for context, sent in zip(context_batch, sent_batch):
                        batch.append(context+' '+sent)

                    LLs.extend(model.get_sentence_likelihood(batch, 
                                                             startPOS = startPOSs, 
                                                             log = log))

        else:
            sents = self.dataframe['sent'].tolist()

            for idx in range(0, len(sents), batch_size):
                sent_batch = sents[idx:idx+batch_size]

                LLs.extend(model.get_sentence_likelihood(sent_batch, 
                                                        log = log))

        assert len(LLs) == len(sents)
        self.dataframe[str(model)+'_LL'] = LLs
