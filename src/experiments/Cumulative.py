import pandas as pd
import dill
import os
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

    def load_dataframe(self):
        if '.csv' == self.name[-4:]:
            self.dataframe = pd.read_csv(self.name)
        elif '.tsv' == self.name[-4:]:
            self.dataframe = pd.read_csv(self.name, sep='\t')

        else:
            sys.stderr.write(f"The file {self.name} is not a recognized format\n")
            sys.exit(1)

    def get_likelihood_results(self, model, batch_size=40):

        if self.dataframe is None:
            self.load_dataframe()

        columns = self.dataframe.columns.tolist()

        LLs = []

        if 'context' in columns:
            sents = self.dataframe['sent'].tolist()
            contexts = self.dataframe['context'].tolist()

            assert len(contexts) == len(sents)

            for idx in range(0, len(sents), batch_size):
                context_batch = contexts[idx:idx+batch_size]
                sent_batch = sents[idx:idx+batch_size]

                encodings = model.tokenizer.batch_encode_plus(context_batch)['input_ids']
                startPOSs = []
                for e in encodings:
                    startPOSs.append(len(e))

                batch = []
                for context, sent in zip(context_batch, sent_batch):
                    batch.append(context+' '+sent)

                LLs.extend(model.get_sentence_likelihood(batch, 
                                                         startPOS = startPOSs))

        else:
            sents = self.dataframe['sent'].tolist()

            for idx in range(0, len(sents), batch_size):
                sent_batch = sents[idx:idx+batch_size]

                LLs.extend(model.get_sentence_likelihood(sent_batch, startPOS=6))

        assert len(LLs) == len(sents)
        self.dataframe[str(model)+'_LL'] = LLs
