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

class BLiMP(Experiment):

    def __init__(self):
        super().__init__()
        self.name = "BLiMP"
        self.dataframe = None

    #TODO: Expand for CLiMP
    #TODO: Generalize to a specific data format
    def load_experiment(self, expType, path=''):
        """Loads either BLiMP data or SLING data. 
        """
        import glob
        import json

        if expType == 'BLiMP':
            load_blimp()

        elif expType == 'SLING':

            files = glob.glob("path/**/*.jsonl", recursive = True)

            data = {}
            for fname in files:
                with open(fname, 'r') as file:
                    for line in file:
                        line = line.strip()
                        d = json.loads(line)
                        for key in d:
                            if key not in data:
                                data[key] = []
                            data[key].append(d[key])
            self.dataframe = pd.DataFrame.from_dict(data)

    def load_blimp(self):
        """Loads all BLiMP data taken from HuggingFaces dataset API 
        https://huggingface.co/docs/datasets/v2.6.0/en/index. We extract 
        useful information and put it into a dataframe. 
        """
        blimp_data = {'linguistics_term':[], 'UID': [], 
            'sentence_good':[], 'sentence_bad':[]}

        exps = datasets.get_dataset_config_names("blimp")

        for exp in exps:
            data = datasets.load_dataset("blimp", exp)
            data = data['train']

            blimp_data['linguistics_term'].extend(data['linguistics_term'])
            blimp_data['UID'].extend(data['UID'])
            blimp_data['sentence_good'].extend(data['sentence_good'])
            blimp_data['sentence_bad'].extend(data['sentence_bad'])

        self.dataframe = pd.DataFrame.from_dict(blimp_data)

    def combine_precompiled_experiments(self, fnames):

        for x, fname in enumerate(fnames):
            if x == 0:
                self.dataframe = pd.read_csv(fname)
            else:
                exisiting_columns = set(self.dataframe.columns.to_list())
                data = pd.read_csv(fname)
                current_columns = data.columns.to_list()
                for column in current_columns:
                    if column not in exisiting_columns:
                        self.dataframe[column] = data[column]

    def save_flattened(self, fname):

        columns = list(filter(lambda x: '_prob' in x, self.dataframe.columns.to_list()))
        base_columns = list(filter(lambda x: '_prob' not in x, self.dataframe.columns.to_list()))

        lstms = list(filter(lambda x: 'lstm' in x.lower(), columns))

        return_data = self.dataframe.copy()

        #Get row-wise average of lstms
        if len(lstms) != 0:
            return_data['lstm_avg_prob'] = return_data[lstms].mean(axis=1)

        #pivot around model names
        return_data = pd.melt(return_data, 
                id_vars=base_columns, var_name='model', value_name='prob')
        #rename model names to something sensible
        return_data['model'] = list(map(lambda x: x.split('/')[-1].replace('_prob', ''), return_data['model'].to_list()))

        return_data.to_csv(fname, index=False)

    def get_likelihood_results(self, model, batch_size=40):

        if self.dataframe is None:
            self.load_experiment()

        good_sents = self.dataframe['sentence_good']
        bad_sents = self.dataframe['sentence_bad']

        assert len(good_sents) == len(bad_sents)

        good_LLs = []
        bad_LLs = []
        for idx in range(0, len(good_sents), batch_size):
            good_batch = good_sents[idx:idx+batch_size]
            bad_batch = bad_sents[idx:idx+batch_size]

            good_LLs.extend(model.get_sentence_likelihood(good_batch))
            bad_LLs.extend(model.get_sentence_likelihood(bad_batch))

        assert len(good_LLs) == len(bad_LLs)
        self.dataframe[str(model)+'_good'] = good_LLs
        self.dataframe[str(model)+'_bad'] = bad_LLs
