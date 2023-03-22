import pandas as pd
import glob
import sys
import json
from .Experiment import Experiment


class MinimalPair(Experiment):

    def __init__(self):
        super().__init__()
        self.name = "MinimalPair"
        self.dataframe = None
        self.measureTypes = set([])
        self.modelNames = set([])

    #TODO: Expand for CLiMP
    #TODO: Generalize to a specific data format
    def load_experiment(self, expType, path='', 
                       phenomenon=None):
        """Loads either BLiMP data or SLING data. 
        """
        import glob
        import json

        if expType == 'BLiMP':
            self.load_blimp()

        elif expType == 'SLING':
            self.load_sling(path)
            if phenomenon:
                self.dataframe = self.dataframe[self.dataframe['phenomenon'] ==
                                                phenomenon]

    def load_dataframe(self, fname):
        if '.csv' == fname[-4:]:
            self.dataframe = pd.read_csv(fname)
        elif '.tsv' == fname[-4:]:
            self.dataframe = pd.read_csv(fname, sep='\t')
        else:
            sys.stderr.write(f"The file {fname} is not a recognized format\n")
            sys.exit(1)

    def load_sling(self, path):
        """Loads all SLING data taken from 
        Song, Krishna, Bhatt, and Iyyer (2022). 
            SLING: Sino Linguistic Evaluation of Large Language Models. 
            https://aclanthology.org/2022.emnlp-main.305/

        Clone their repo: 
        https://github.com/Yixiao-Song/SLING_Data_Code

        Args:
            path (str): path to SLING_Data_Code

        updates self.dataframe with SLING data 
        """
        files = glob.glob(f"{path}/**/*.jsonl", recursive = True)

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
        # Check that you have the data
        try:
            import datasets
        except:
            sys.stderr.write('Install datasets to run BLiMP: pip install datasets\n')
            sys.exit(1)

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

    def gatherInfo(self):
        columns = self.dataframe.columns.tolist()
        for entry in columns:
            if not ('_good' in entry or '_bad' in entry):
                continue
            if entry in {'sentence_good', 'sentence_bad'}:
                continue
            vals = entry.split('_')
            self.measureTypes.add(vals[-1])
            vals = vals[:-2]
            modelName = '_'.join(vals)
            self.modelNames.add(modelName)

    #TODO: implement
    def flatten(self):
        """ Flatten the dataframe to have variable for models and another for
        the relevant comparison between good and bad sentences.
        Paramters:
            self.dataframe: Internal representation of the data.
        Modifies the self.dataframe variable
        """

        self.gatherInfo()
        for model in self.modelNames:
            for measure in self.measureTypes:
                if measure == "ppl":
                    self.dataframe[f"{model}_score"] = (
                        self.dataframe[f"{model}_good_{measure}"] <
                        self.dataframe[f"{model}_bad_{measure}"]).astype(int)
                else:
                    self.dataframe[f"{model}_score"] = (
                        self.dataframe[f"{model}_good_{measure}"] >
                        self.dataframe[f"{model}_bad_{measure}"]).astype(int)

        base_columns = list(filter(lambda x: '_score' not in x, 
                                   self.dataframe.columns))
        self.dataframe = pd.melt(self.dataframe,
                          id_vars=base_columns,
                          var_name='model',
                          value_name='score')

        models = self.dataframe['model'].tolist()
        for idx, m in enumerate(models):
            models[idx] = m.replace('_score', '')
        self.dataframe['model'] = models

    def get_results(self, model, measure='ppl', batch_size=40):

        if self.dataframe is None:
            sys.stderr.write('Did not load MinimalPair experiment\n')
            sys.exit(1)

        good_sents = self.dataframe['sentence_good'].tolist()
        bad_sents = self.dataframe['sentence_bad'].tolist()

        assert len(good_sents) == len(bad_sents)

        good = []
        bad = []
        for idx in range(0, len(good_sents), batch_size):
            good_batch = good_sents[idx:idx+batch_size]
            bad_batch = bad_sents[idx:idx+batch_size]

            if measure == 'LL':
                good.extend(model.get_sentence_likelihood(good_batch))
                bad.extend(model.get_sentence_likelihood(bad_batch))
            elif measure == 'ppl':
                good_ppls = model.get_by_sentence_perplexity(good_batch)
                # Strip off sentences
                good_ppls = list(map(lambda x: x[1], good_ppls))
                good.extend(good_ppls)

                bad_ppls = model.get_by_sentence_perplexity(bad_batch)
                # Strip off sentences
                bad_ppls = list(map(lambda x: x[1], bad_ppls))
                bad.extend(bad_ppls)
            else:
                sys.stderr.write(f"The measure {measure} not supported for " +
                                 "MinimalPair experiments at this time.\n")
                sys.exit(1)

        assert len(good) == len(bad) != 0

        self.dataframe[str(model)+'_good_'+measure] = good
        self.dataframe[str(model)+'_bad_'+measure] = bad

        self.measureTypes.add(measure)
        self.modelNames.add(str(model))
