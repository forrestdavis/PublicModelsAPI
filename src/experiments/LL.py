import pandas as pd
import dill
import os
import sys
from ..models.models import load_models
from .Experiment import Experiment

class LL(Experiment):

    def __init__(self, fname):
        super().__init__()
        self.name = fname
        self.dataframe = None

    def load_dataframe(self):
        if '.csv' == self.name[-4:]:
            self.dataframe = pd.read_csv(self.name)
        elif '.tsv' == self.name[-4:]:
            self.dataframe = pd.read_csv(self.name, sep='\t')
        else:
            sys.stderr.write(f"The file {self.name} is not a recognized format\n")
            sys.exit(1)

    def to_dataframe(self):
        """Makes pandas dataframe from experiment.
        """
        return self.dataframe

    def load_experiment(self):
        """Loads the respective experiment and sets
        to self._stimuli

        Returns:
            list: collection of Sentence instances
        """
        self.load_dataframe()

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

    def get_likelihood_results(self, model, batch_size=40,
                             lowercase=True):

        pass

        '''
        if self.dataframe is None:
            self.load_experiment()

        contexts = self.dataframe['sent'].to_list()
        targets = self.dataframe['target'].to_list()

        target_measure = []
        for idx in range(0, len(targets), batch_size):
            context_batch = contexts[idx:idx+batch_size]
            if lowercase:
                context_batch = list(map(lambda x: x[0].lower()+x[1:], context_batch))
            target_batch = targets[idx:idx+batch_size]
            #Expand out special tokens 
            target_batch = self.expand_strings(target_batch)
            if return_type == 'prob':
                target_measure.extend(model.get_targeted_word_probabilities(context_batch, target_batch))
            elif return_type == 'surp':
                target_measure.extend(model.get_targeted_word_surprisals(context_batch, target_batch))
            else:
                import sys
                sys.stderr.write(f"return type {return_type} not recognized")
                sys.exit(1)

        assert len(contexts) == len(target_measure)
        if return_type == 'prob':
            self.dataframe[str(model)+'_prob'] = target_measure
        elif return_type == 'surp':
            self.dataframe[str(model)+'_surp'] = target_measure

        '''
