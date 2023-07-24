import pandas as pd
import dill
import os
import math
import sys
from ..models.models import load_models
from .Experiment import Experiment

class Acceptability(Experiment):

    def __init__(self, fname):
        super().__init__()
        self.name = fname
        self.dataframe = None
        self.outputdata = None

        self.load_dataframe()
        self.measureNames = set([])
        self.modelNames = set([])

    def load_dataframe(self):
        if '.csv' == self.name[-4:]:
            self.dataframe = pd.read_csv(self.name)
        elif '.tsv' == self.name[-4:]:
            self.dataframe = pd.read_csv(self.name, sep='\t')

        else:
            sys.stderr.write(f"The file {self.name} is not a recognized format\n")
            sys.exit(1)

    #TODO: For god's sake find a cleaner way to do this
    def flatten(self):
        """ Flatten the dataframe to have joint variable for models and another for
        the relevant measure. 

        Paramters: 
            self.dataframe: Internal representation of the data. 

        Modifies the self.dataframe variable 
        """
        self.dataframe['model'] = ''
        AllColumns = list(self.dataframe.columns)
        columns = list(filter(lambda x: x in self.measureNames, AllColumns))
        base_columns = list(filter(lambda x: x not in self.measureNames, AllColumns))

        self.dataframe = pd.melt(self.dataframe, base_columns, var_name='measure', 
                                value_name='rating')

        models = []
        measures = []
        for name in self.dataframe['measure'].tolist():
            for model in self.modelNames:
                if model +'_' in name:
                    models.append(model)
                    measures.append(name.replace(model+'_', ''))
                    break

        self.dataframe['measure'] = measures
        self.dataframe['model'] = models

    #TODO: Make this faster and clarify the docstring
    def get_acceptability_measures(self, model, unigram_model,
                                   batch_size=40,
                                  measureType='both'):
        """ Takes output from a model and maps it to the measures in 
        Lau, Clark, and Lappin (2017). Grammaticality, Acceptability, and 
            Probability: A Probabilistic View of Linguistic Knowledge. 
            https://doi.org/10.1111/cogs.12414

        Note: Relies on RTModel instances with get_aligned_words_surprisal
        working. 

        Arguments: 
            model (RTModel): RTModel instance
            unigram_model (dict) : dictionary with mapping from words to counts
                            will do add one smoothing incase of missing
                            words (THIS NEEDS TO BE RETHOUGHT)
            batch_size (int): batch size to use for measures
            measureType (str): either sentence, word, or both
                               sentence skips word level ones
                               word skips sentence level ones
                               both includes everything

        The specific measure definitions are given on pages 1222 and 1223
        of Lau et al. (2017). 

        These include: 
            LogProb
            Mean LogProb
            Norm LogProb (div)
            Norm LogProb (sub)
            SLOR
            Word LogProb Min- 1
            Word LogProb Min- 2
            Word LogProb Min- 3
            Word LogProb Min- 4
            Word LogProb Min- 5
            Word LogProb Mean
            Word LogProb Mean-Q1
            Word LogProb Mean-Q2
        """
        if self.dataframe is None:
            self.load_dataframe()

        sents = self.dataframe['sent'].tolist()
        
        if measureType == 'both':
            measures = {'LogProb': [], 'MeanLP': [], 'NormLPDiv': [], 
                   'NormLPSub': [], 'SLOR': [], 
                   'WordLPMin-1': [], 'WordLPMin-2': [], 'WordLPMin-3': [], 
                   'WordLPMin-4': [], 'WordLPMin-5': [], 'WordLPMean': [], 
                   'WordLPMeanQ1': [], 'WordLPMeanQ2': []}
        elif measureType == 'sentence':
            measures = {'LogProb': [], 'MeanLP': [], 'NormLPDiv': [], 
                   'NormLPSub': [], 'SLOR': []}
        elif measureType == 'word':
            measures = {'WordLPMin-1': [], 'WordLPMin-2': [], 'WordLPMin-3': [], 
                   'WordLPMin-4': [], 'WordLPMin-5': [], 'WordLPMean': [], 
                   'WordLPMeanQ1': [], 'WordLPMeanQ2': []}

        TOTAL = sum(unigram_model.values()) + len(unigram_model)

        # Get batches
        for idx in range(0, len(sents), batch_size):

            batch = sents[idx:idx+batch_size]
            batch_surps = model.get_aligned_words_surprisals(batch)

            # Calculate measures
            # TODO: This is very slow, I wonder if I can speed this
            #       up by more directly yielding these measures...

            for surps in batch_surps:
                LP = 0
                LP_Unigram = 0
                length = 0

                # Tuple (word, LP, LP_unigram)
                words = []
                for idx in range(1, len(surps)):
                    surp = surps[idx]

                    #TODO: THIS NEEDS TO BE TESTED THOROUGHLY
                    #TODO: Externalize this
                    assert len(model.word_to_idx(surp.word)) > 0,\
                        f"{surp.word}"\
                        "{model.word_to_idx(surp.word)}"

                    if idx == len(surps)-1:
                        wordID = model.word_to_idx(surp.word, False, True)[0]
                    else:
                        wordID = model.word_to_idx(surp.word, False, False)[0]

                    if model.token_is_punct(wordID):
                        continue

                    length += 1
                    LP += -surp.surp
                    uni = 0
                    if surp.word in unigram_model:
                        uni += unigram_model[surp.word] + 1
                    if '\u0120'+surp.word in unigram_model:
                        uni += unigram_model['\u0120'+surp.word] + 1
                    if uni == 0:
                        uni = 1
                        uni = math.log2(uni/TOTAL+1)
                    else:
                        uni = math.log2(uni/TOTAL)

                    LP_Unigram += uni

                    words.append((surp.word, -surp.surp, uni, surp.surp/uni))

                if measureType in {'sentence', 'both'}:
                    measures['LogProb'].append(LP)
                    measures['MeanLP'].append(LP/length)
                    measures['NormLPDiv'].append(-(LP/LP_Unigram))
                    measures['NormLPSub'].append(LP - LP_Unigram)
                    measures['SLOR'].append((LP-LP_Unigram)/length)

                if measureType in {'word', 'both'}:
                    #By word
                    words.sort(key = lambda x: x[1])
                    measures['WordLPMin-1'].append(words[0][-1])
                    measures['WordLPMin-2'].append(words[1][-1])
                    measures['WordLPMin-3'].append(words[2][-1])
                    measures['WordLPMin-4'].append(words[3][-1])
                    measures['WordLPMin-5'].append(words[4][-1])
                    measures['WordLPMean'].append(sum(map(lambda x: x[-1],
                                                         words))/len(words))
                    Q1 = words[:int(len(words)*0.25)]
                    Q2 = words[:int(len(words)*0.5)]
                    measures['WordLPMeanQ1'].append(sum(map(lambda x: x[-1], Q1))/len(Q1))
                    measures['WordLPMeanQ2'].append(sum(map(lambda x: x[-1], Q2))/len(Q2))

        # Add to dataframe
        for row in measures:
            self.dataframe[str(model)+'_'+row] = measures[row]
            self.measureNames.add(str(model)+'_'+row)
            self.modelNames.add(str(model))

    def plot(self, X, Y,
             hue=None, measure=None, 
             plotType = 'box', outname=None):

        import seaborn as sns
        import matplotlib.pyplot as plt

        if self.dataframe is None:
            self.load_dataframe()

        plot_data = self.dataframe.copy()

        # Filter out other measures
        plot_data = plot_data[plot_data['measure'] == measure]

        if plotType == 'box':
            # Add a dummy variable (useful for plotting simple contrasts)
            if X is None:
                X = 'exp'
                if 'exp' not in plot_data.columns.tolist():
                    plot_data['exp'] = 'na'

            if hue is not None:
                g = sns.catplot(
                    data=plot_data,
                    x=X,
                    y=Y,
                    col='model',
                    hue=hue,
                    kind='box')
            else:
                g = sns.catplot(
                    data=plot_data,
                    x=X,
                    y=Y,
                    col='model',
                    kind='box')

        elif plotType == 'corr':

            if hue is not None:
                g = sns.lmplot(data=plot_data, 
                               x=X, y=Y, 
                               col='model')
            else:
                g = sns.lmplot(data=plot_data, 
                               x=X, y=Y, hue=hue, 
                               col='model')
        if outname is None:
            plt.show()
        else:
            plt.savefig(outname)

        return

