from .Experiment import Experiment
import pandas as pd

class Incremental(Experiment):

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

    def get_incremental(self, model, batch_size=40, 
                        lowercase=True, include_punctuation=False,
                        return_type='prob'):

        sents = self.dataframe['sent'].to_list()

        measures = []
        for idx in range(0, len(sents), batch_size):
            sent_batch = sents[idx:idx+batch_size]

            if lowercase:
                sent_batch = list(map(lambda x: x[0].lower()+x[1:], sent_batch))

            if return_type == 'surp':
                measures.extend(model.get_aligned_words_surprisals(sent_batch,
                                                                  include_punctuation))
            elif return_type == 'prob':
                measures.extend(model.get_aligned_words_probabilities(sent_batch,
                                                                  include_punctuation))
            else:
                import sys
                sys.stderr.write(f"return type {return_type} not recognized")
                sys.exit(1)

        assert len(measures) == len(sents)

        newdata = {'sent': [], 'word':[], 'pos':[], 'isSplit':[], 
                   'isUnk':[], 'withPunct':[], return_type:[]}

        for measure, sent in zip(measures, sents):
            for pos, word in enumerate(measure):
                newdata['sent'].append(sent)
                newdata['word'].append(word.word)
                newdata['pos'].append(pos)
                newdata['isSplit'].append(word.isSplit)
                newdata['isUnk'].append(word.isUnk)
                newdata['withPunct'].append(word.withPunct)
                if return_type == 'prob':
                    value = word.prob
                else:
                    value = word.surp

                newdata[return_type].append(value)

        self.outputdata = pd.DataFrame.from_dict(newdata)

    def save(self, filename):

        assert self.outputdata is not None

        self.outputdata.to_csv(filename, sep='\t', index=False)
