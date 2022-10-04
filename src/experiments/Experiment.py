import pandas as pd
import dill
from collections import defaultdict
from ..models.models import load_models

"""Defines basic class for experiment"""
class Experiment(object):

    def __init__(self):
        """Initialize dataset.

        Args:
            config (dict): AAA
        """
        self._stimuli = []
        self.name = ''
        self.model_instances = []

    def get_models(self, run_config):
        """Loads model instances and 
        dumps them in model_instances.
        Model instances will be run 
        for chosen experiment.
        """
        self.model_instances = load_models(run_config)

    def __str__(self):
        return self.name

    def __iter__(self):
        """Useful iter so you can loop over 
        stimuli.
        """
        for stimulus in self.stimuli:
            yield stimulus

    def __len__(self):
        return len(self.stimuli)

    @property
    def stimuli(self):
        """Contains the groupings of text from experiment.
        """
        return self._stimuli

    def save(self, filename):
        """Saves measures from experiment by dumping into 
        dataframe and saving as tsv.
        Args:
            filename: output filename.
        """
        dataframe = self.to_dataframe()
        dataframe.to_csv(filename, sep='\t', index=False)

    def save_binary(self, filename):
        """Saves stimuli from experiment as binary.
        Args: 
            filename: output filename
        """
        with open(filename, 'wb') as f:
            dill.dump(self.stimuli, f)

    def load_binary(self, filename):
        """Loads stimuli from experiment as binary.
        Args: 
            filename: binary filename
        """
        with open(filename, 'rb') as f:
            self._stimuli = dill.load(f)

    def to_dataframe(self):
        """Makes pandas dataframe from experiment.
        """
        raise NotImplementedError

    def load_experiment(self):
        """Loads the respective experiment and sets
        to self._stimuli

        Returns:
            list: collection of Sentence instances
        """
        raise NotImplementedError

    #TODO
    def update_words_with_surprisals(self, aligned_words_surprisals):
        """FIX"""

        for doc_idx, document in enumerate(self.documents):
            doc_words_surps = aligned_words_surprisals[doc_idx]
            for word in document.words():
                word_surp = doc_words_surps.pop(0)
                assert word_surp.word == word.text
                word.unkd = word_surp.isUnk
                word.subworded = word_surp.isSplit
                word.add_surp(word_surp.modelName, word_surp.surp)

"""The various containers for different grains of data in Experiment: Word, Sentence, Stimulus"""
class Stimulus(object):
    def __init__(self, stimID):
        """Stimulus is meant to hold stimulus level groupings of text from the experiment.
            It contains sentences of type Sentence
        Args: 
            sentID: identifier of stimulus (meant to be unique, item-like)
        Attributes:
            text: string of text
            sentID: identifier of sentence
            words: instances of Word from experiment
        """

        self.stimID = stimID
        self.sentences = []

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence

    def words(self):
        """Is an iter if you want to just look at words.
        """
        for sentence in self.sentences:
            for word in sentence:
                yield word

    def get_words(self):
        return list(self.words())

    def __str__(self):
        return f"stimID: {self.stimID}"


class Sentence(object):

    def __init__(self, text, sentID):
        """Sentence is meant to hold sentence level groupings of text from the experiment.
            It contains words of type Word 
        Args: 
            text: text from experiment (a sentence)
            sentID: identifier of sentence (meant to be unique)
        Attributes:
            text: string of text
            sentID: identifier of sentence
            words: instances of Word from experiment """

        self.text = text
        self.sentID = sentID
        self.words = []

    def __iter__(self):
        for word in self.words:
            yield word

    def __str__(self):
        return f"SentID: {self.sentID} Text: {self.text}"

class Word(object):

    def __init__(self, text, wordID):
        """Word is meant to hold each word of text from the experiment. 
            There are different types of Words depending on the measurement
            type.
        Args: 
            text: text from experiment (a sentence)
            wordID: identifier of word (meant to be unique)
        Attributes:
            text: string of text
            wordtID: identifier of word
            surps (dict): for each unique model, it's surprisal value for that word
            pos_in_sent: where in the sentence the word is 
            length: string length of word
            text_clean: text with no punctuation
            unkd: whether the word is unk'd by model 
            subworded: whether the word was split into subwords (as in Byte encoding from GPT2, or because of punct)
            word_POS: POS of word
        """

        self.text = text
        self.wordID = wordID
        #model X surp value
        self.surps = {}
        #model X sim 
        self.sims = {}

        #some optional things
        self.pos_in_sent = 0
        self.length = 0
        self.text_clean = ''
        self.unkd = 0
        self.subworded = 0
        self.withPunct = 0


    def __str__(self):
        return f"WordID: {self.wordID} Text: {self.text}"


    def add_surp(self, model_name, surp):
        """Adds surprisal to word. 
            Note: Only allows one surprisal per model
        
        Args:
            model_name: name of model (ie path and model filename)
            surp: surprisal value for this model for this word

        """
        if model_name not in self.surps:
            self.surps[model_name] = surp
        #We won't allow more than one surprisal from the same model
        #for the same word instance at this time
        else:
            sys.stderr.write('model_name '+model_name+' has two surprisal values for this '\
            'word: '+str(self)+'\n')
            sys.exit(1)
