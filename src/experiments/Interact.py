from .Experiment import Experiment

class Interact(Experiment):

    def __init__(self, run_config):
        super().__init__()
        self.name = 'Interact'

        #Load model
        self.get_models(run_config)

        if ('include_punct' not in run_config or 
                not run_config['include_punct']):
            self.include_punctuation = False
        else:
            self.include_punctuation = True

        if 'language' not in run_config:
            self.language = 'en'
        else:
            self.language = run_config['language']

    def run_interact(self):

        #header = ['word', 'isSplit', 'isUnk', 'withPunct', 'modelName', 'surp', 'prob']
        word = "word" + ' '*16
        split = "Split" 
        unk = "Unk"
        punct = "Punct"
        model = "ModelName" + ' '*11
        surp = "surp" + ' '*4
        prob = "prob" + ' '*6
        header = f"{word} | {split} | {unk} | {punct} | {model} | {surp} | {prob}"

        while True:
            sent = input('string: ').strip()
            print(header)
            print('-'*len(header))
            output = self.get_interactive_output(sent, self.include_punctuation, 
                                                self.language)
            for word in output:
                surp = round(word.surp, 3)
                prob = round(2**(-word.surp), 5)
                print_out = f"{word.word: <20} | {word.isSplit:5} | {word.isUnk:3} | {word.withPunct:5} | {word.modelName.split('/')[-1]: <20} | {surp: >8} | {prob: >10}"
                print(print_out)

    def get_interactive_output(self, sent, include_punctuation=False, 
                              language='en'):

        assert len(self.model_instances) == 1, 'No models loaded for interactive testing'

        for model in self.model_instances:
            #Get output and flatten
            output_by_word = model.get_aligned_words_surprisals(
                                sent, include_punctuation, 
                                    language)[0]
            return output_by_word
