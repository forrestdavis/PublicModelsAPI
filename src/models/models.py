"""Returns model.
"""

from .RTModel import RTModel
from .lstm_model import LSTMModel
from .gpt2_model import GPT2Model
from .gptneo_model import GPTNeoModel
from .gptj_model import GPTJModel
from .tfxl_model import TFXLModel
from .bert_model import BERTModel
from .roberta_model import ROBERTAModel
from .unigram_model import UnigramModel
from .xglm_model import XGLMModel

model_nickname_to_RTModel_class = {
        'lstm': LSTMModel,
        'gpt2': GPT2Model, 
        'gptneo': GPTNeoModel,
        'gptj': GPTJModel,
        'transfo-xl-wt103': TFXLModel,
        'tfxl': TFXLModel,
        'bert': BERTModel,
        'roberta': ROBERTAModel,
        'unigram': UnigramModel, 
        'xglm': XGLMModel,
        }

def load_models(config):
    """Loads instances of models specified in config.
    Args:
        config (dict): model-level config dict with model fields (see README)
    Returns:
        list: List of RTModel instances
    """

    return_models = []
    for model_type in config['models']:
        if model_type not in model_nickname_to_RTModel_class:
            raise ValueError(f"Unrecognized model: {model_type}.")

        if model_type != 'lstm':
            for model_instance in config['models'][model_type]:
                model_class = model_nickname_to_RTModel_class.get(model_type)
                assert issubclass(model_class, RTModel)
                return_models.append(model_class(model_instance))
        else:
            model_files = config['models'][model_type]['model_files']
            vocab_files = config['models'][model_type]['vocab_files']
            neural_complexity_path = config['nc_path']
            assert len(model_files) == len(vocab_files), "You must have a vocab file for each model file for LSTMs"
            for model_file, vocab_file in zip(model_files, vocab_files):
                model_class = model_nickname_to_RTModel_class.get(model_type)
                assert issubclass(model_class, RTModel)
                return_models.append(model_class(model_file, vocab_file, neural_complexity_path))

    return return_models
