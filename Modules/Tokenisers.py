from transformers import AutoTokenizer

from Modules.Config import Config

class WordTokeniser:
    def __init__(self) -> None:
        raise NotImplementedError
    
class SubWordTokeniser:
    def __init__(self) -> None:
        raise NotImplementedError
    
def get_huggingface_tokeniser(model_name: str) -> AutoTokenizer:
    '''
    Generates one of HuggingFace's tokenisers.

    :param model_name: (str) Name of the tokeniser (e.g. distilbert-base-uncased)
    :return: (AutoTokeniser) A HuggingFace tokeniser
    '''
    return AutoTokenizer.from_pretrained(model_name)

def generate_tokeniser(config: Config):
    '''
    Generates a tokeniser specified in the config.

    :param config: (Config) The transformer config.
    :return: A tokeniser instance
    '''
    tokeniser_type = config.tokeniser_params['type']

    if tokeniser_type == 'WordTokeniser':
        return WordTokeniser()
    elif tokeniser_type == 'SubWordTokeniser':
        return SubWordTokeniser()
    else:
        try:
            return get_huggingface_tokeniser(tokeniser_type)
        except OSError as e:
            raise ValueError(f"The tokeniser '{tokeniser_type}' doesn't exist: {e}")

