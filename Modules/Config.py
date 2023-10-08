import yaml
import os

class Config():
    def __init__(self, config_path: os.PathLike):
        """
        Formats config data from a yaml for use by KonnichiwaLM.

        :param config_path: (os.PathLike) Path to the config yaml to be read.
        """
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        self.transformer_params = params['transformer_params']
        self.tokeniser_params = params['tokeniser_params']

        # Converting str to float
        self.transformer_params['embedding']['epsilon'] = float(self.transformer_params['embedding']['epsilon'])