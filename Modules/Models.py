import torch
import torch.nn as nn
import os

from Modules.Transformer.Encoder import Encoder
from Modules.Config import Config

class Classifier(nn.Module):
    '''
    A classification model which categorises tokenised inputs into a 
    '''
    def __init__(self, config_path: os.PathLike, num_labels: int) -> None:
        super().__init__()
        config = Config(config_path)

        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.transformer_params['dropout'])
        self.classification_layer = nn.Linear(config.transformer_params['d_model'], num_labels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encoder(x)
        x = self.dropout(x)
        return self.classification_layer(x)
    
class GPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError