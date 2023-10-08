import torch
import torch.nn as nn

from Modules.Encoder import Encoder
from Modules.Config import Config

class SentimentClassifier(nn.Module):
    def __init__(self, config: Config, num_labels: int) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encoder(x)
        x = self.dropout(x)
        return self.classifier(x)