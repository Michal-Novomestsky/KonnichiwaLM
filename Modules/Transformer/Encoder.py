import torch
import torch.nn as nn

from Modules.Config import Config
from Modules.Transformer.TransformerLayers import MultiHeadAttention, FeedForward

class Encoder(nn.Module):
    def __init__(self, config: Config) -> None:
        '''
        An encoder block with dropout and pre-layer normalisation.

        Transforms a one hot tensor of size vocab_size to a hidden state of size d_model

        :param config: (Config) The transformer config file.
        '''
        super().__init__()
        self.embedding = Embedding(config)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.transformer_params['encoder_layers'])])

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Converting one-hot vectors into embeddings
        x = self.embedding(x)
        # Feeding through hidden layers
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class Embedding(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.transformer_params['embedding']['vocab_size'], config.transformer_params['d_model'])
        self.positional_embedding = nn.Embedding(config.transformer_params['embedding']['max_positional_embeddings'], config.transformer_params['d_model'])
        self.layer_norm = nn.LayerNorm(config.transformer_params['d_model'], eps=config.transformer_params['embedding']['epsilon'])
        self.dropout = nn.Dropout()

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Creating position IDs (i.e. indices [0, seq_len-1] corresponding to each position in the input sequence)
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        tkn_embd = self.token_embedding(x)
        pos_embd = self.positional_embedding(pos_ids)

        return self.layer_norm(tkn_embd + pos_embd)
    
class EncoderLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        '''
        A single Encoder layer. Performs self-attention with dropout and pre-layer normalisation:

        -------------------------  ---------------------------------------
        |                       |  |                                     |
        ^                       +  ^                                     +
        x --> LayerNorm --> Attention --> LayerNorm --> Feedforward --> output

        Pre-layer norm is generally more stable than post-layer, as helps prevent gradients from diverging
        and doesn't necessarily require warm-up.

        :param config: (Config) The transformer config file.
        '''
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.transformer_params['d_model'])
        self.layer_norm_2 = nn.LayerNorm(config.transformer_params['d_model'])
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Generating Q, K and V inputs with a layer norm
        hidden_state = self.layer_norm_1(x)
        # Applying attention with a skip-connection
        x += self.attention(hidden_state)
        # Feedforward with a skip-connection
        x += self.feed_forward(self.layer_norm_2(x))
        return x
        