import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from Modules.Config import Config

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        '''
        Returns a multi-head attention-scaled hidden state.

        Each head may learn to "focus" on specfic parts of the input, e.g. syntax, semantics, etc.

        :param config: (Config) The transformer config file.
        '''
        super().__init__()
        d_model = config.transformer_params['d_model']
        num_heads = config.transformer_params['num_attention_heads']
        head_dim = d_model // num_heads # The heads should cumilatively have approximately the same compute as a single large head
        
        self.heads = nn.ModuleList([AttentionHead(d_model, head_dim) for _ in range(num_heads)])
        self.linear_output = nn.Linear(d_model, d_model)

    def forward(self, hidden_state: torch.tensor) -> torch.tensor:
        # Passing it through each head and concatenating all the results restore the tensor to the size (1, d_model) TODO correct dimensions?
        attn_state = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        return self.linear_output(attn_state) # Then passing the attention through a single linear layer

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, head_dim: int) -> None:
        '''
        Returns an attention-scaled hidden state using scaled dot product attention.

        Returns softmax(QK^T/sqrt(dim_k))V, where Query Q, Key K and Value V are calculated by passing a hidden
        state through linear layers.

        :param d_model: (int) Dimension of the embedding vector.
        :param head_dim: (int) Output dimension of the head.
        '''
        super().__init__()
        self.q = nn.Linear(d_model, head_dim)
        self.k = nn.Linear(d_model, head_dim)
        self.v = nn.Linear(d_model, head_dim)

    def forward(self, hidden_state: torch.tensor) -> torch.tensor:
        q = self.q(hidden_state)
        k = self.k(hidden_state)
        v = self.v(hidden_state)

        # weights are defined as softmax(QK^T/sqrt(dim_k)), which then scale the values V
        dim_k = q.size(-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(dim_k) # Scores are how much queries align with keys (hence dot product QK^T, normalised by sqrt(dimension))
        weights = F.softmax(scores, dim=-1)
        
        # Scaling values by the calculated weights
        return torch.bmm(weights, v)

class FeedForward(nn.Module):
    def __init__(self, config: Config) -> None:
        '''
        Two linear layers with a GELU in between and dropout (1D convolutional layer with kernel size 1).

        :param config: (Config) The transformer config file.
        '''
        super().__init__()
        self.linear_1 = nn.Linear(config.transformer_params['d_model'], config.transformer_params['d_ff'])
        self.linear_2 = nn.Linear(config.transformer_params['d_ff'], config.transformer_params['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.transformer_params['dropout'])

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x