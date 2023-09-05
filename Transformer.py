import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoding = self.encoder(self.src_embed(src), src_mask)
        return self.decoder(self.tgt_embed(tgt), encoding, src_mask, tgt_mask)
    
class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.hidden_layers)])

class Embedding(nn.Module):
    def __init__(self, config) -> None:
        raise NotImplementedError
    
class EncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.d_model)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        '''
        Returns a multi-head attention-scaled hidden state.

        Each head may learn to "focus" on specfic parts of the input, e.g. syntax, semantics, etc.
        '''
        super().__init__()
        d_model = config.d_model
        num_heads = config.num_attention_heads
        head_dim = d_model // num_heads # Each head should have an equal amount of compute
        
        self.heads = nn.ModuleList([AttentionHead(d_model, head_dim) for _ in range(num_heads)])
        self.linear_output = nn.Linear(d_model, d_model)

    def forward(self, hidden_state) -> torch.tensor:
        # Passing it through each head and concatenating all the results restore the tensor to the size (1, d_model) TODO correct dimensions?
        attn_state = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        return self.linear_output(attn_state) # Then passing the attention through a single linear layer

class AttentionHead(nn.Module):
    def __init__(self, d_model, head_dim) -> None:
        '''
        Returns an attention-scaled hidden state.

        Uses squared dot product attention:
        Returns softmax(QK^T/sqrt(dim_k))V, where Query Q, Key K and Value V are calculated by passing a hidden
        state through linear layers.
        '''
        super().__init__()
        self.q = nn.Linear(d_model, head_dim)
        self.k = nn.Linear(d_model, head_dim)
        self.v = nn.Linear(d_model, head_dim)

    def forward(self, hidden_state) -> torch.tensor:
        q = self.q(hidden_state)
        k = self.k(hidden_state)
        v = self.v(hidden_state)

        # weights = softmax(QK^T/sqrt(dim_k))
        dim_k = q.size(-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(dim_k) # Scores are how much queries align with keys (hence dot product QK^T, normalised by sqrt(dimension))
        weights = F.softmax(scores, dim=-1)
        
        # Scaling values by the calculated weights
        return torch.bmm(weights, v)

class FeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear_1 = nn.Linear()
        