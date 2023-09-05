import torch.nn as nn
import torch.nn.functional as F

from Modules.Transformer.Encoder import Encoder
from Modules.Config import Config

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder, src_embed, tgt_embed, generator) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoding = self.encoder(self.src_embed(src), src_mask)
        return self.decoder(self.tgt_embed(tgt), encoding, src_mask, tgt_mask)