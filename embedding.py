# Import Module
import math

# Import PyTorch
import torch
import torch.nn as nn

# Positional Embedding
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# Token Embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, pad_idx=0):
        super().__init__(vocab_size, embed_size, padding_idx=pad_idx, scale_grad_by_freq=True)

# Total Embedding
class TransformerEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEmbedding : adding positional information using sin, cos
    sum of all these features are output of Embedding
    """

    def __init__(self, vocab_size, d_model, embed_size, pad_idx=0, max_len=512, 
            embedding_dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, pad_idx=pad_idx)
        self.linear_layer = nn.Linear(embed_size, d_model)
        self.position = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence):
        x = self.linear_layer(self.token(sequence)) + self.position(sequence)
        return self.norm(x)