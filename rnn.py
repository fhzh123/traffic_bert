# Import Modules
import math
import random
import numpy as np

# Import PyTorch
import torch
import torch.nn.functional as F
from torch import nn

class Encoder(nn.Module):
    def __init__(self, d_embedding, d_hidden, n_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.d_hidden = d_hidden
        self.d_embedding = d_embedding
        self.embed = nn.Linear(1, d_embedding)
        self.gru = nn.GRU(d_embedding, d_hidden, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(d_hidden*2, d_hidden)
        self.dropout = dropout
        
    def forward(self, src, hidden=None, cell=None):
        # Source sentence embedding
        embeddings = self.embed(src.unsqueeze(2))  # (max_caption_length, batch_size, embed_dim)
        # Bidirectional GRU
        outputs, hidden = self.gru(embeddings, hidden)
        # sum bidirectional outputs
        outputs = torch.tanh(self.linear(outputs)) # (max_caption_length, batch_size, embed_dim)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, d_embedding, d_hidden, n_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        self.d_hidden = d_hidden
        self.d_embedding = d_embedding
        self.embed = nn.Linear(1, d_embedding)
        self.gru = nn.GRU(d_embedding, d_hidden, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(d_hidden*2, d_hidden)
        self.dropout = dropout
        
    def forward(self, src, t, hidden, cell=None):
        # Source sentence embedding
        # embeddings = self.embed(src.unsqueeze(2))  # (max_caption_length, batch_size, embed_dim)
        # Last Hidden
        if t == 0:
            last_hidden = hidden.view(12, -1, self.d_hidden)[-1].unsqueeze(1)
        else:
            last_hidden = hidden
        # Bidirectional GRU
        outputs, hidden = self.gru(src, last_hidden)
        # sum bidirectional outputs
        outputs = torch.tanh(self.linear(outputs)) # (max_caption_length, batch_size, embed_dim)
        return outputs, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, dropout=0.0):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(self.encoder.d_hidden, self.decoder.d_hidden)
        self.last_linear = nn.Linear(self.decoder.d_hidden, 1)

        self.dropout = dropout

    def forward(self, src, trg):
        max_len = trg.size(1)
        batch_size = trg.size(0)
        outputs = torch.zeros(max_len, batch_size, self.encoder.d_hidden).cuda()
        # Encoding source sentences
        encoder_output, hidden = self.encoder(src)
        hidden = torch.tanh(self.linear(hidden))
        # Decoding
        output = torch.zeros(batch_size, 256).unsqueeze(1).cuda()
        for t in range(12):
            output, hidden = self.decoder(output, t, hidden)
            outputs[t] = output.squeeze(1)
        result = self.last_linear(outputs)
        return result.transpose(0, 1)