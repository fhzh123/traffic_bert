# Import Module
import os
import numpy as np

# Import PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.activation import MultiheadAttention

# Import Custom Module
from embedding import TransformerEmbedding

class littleBERT(nn.Module):
    def __init__(self, n_head, d_model=512, d_embedding=256, 
                 n_layers=1, dim_feedforward=1536,
                 dropout=0.0, embedding_dropout=0.0):
        super(littleBERT, self).__init__()

        # Setting
        self.d_model = d_model

        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = embedding_dropout

        # Source Embedding Part
        self.embed = nn.Linear(1, d_embedding)

        # Output Linear Part
        self.src_output_linear = nn.Linear(d_model, d_embedding)
        self.src_output_bilinear = nn.Bilinear(d_embedding, d_embedding, d_embedding)
        self.src_output_linear2 = nn.Linear(d_embedding, 1)

        # Transformer
        encoder_self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, encoder_self_attn, dim_feedforward,
                activation='gelu', dropout=dropout) for i in range(n_layers)])

    def forward(self, src, src_rev):

        encoder_out1 = self.embed(src).transpose(0, 1)
        encoder_out2 = self.embed(src_rev).transpose(0, 1)

        for i in range(len(self.encoders)):
            encoder_out1 = self.encoders[i](encoder_out1)
        for i in range(len(self.encoders)):
            encoder_out2 = self.encoders[i](encoder_out2)
        
        encoder_out1 = self.dropout(F.gelu(self.src_output_linear(encoder_out1)))
        encoder_out2 = self.dropout(F.gelu(self.src_output_linear(encoder_out2)))
        encoder_out = self.src_output_bilinear(encoder_out1, encoder_out2)
        encoder_out = self.src_output_linear2(encoder_out).transpose(0, 1).contiguous()

        return encoder_out

    def predict(self, src1, src2):

        encoder_out1 = self.embed(src).transpose(0, 1)
        encoder_out2 = self.embed(src_rev).transpose(0, 1)

        for i in range(len(self.encoders)):
            encoder_out1 = self.encoders[i](encoder_out1)
        for i in range(len(self.encoders)):
            encoder_out2 = self.encoders[i](encoder_out2)
        
        encoder_out1 = F.gelu(self.src_output_linear(encoder_out1))
        encoder_out2 = F.gelu(self.src_output_linear(encoder_out2))
        encoder_out = self.src_output_bilinear(encoder_out1, encoder_out2)
        encoder_out = self.src_output_linear2(encoder_out).transpose(0, 1).contiguous()

        return encoder_out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1, 
            activation="relu"):
        
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src