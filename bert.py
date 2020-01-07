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

class littleBERT(nn.Module):
    def __init__(self, n_head, d_model=512, d_embedding=256, 
                 n_layers=1, dim_feedforward=1536, dropout=0.0, src_rev_usage=True, repeat_input=False):
        super(littleBERT, self).__init__()

        # Setting
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.src_rev_usage = src_rev_usage
        self.repeat_input = repeat_input

        self.dropout = nn.Dropout(dropout)

        # Source Embedding Part - (1) continous variable to vector
        # self.prelu = nn.PReLU()
        # self.lrelu = nn.LeakyReLU()
        self.embed1 = nn.Linear(1, d_embedding)
        self.embed2 = nn.Linear(d_embedding, d_model)

        # Source Embedding Part - (2) weekday
        self.embed_weekday1 = nn.Embedding(7, self.d_embedding)
        self.embed_weekday2 = nn.Embedding(self.d_embedding, self.d_model)

        # Source Embedding Part - (2) weekday
        self.embed1_rev = nn.Linear(1, d_embedding)
        self.embed2_rev = nn.Linear(d_embedding, d_model)

        # Output Linear Part
        self.src_output_linear = nn.Linear(d_model, d_embedding)
        self.src_output_concatlinear = nn.Linear((d_embedding + d_embedding), d_embedding)
        self.src_output_bilinear = nn.Bilinear(d_embedding, d_embedding, d_embedding)
        self.src_output_linear2 = nn.Linear(d_embedding, 1)

        # Transformer
        encoder_self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, encoder_self_attn, dim_feedforward,
                activation='gelu', dropout=dropout) for i in range(n_layers)])

    def forward(self, src, src_rev):

        if self.repeat_input:
            encoder_out1 = self.embed2(src.unsqueeze(2).repeat(1, 1, self.d_embedding)).transpose(0, 1)
        else:
            encoder_out1 = self.embed2(self.embed1(src.unsqueeze(2))).transpose(0, 1)

        if self.src_rev_usage:
            if self.repeat_input:
                encoder_out2 = self.embed2(src_rev.unsqueeze(2).repeat(1, 1, self.d_embedding)).transpose(0, 1)
            else:
                encoder_out2 = self.embed2_rev(self.embed1_rev(src_rev.unsqueeze(2))).transpose(0, 1)

        for i in range(len(self.encoders)):
            encoder_out1 = self.encoders[i](encoder_out1)
        if self.src_rev_usage:
            for i in range(len(self.encoders)):
                encoder_out2 = self.encoders[i](encoder_out2)
        
        encoder_out1 = self.dropout(F.gelu(self.src_output_linear(encoder_out1)))
        if self.src_rev_usage:
            encoder_out2 = self.dropout(F.gelu(self.src_output_linear(encoder_out2)))
            encoder_out_cat = torch.cat((encoder_out1, encoder_out2), dim=2)
            encoder_out = self.src_output_concatlinear(encoder_out_cat)
            encoder_out = self.src_output_linear2(encoder_out).transpose(0, 1).contiguous()
        else:
            encoder_out = self.src_output_linear2(encoder_out1).transpose(0, 1).contiguous()

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

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src