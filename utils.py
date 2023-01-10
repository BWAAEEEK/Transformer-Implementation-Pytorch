import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w2(self.dropout(self.activation(self.w1(x))))


class SubLayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))
