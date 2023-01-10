import torch
import torch.nn as nn
from attention import MultiHeadAttention
from utils import SubLayerConnection, PositionWiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_dim, dropout):
        super().__init__()

        self.attention = MultiHeadAttention(attn_heads, hidden, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model=hidden, d_ff=feed_dim, dropout=dropout)
        self.input_sublayer = SubLayerConnection(hidden, dropout)
        self.output_sublayer = SubLayerConnection(hidden, dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x), mask=mask)
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

