import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(query, key, value, mask=None, dropout=None):
        print(query.size(-1))
        score = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(query.size(-1)))

        if mask is not None:
            score = score.masked_fill(mask == 0, -torch.inf)

        p_attn = torch.softmax(score, dim=-1)

        print(score[mask == 0])

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        self.dropout = dropout

        self.h = h  # the number of attention head
        self.d_k = d_model // h  # the dimension size for one head

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query = self.q(query).view(batch_size, -1, self.h, self.d_k)
        key = self.k(key).view(batch_size, -1, self.h, self.d_k)
        value = self.v(value).view(batch_size, -1, self.h, self.d_k)

        x, attn = self.attention(query, key, value, mask, self.dropout)

        x = x.reshape(batch_size, -1, self.h * self.d_k)

        return x


if __name__ == "__main__":
    transformer = MultiHeadAttention(5, 10)

    q = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], dtype=torch.float)
    k = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], dtype=torch.float)
    v = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], dtype=torch.float)

    out = transformer(q, k, v)

