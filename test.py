import torch
import torch.nn as nn
import math
import torch_directml

device = torch_directml.device()


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性投影
        k = (
            self.k_linear(k)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = (
            self.q_linear(q)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(v)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 应用注意力权重到V
        output = torch.matmul(attention, v)

        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(self, d_model=64, d_ff=256, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model * 4, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class MiniTransformer(nn.Module):
    def __init__(
        self, vocab_size=1000, d_model=64, num_heads=4, num_layers=2, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.linear(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# 使用示例
if __name__ == "__main__":
    # 超参数
    vocab_size = 1000
    d_model = 64
    num_heads = 4
    num_layers = 2
    seq_len = 10
    batch_size = 4

    # 创建模型
    model = MiniTransformer(vocab_size, d_model, num_heads, num_layers)

    # 创建随机输入
    input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    output = model(input_seq)

    print("输入形状:", input_seq.shape)
    print("输出形状:", output.shape)  # 应该是 (batch_size, seq_len, vocab_size)
