import torch
import torch.nn as nn

class JarvisTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, max_length=256, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        if seq_len > self.max_length:
            raise ValueError(f"Sequence too long: {seq_len} > {self.max_length}")

        positions = torch.arange(seq_len, device=tokens.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(tokens) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool),
            diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)
        return self.output(x)
