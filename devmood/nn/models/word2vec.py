import torch.nn as nn
import torch.nn.functional as f


class SkipGram(nn.Module):
    def __init__(self, in_dim, h_dim, fake_idx):
        super().__init__()
        self.embedding_layer_in = nn.Embedding(in_dim, h_dim, padding_idx=fake_idx)
        self.hidden_layer = nn.Linear(h_dim, in_dim)

    def forward(self, input):
        embeds = self.embedding_layer_in(input).sum(dim=1)
        hidden = self.hidden_layer(embeds)
        return f.log_softmax(hidden, dim=1)
