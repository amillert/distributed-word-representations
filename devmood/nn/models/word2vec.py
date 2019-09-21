import torch.nn as nn
import torch.nn.functional as f


class CBOW(nn.Module):
    def __init__(self, in_dim, h_dim, fake_idx):
        super().__init__()
        self.embedding_layer_in = nn.Embedding(in_dim, h_dim, padding_idx=fake_idx)
        # vocab x hidden
        self.hidden_layer = nn.Linear(h_dim, in_dim)
        # hidden x vocab

    def forward(self, input):
        # input:
        # B x D
        embeds = self.embedding_layer_in(input).sum(dim=1)
        # embedding layer:
        # V x H
        # embeds:
        # B x D x H sum(dim=1) -> B x H
        hidden = self.hidden_layer(embeds).squeeze(0)
        # hidden layer:
        # H x V
        # hidden:
        # B x V
        return f.log_softmax(hidden, dim=1)

