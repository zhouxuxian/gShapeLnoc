import torch
import torch.nn as nn
from models.GTShapelet import GTShapelet

#包含了图编码器和shapelet部分
class LncLoc(nn.Module):
    def __init__(self, k, embed_dim=256, num_heads=4, hidden_dim=256):
        super(LncLoc, self).__init__()
        self.graph_encoder = GTShapelet(k, embed_dim, num_heads=num_heads)
        self.classify = nn.Sequential(
            nn.Linear(embed_dim + 200, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, g, sw, sf):
        sw = torch.nn.functional.normalize(sw, p=2, dim=1)
        graph_repr = self.graph_encoder(g, sw)
        graph_repr = torch.concat((sf, graph_repr), dim=1)
        ####################
        # sf = torch.nn.functional.normalize(sf, p=2, dim=1)
        # graph_repr = torch.concat((sf, graph_repr), dim=1)
        # graph_repr = torch.nn.functional.normalize(graph_repr, p=2, dim=1)
        #####################
        return self.classify(graph_repr)

    def load_pretrain_encoder(self, source):
        source_state = source
        for name, param in self.named_parameters():
            if name in source_state:
                param.data = source_state[name]

    def set_parameter(self):
        for name, param in self.named_parameters():
            if 'graph_encoder' in name:
                continue
            if 'norm' in name:
                continue
            if 'bias' in name:
                nn.init.zeros_(param)
                continue
            nn.init.kaiming_uniform_(param)
