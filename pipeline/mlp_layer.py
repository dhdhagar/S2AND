import torch
from utils.convert_lgbm_to_torch import convert_pretrained_model


class MLPLayer(torch.nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.mlp_model = convert_pretrained_model(dropout_prob)

    def forward(self, x):
        y = self.mlp_model(x)
        # Since output is from hummingbird model, need to extract only the probabilities of class label 1
        edge_weights = y[1][:, 1:]
        edge_weights = torch.reshape(edge_weights, (-1,))
        return edge_weights