import torch
from utils.convert_lgbm_to_torch import convert_pretrained_model


class MLPLayer(torch.nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.mlp_model = convert_pretrained_model(dropout_prob)

    def forward(self, x):
        return self.mlp_model(x)