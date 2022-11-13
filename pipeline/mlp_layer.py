import torch
from utils.convert_lgbm_to_torch import convert_pretrained_model


class MlpLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_model = convert_pretrained_model()

    def forward(self, x):
        return self.mlp_model(x)