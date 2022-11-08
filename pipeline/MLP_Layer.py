import torch
from utils.convert_lgbm_to_torch import convert_pretrained_model


class MLP_Layer(torch.Module):
    def __init__(self):
        super().__init__()
        self.model = convert_pretrained_model()

    def forward(self, x):
        return self.model(x)