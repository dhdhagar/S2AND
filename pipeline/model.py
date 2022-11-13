import torch

from pipeline.mlp_layer import MlpLayer
from pipeline.sdp_layer import SdpLayer
from pipeline.trellis_cut_layer import TrellisCutLayer


class model(torch.nn.Module):
    def __init__(self):
        self.mlp_layer = MlpLayer()
        self.sdp_layer = SdpLayer(max_num_ecc=1, max_sdp_iters=50000, max_pos_feats=6)
        self.trellis_cut_estimator = TrellisCutLayer()

    def forward(self, x, gold_clustering):
        edge_weights = self.mlp_layer(x)
        print("Size of OP of mlp layer is "+edge_weights.size())
        output_probs = self.sdp_layer(edge_weights, gold_clustering)
        print("Size of OP of sdp layer is "+output_probs.size())
        return output_probs
