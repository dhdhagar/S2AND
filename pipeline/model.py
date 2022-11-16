import torch

from pipeline.mlp_layer import MLPLayer
from pipeline.sdp_layer import SDPLayer
from pipeline.trellis_cut_layer import TrellisCutLayer


class model(torch.nn.Module):
    def __init__(self, block_size):
        self.mlp_layer = MLPLayer()
        self.sdp_layer = SDPLayer(num_points=block_size, max_num_ecc=1, max_sdp_iters=50000)
        self.trellis_cut_estimator = TrellisCutLayer()

    def forward(self, x, gold_clustering):
        edge_weights = self.mlp_layer(x)
        print("Size of OP of mlp layer is "+edge_weights.size())
        output_probs = self.sdp_layer(edge_weights, gold_clustering)
        print("Size of OP of sdp layer is "+output_probs.size())
        pred_clustering = self.trellis_cut_estimator(output_probs)
        print("Size of OP of Trellis Cut layer is "+pred_clustering.size())
        return pred_clustering
