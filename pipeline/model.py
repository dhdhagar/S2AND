import torch

from pipeline.mlp_layer import MLPLayer
from pipeline.sdp_layer import SDPLayer
from pipeline.trellis_cut_layer import TrellisCutLayer


class model(torch.nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.mlp_layer = MLPLayer()
        self.sdp_layer = SDPLayer(max_sdp_iters=50000)
        self.trellis_cut_estimator = TrellisCutLayer()

    def forward(self, x):
        edge_weights = self.mlp_layer(x)
        # Take only the probabilities of belonging to class 1 as output from mlp
        edge_weights = edge_weights[1][:, 1:]
        print(edge_weights)
        print("Size of OP of mlp layer is", edge_weights.size())
        output_probs = self.sdp_layer(edge_weights)
        print("Size of OP of sdp layer is", output_probs.size())
        pred_clustering = self.trellis_cut_estimator(output_probs)
        print("Size of OP of Trellis Cut layer is", pred_clustering.size())
        return pred_clustering
