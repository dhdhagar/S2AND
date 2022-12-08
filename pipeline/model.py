import torch

from pipeline.mlp_layer import MLPLayer
from pipeline.sdp_layer import SDPLayer
from pipeline.trellis_cut_layer import TrellisCutLayer
from pipeline.uncompress_layer import UncompressTransformLayer


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_layer = MLPLayer()
        self.uncompress_layer = UncompressTransformLayer()
        self.sdp_layer = SDPLayer(max_sdp_iters=50000)
        self.trellis_cut_estimator = TrellisCutLayer()

    def forward(self, x):
        edge_weights = self.mlp_layer(x)
        print("Size of OP of mlp layer is", edge_weights.size())

        edge_weights_uncompressed = self.uncompress_layer(edge_weights)
        print("Size of Uncompressed similarity matrix is", edge_weights_uncompressed.size())

        output_probs = self.sdp_layer(edge_weights_uncompressed)
        print("Size of OP of sdp layer is", output_probs.size())

        pred_clustering = self.trellis_cut_estimator(edge_weights_uncompressed, output_probs)
        print("Size of OP of Trellis Cut layer is", pred_clustering.size())
        return pred_clustering
