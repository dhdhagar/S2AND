import torch

from pipeline.mlp_layer import MLPLayer
from pipeline.sdp_layer import SDPLayer
from pipeline.trellis_cut_layer import TrellisCutLayer
from pipeline.uncompress_layer import UncompressTransformLayer


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_layer = MLPLayer(n_features=39)
        self.uncompress_layer = UncompressTransformLayer()
        self.sdp_layer = SDPLayer(max_sdp_iters=50000)
        self.trellis_cut_estimator = TrellisCutLayer()

    def forward(self, x):
        edge_weights = self.mlp_layer(x.float())
        # Reshape to the size required by uncompress layer, viz 1d list
        edge_weights = torch.reshape(edge_weights, (-1,))
        print("Size of OP of mlp layer is", edge_weights.size())
        print(edge_weights)

        edge_weights_uncompressed = self.uncompress_layer(edge_weights)
        print("Size of Uncompressed similarity matrix is", edge_weights_uncompressed.size())

        output_probs = self.sdp_layer(edge_weights_uncompressed)
        # Convert upper triangular output to a symmetric matrix
        output_probs = output_probs + torch.transpose(output_probs, 0, 1) - torch.diag(output_probs)
        print("Size of OP of sdp layer is", output_probs.size())

        pred_clustering = self.trellis_cut_estimator(edge_weights_uncompressed, output_probs)
        print("Size of OP of Trellis Cut layer is", pred_clustering.size())
        return pred_clustering
