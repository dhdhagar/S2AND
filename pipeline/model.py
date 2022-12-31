import torch

from pipeline.mlp_layer import MLPLayer
from pipeline.sdp_layer import SDPLayer
from pipeline.hac_cut_layer import HACCutLayer
from pipeline.trellis_cut_layer import TrellisCutLayer
from pipeline.uncompress_layer import UncompressTransformLayer
import logging
from IPython import embed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class EntResModel(torch.nn.Module):
    def __init__(self, n_features, hidden_dim, n_hidden_layers, dropout_p, hidden_config, activation):
        super().__init__()
        self.mlp_layer = MLPLayer(n_features=n_features,
                                  dropout_p=dropout_p,
                                  add_batchnorm=True,
                                  hidden_dim=hidden_dim,
                                  n_hidden_layers=n_hidden_layers,
                                  activation=activation,
                                  hidden_config=hidden_config)
        self.uncompress_layer = UncompressTransformLayer()
        self.sdp_layer = SDPLayer(max_sdp_iters=50000)
        self.hac_cut_layer = HACCutLayer()

    def forward(self, x, N):
        edge_weights = torch.squeeze(self.mlp_layer(x))
        edge_weights_uncompressed = self.uncompress_layer(edge_weights, N)
        output_probs = self.sdp_layer(edge_weights_uncompressed, N)
        pred_clustering = self.hac_cut_layer(output_probs, edge_weights_uncompressed)
        return pred_clustering
