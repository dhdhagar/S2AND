import torch

from e2e_pipeline.mlp_layer import MLPLayer
from e2e_pipeline.sdp_layer import SDPLayer
from e2e_pipeline.hac_cut_layer import HACCutLayer
from e2e_pipeline.trellis_cut_layer import TrellisCutLayer
from e2e_pipeline.uncompress_layer import UncompressTransformLayer
import logging
from IPython import embed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class PairwiseModel(torch.nn.Module):
    def __init__(self, n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                 neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm, activation,
                 negative_slope, hidden_config):
        super().__init__()
        self.mlp_layer = MLPLayer(n_features=n_features, neumiss_depth=neumiss_depth, dropout_p=dropout_p,
                                  dropout_only_once=dropout_only_once, add_neumiss=add_neumiss, neumiss_deq=neumiss_deq,
                                  hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, add_batchnorm=add_batchnorm,
                                  activation=activation, negative_slope=negative_slope, hidden_config=hidden_config)

    def forward(self, x, N=None, warmstart=False, verbose=False):
        """
        N, warmstart: unused; added to keep forward signature consistent across models
        """
        edge_weights = torch.squeeze(self.mlp_layer(x))

        if verbose:
            logger.info(f"Size of W = {edge_weights.size()}")
            logger.info(f"\n{edge_weights}")

        return edge_weights
