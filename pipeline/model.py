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
    def __init__(self, n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                 neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm, activation,
                 negative_slope, hidden_config, sdp_max_iters, sdp_eps):
        super().__init__()
        self.mlp_layer = MLPLayer(n_features=n_features, neumiss_depth=neumiss_depth, dropout_p=dropout_p,
                                  dropout_only_once=dropout_only_once, add_neumiss=add_neumiss, neumiss_deq=neumiss_deq,
                                  hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, add_batchnorm=add_batchnorm,
                                  activation=activation, negative_slope=negative_slope, hidden_config=hidden_config)
        self.uncompress_layer = UncompressTransformLayer()
        self.sdp_layer = SDPLayer(max_iters=sdp_max_iters, eps=sdp_eps)
        self.hac_cut_layer = HACCutLayer()

    def forward(self, x, N, verbose=False):
        edge_weights = torch.squeeze(self.mlp_layer(x))
        edge_weights_uncompressed = self.uncompress_layer(edge_weights, N)
        output_probs = self.sdp_layer(edge_weights_uncompressed, N)
        pred_clustering = self.hac_cut_layer(output_probs, edge_weights_uncompressed)

        if verbose:
            logger.info("Size of W is %s", edge_weights.size())
            logger.info("W")
            logger.info(edge_weights)

            logger.info("Size of Uncompressed W is %s", edge_weights_uncompressed.size())
            logger.info(edge_weights_uncompressed)

            logger.info("Size of X is %s", output_probs.size())
            logger.info("X")
            logger.info(output_probs)

            logger.info("Size of HAC Cut OP is %s", pred_clustering.size())
            logger.info("HAC Cut OP")
            logger.info(pred_clustering)

        return pred_clustering
