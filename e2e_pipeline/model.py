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

class EntResModel(torch.nn.Module):
    def __init__(self, n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                 neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm, activation,
                 negative_slope, hidden_config, sdp_max_iters, sdp_eps, use_rounded_loss=True,
                 scale_sdp_input=False):
        super().__init__()
        # Layers
        self.mlp_layer = MLPLayer(n_features=n_features, neumiss_depth=neumiss_depth, dropout_p=dropout_p,
                                  dropout_only_once=dropout_only_once, add_neumiss=add_neumiss, neumiss_deq=neumiss_deq,
                                  hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, add_batchnorm=add_batchnorm,
                                  activation=activation, negative_slope=negative_slope, hidden_config=hidden_config)
        self.uncompress_layer = UncompressTransformLayer(scale=scale_sdp_input)
        self.sdp_layer = SDPLayer(max_iters=sdp_max_iters, eps=sdp_eps)
        self.hac_cut_layer = HACCutLayer()
        # Configs
        self.use_rounded_loss = use_rounded_loss

    def forward(self, x, N, warmstart=False, verbose=False):
        edge_weights = torch.squeeze(self.mlp_layer(x))
        if verbose:
            logger.info(f"Size of W = {edge_weights.size()}")
            logger.info(f"\n{edge_weights}")
        if warmstart:
            return edge_weights

        edge_weights_uncompressed = self.uncompress_layer(edge_weights, N)
        if verbose:
            logger.info(f"Size of W_matrix = {edge_weights_uncompressed.size()}")
            logger.info(f"\n{edge_weights_uncompressed}")

        output_probs = self.sdp_layer(edge_weights_uncompressed, N)
        if verbose:
            logger.info(f"Size of X = {output_probs.size()}")
            logger.info(f"\n{output_probs}")
        if self.training and not self.use_rounded_loss:
            return output_probs

        pred_clustering = self.hac_cut_layer(output_probs, edge_weights_uncompressed)
        if verbose:
            logger.info(f"Size of X_r = {pred_clustering.size()}")
            logger.info(f"\n{pred_clustering}")

        return pred_clustering
