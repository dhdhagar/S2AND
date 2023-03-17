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

class CCInference(torch.nn.Module):
    """
    Correlation clustering inference-only model. Expects edge weights and the number of nodes as input.
    """

    def __init__(self, sdp_max_iters, sdp_eps, sdp_scale, use_sdp):
        super().__init__()
        self.uncompress_layer = UncompressTransformLayer()
        self.sdp_layer = SDPLayer(max_iters=sdp_max_iters, eps=sdp_eps, scale_input=sdp_scale)
        self.hac_cut_layer = HACCutLayer()
        self.use_sdp = use_sdp

    def forward(self, edge_weights, N, min_id=0, threshold=None, verbose=False):
        edge_weights = torch.squeeze(edge_weights)
        if threshold is not None:
            # threshold is used to convert a similarity score (in [0,1]) into edge weights (in R, i.e. + and -)
            edge_weights = torch.sigmoid(edge_weights) - threshold
        edge_weights_uncompressed = self.uncompress_layer(edge_weights, N)
        output_probs = self.sdp_layer(edge_weights_uncompressed, N, use_sdp=self.use_sdp)
        pred_clustering = self.hac_cut_layer(output_probs, edge_weights_uncompressed)

        if verbose:
            logger.info(f"Size of W = {edge_weights.size()}")
            logger.info(f"\n{edge_weights}")

            logger.info(f"Size of W_matrix = {edge_weights_uncompressed.size()}")
            logger.info(f"\n{edge_weights_uncompressed}")

            logger.info(f"Size of X = {output_probs.size()}")
            logger.info(f"\n{output_probs}")

            logger.info(f"Size of X_r = {pred_clustering.size()}")
            logger.info(f"\n{pred_clustering}")

        return (self.hac_cut_layer.cluster_labels + min_id).tolist()
