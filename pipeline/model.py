import torch

from pipeline.mlp_layer import MLPLayer
from pipeline.sdp_layer import SDPLayer
from pipeline.hac_cut_layer import HACCutLayer
from pipeline.trellis_cut_layer import TrellisCutLayer
from pipeline.uncompress_layer import UncompressTransformLayer
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class model(torch.nn.Module):
    def __init__(self, hidden_dim, n_hidden_layers, dropout_p, hidden_config, activation):
        super().__init__()
        self.mlp_layer = MLPLayer(n_features=39,
                                  dropout_p=dropout_p,
                                  add_batchnorm=True,
                                  hidden_dim=hidden_dim,
                                  n_hidden_layers=n_hidden_layers,
                                  activation=activation,
                                  hidden_config=hidden_config)
        self.uncompress_layer = UncompressTransformLayer()
        self.sdp_layer = SDPLayer(max_sdp_iters=50000)
        self.hac_cut_layer = HACCutLayer()
        #self.trellis_cut_estimator = TrellisCutLayer()

    def forward(self, x):
        edge_weights = self.mlp_layer(x.float())
        # Reshape to the size required by uncompress layer, viz 1d list
        edge_weights = torch.reshape(edge_weights, (-1,))
        logging.info("Size of W is %s", edge_weights.size())
        logging.info("W")
        logging.info(edge_weights)

        edge_weights_uncompressed = self.uncompress_layer(edge_weights)
        logging.info("Size of Uncompressed W is %s", edge_weights_uncompressed.size())
        logging.info(edge_weights_uncompressed)

        output_probs = self.sdp_layer(edge_weights_uncompressed)
        logging.info("Size of X is %s", output_probs.size())
        logging.info("X")
        logging.info(output_probs)

        pred_clustering = self.hac_cut_layer(output_probs, edge_weights_uncompressed)
        logging.info("Size of HAC Cut OP is %s", pred_clustering.size())
        logging.info("HAC Cut OP")
        logging.info(pred_clustering)
        return pred_clustering
