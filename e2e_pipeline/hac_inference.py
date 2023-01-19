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

class HACInference:
    """
    HAC inference-only.
    """

    def __init__(self):
        super().__init__()

    def fit(self):
        pass

    def cluster(self):
        pass
