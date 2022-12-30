import torch
import numpy as np
import logging
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from IPython import embed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SDPLayer(torch.nn.Module):
    # Taken from the ECC Clusterer class in ecc_layer.py
    def __init__(self, max_sdp_iters: int, N_max: int):
        super().__init__()
        self.max_sdp_iters = max_sdp_iters

        # Initialize the cvxpy layer
        self.X = cp.Variable((N_max, N_max), PSD=True)
        self.W = cp.Parameter((N_max, N_max))
        # build constraints set
        constraints = [
            cp.diag(self.X) == np.ones((N_max,)),
            self.X[:N_max, :] >= 0,
        ]
        # create problem
        self.prob = cp.Problem(cp.Maximize(cp.trace(self.W @ self.X)), constraints)
        # Note: maximizing the trace is equivalent to maximizing the sum_E (w_uv * X_uv) objective
        #       because W is upper-triangular and X is symmetric
        # Build the SDP cvxpylayer
        self.cvxpy_layer = CvxpyLayer(self.prob, parameters=[self.W], variables=[self.X])

    def build_and_solve_sdp(self, W_val, N, verbose=False):
        # Forward pass through the SDP cvxpylayer
        pw_probs = self.cvxpy_layer(W_val, solver_args={
            "solve_method": "SCS",
            "verbose": verbose,
            # "warm_start": True,  # Enabled by default
            "max_iters": self.max_sdp_iters,
            "eps": 1e-3,
        })[0]

        sdp_obj_value = None
        if verbose:
            with torch.no_grad():
                sdp_obj_value = torch.sum(W_val * torch.triu(pw_probs, diagonal=1)).item()
                logger.info(f'SDP objective = {sdp_obj_value}')

        return pw_probs[:N, :N], sdp_obj_value

    def forward(self, edge_weights_uncompressed, N, verbose=False):
        return self.build_and_solve_sdp(edge_weights_uncompressed, N, verbose)
