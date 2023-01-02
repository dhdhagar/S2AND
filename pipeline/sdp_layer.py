import math

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
    def __init__(self,
                 max_sdp_iters: int, problem_N=None):
        super().__init__()
        self.max_sdp_iters = max_sdp_iters
        self.num_ecc = 0
        # TODO: Implement problem_N functionality which creates a standard large matrix once and just solves it
        #       in the forward pass each time, rather than building it out each time

    def build_and_solve_sdp(self, W_val, N, verbose=False):
        # Initialize the cvxpy layer
        self.X = cp.Variable((N, N), PSD=True)
        self.W = cp.Parameter((N, N))

        # build out constraint set
        constraints = [
            cp.diag(self.X) == np.ones((N,)),
            self.X[:N, :] >= 0,
        ]

        # create problem
        self.prob = cp.Problem(cp.Maximize(cp.trace(self.W @ self.X)), constraints)
        # Note: maximizing the trace is equivalent to maximizing the sum_E (w_uv * X_uv) objective
        # because W is upper-triangular and X is symmetric

        # Build the SDP cvxpylayer
        self.cvxpy_layer = CvxpyLayer(self.prob, parameters=[self.W], variables=[self.X])

        # Forward pass through the SDP cvxpylayer
        pw_probs = self.cvxpy_layer(W_val, solver_args={
            "solve_method": "SCS",
            "verbose": verbose,
            # "warm_start": True,  # Enabled by default
            "max_iters": self.max_sdp_iters,
            "eps": 1e-3,
        })[0]

        with torch.no_grad():
            sdp_obj_value = torch.sum(W_val * torch.triu(pw_probs, diagonal=1)).item()
            if verbose:
                logger.info(f'SDP objective = {sdp_obj_value}')

        return sdp_obj_value, pw_probs

    def forward(self, edge_weights_uncompressed, N, verbose=False):
        _, pw_probs = self.build_and_solve_sdp(edge_weights_uncompressed, N, verbose)
        return pw_probs
