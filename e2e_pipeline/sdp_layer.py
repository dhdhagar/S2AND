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


class CvxpyException(Exception):
    def __init__(self, data=None):
        self.data = data


class SDPLayer(torch.nn.Module):
    def __init__(self, max_iters: int = 50000, eps: float = 1e-3):
        super().__init__()
        self.max_iters = max_iters
        self.eps = eps
        self.objective_value = None  # Stores the last run objective value

    def build_and_solve_sdp(self, W_val, N, verbose=False):
        """
        W_val is an NxN upper-triangular (shift 1) matrix of edge weights
        Returns a symmetric NxN matrix of fractional, decision values with a 1-diagonal
        """
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
        try:
            pw_probs = self.cvxpy_layer(W_val, solver_args={
                "solve_method": "SCS",
                "verbose": verbose,
                "max_iters": self.max_iters,
                "eps": self.eps
            })[0]
        except:
            logger.error(f'CvxpyException: Error running forward pass on W_val of shape {W_val.shape}')
            raise CvxpyException(data={
                                     'W_val': W_val.detach().cpu().numpy(),
                                     'solver_args': {
                                         "solve_method": "SCS",
                                         "verbose": verbose,
                                         "max_iters": self.max_iters,
                                         "eps": self.eps
                                     }
                                 })

        with torch.no_grad():
            objective_matrix = W_val * torch.triu(pw_probs, diagonal=1)
            objective_value_IC = torch.sum(objective_matrix).item()
            objective_value_MA = objective_value_IC - torch.sum(objective_matrix[objective_matrix < 0]).item()
            if verbose:
                logger.info(f'SDP objective: intra-cluster={objective_value_IC}, max-agree={objective_value_MA}')

        return objective_value_MA, pw_probs

    def forward(self, edge_weights_uncompressed, N, verbose=False):
        objective_value, pw_probs = self.build_and_solve_sdp(edge_weights_uncompressed, N, verbose)
        self.objective_value = objective_value
        return pw_probs
