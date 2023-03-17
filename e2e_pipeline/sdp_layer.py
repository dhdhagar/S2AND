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


def get_max_agree_objective(weights, probs, verbose=False):
    with torch.no_grad():
        objective_matrix = weights * torch.triu(probs, diagonal=1)
        objective_value_IC = torch.sum(objective_matrix).item()
        objective_value_MA = objective_value_IC - torch.sum(objective_matrix[objective_matrix < 0]).item()
        if verbose:
            logger.info(f'SDP objective: intra-cluster={objective_value_IC}, max-agree={objective_value_MA}')
        return objective_value_MA


class SDPLayer(torch.nn.Module):
    def __init__(self, max_iters: int = 50000, eps: float = 1e-3, scale_input=False):
        super().__init__()
        self.max_iters = max_iters
        self.eps = eps
        self.scale_input = scale_input
        self.objective_value = None  # Stores the last run objective value

    def build_and_solve_sdp(self, W_val, N, verbose=False):
        """
        W_val is an NxN upper-triangular (shift 1) matrix of edge weights
        Returns a symmetric NxN matrix of fractional, decision values with a 1-diagonal
        """
        # Initialize the cvxpy layer
        X = cp.Variable((N, N), PSD=True)
        W = cp.Parameter((N, N))

        # build out constraint set
        constraints = [
            cp.diag(X) == np.ones((N,)),
            X[:N, :] >= 0,
        ]

        try:
            if self.training:
                # create problem
                prob = cp.Problem(cp.Maximize(cp.trace(W @ X)), constraints)
                # Note: maximizing the trace is equivalent to maximizing the sum_E (w_uv * X_uv) objective
                # because W is upper-triangular and X is symmetric
                # Build the SDP cvxpylayer
                cvxpy_layer = CvxpyLayer(prob, parameters=[W], variables=[X])
                # Forward pass through the SDP cvxpylayer
                pw_prob_matrix = cvxpy_layer(W_val, solver_args={
                    "solve_method": "SCS",
                    "verbose": verbose,
                    "max_iters": self.max_iters,
                    "eps": self.eps
                })[0]
            else:
                # create problem
                prob = cp.Problem(cp.Maximize(cp.trace(W_val.cpu().numpy() @ X)), constraints)
                prob.solve(
                    solver=cp.SCS,
                    verbose=verbose,
                    max_iters=self.max_iters,
                    eps=self.eps
                )
                pw_prob_matrix = torch.tensor(X.value, device=W_val.device)
            # Fix to prevent invalid solution values close to 0 and 1 but outside the range
            pw_prob_matrix = torch.clamp(pw_prob_matrix, min=0, max=1)
        except:
            logger.error(f'CvxpyException: Error running forward pass on W_val of shape {W_val.shape}')
            raise CvxpyException(data={
                                     'W_val': W_val.detach().tolist(),
                                     'solver_args': {
                                         "solve_method": "SCS",
                                         "verbose": verbose,
                                         "max_iters": self.max_iters,
                                         "eps": self.eps
                                     }
                                 })
        objective_value_MA = get_max_agree_objective(W_val, pw_prob_matrix, verbose=verbose)
        return objective_value_MA, pw_prob_matrix

    def get_sigmoid_matrix(self, W_val, N, verbose=False):
        pw_prob_matrix = torch.sigmoid(W_val)
        objective_value_MA = get_max_agree_objective(W_val, pw_prob_matrix, verbose=verbose)
        return objective_value_MA, pw_prob_matrix

    def forward(self, edge_weights_uncompressed, N, use_sdp=True, return_triu=False, verbose=False):
        W_val = edge_weights_uncompressed
        if self.scale_input:
            with torch.no_grad():
                scale_factor = torch.max(torch.abs(W_val))
            if verbose:
                logger.info(f"Scaling W_val by {scale_factor}")
            W_val /= scale_factor

        solver = self.build_and_solve_sdp if use_sdp else self.get_sigmoid_matrix
        self.objective_value, pw_prob_matrix = solver(W_val, N, verbose)

        if return_triu:
            triu_indices = torch.triu_indices(len(pw_prob_matrix), len(pw_prob_matrix), offset=1)
            return pw_prob_matrix[triu_indices[0], triu_indices[1]]
        return pw_prob_matrix
