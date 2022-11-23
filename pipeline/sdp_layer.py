import math

import torch
import numpy as np
import logging
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class SDPLayer(torch.nn.Module):
    # Taken from the ECC Clusterer class in ecc_layer.py
    def __init__(self,
                 max_sdp_iters: int):
        super().__init__()
        self.max_sdp_iters = max_sdp_iters
        self.num_ecc = 0

    def build_and_solve_sdp(self):
        # Initialize the cvxpy layer
        n = self.num_points
        self.X = cp.Variable((n, n), PSD=True)
        self.W = cp.Parameter((n, n))

        # build out constraint set
        constraints = [
            cp.diag(self.X) == np.ones((n,)),
            self.X[:n, :] >= 0,
        ]

        # create problem
        self.prob = cp.Problem(cp.Maximize(cp.trace(self.W @ self.X)), constraints)
        # Note: maximizing the trace is equivalent to maximizing the sum_E (w_uv * X_uv) objective
        # because W is upper-triangular and X is symmetric

        # Build the SDP cvxpylayer
        self.cvxpy_layer = CvxpyLayer(self.prob, parameters=[self.W], variables=[self.X])
        print("Built the cvxpy layer")

        logging.info('Solving optimization problem')
        # Forward pass through the SDP cvxpylayer
        pw_probs = self.cvxpy_layer(self.W_val, solver_args={
            "solve_method": "SCS",
            "verbose": True,
            # "warm_start": True,  # Enabled by default
            "max_iters": self.max_sdp_iters,
            "eps": 1e-3
        })

        # Perform the necessary transforms to get final upper triangular matrix of clustering probabilities
        pw_probs = torch.triu(pw_probs[0], diagonal=1)
        with torch.no_grad():
            sdp_obj_value = torch.sum(self.W_val * pw_probs).item()

        # number of active graph nodes we are clustering
        active_n = self.num_points

        # run heuristic max forcing for now
        if self.num_ecc > 0:
            var_assign = []
            for ((_, ecc_idx), satisfying_points) in self.var_vals.items():
                max_satisfy_pt = max(
                    satisfying_points,
                    key=lambda x: self.X.value[ecc_idx + self.num_points, x]
                )
                var_assign.append((ecc_idx, max_satisfy_pt))

            for ecc_idx, point_idx in var_assign:
                self.L.value[ecc_idx, point_idx] = 0.9

            pw_probs = self.X.value[:active_n, :active_n]

            for ecc_idx, point_idx in var_assign:
                self.L.value[ecc_idx, point_idx] = 0.0
        else:
            # pw_probs = self.X.value[:active_n, :active_n]
            pw_probs = pw_probs[:active_n, :active_n]

        # if self.incompat_mx is not None:
        #     # discourage incompatible nodes from clustering together
        #     self.incompat_mx = np.concatenate(
        #         (np.zeros((active_n, self.num_points), dtype=bool),
        #          self.incompat_mx), axis=1
        #     )
        #     pw_probs[self.incompat_mx] -= np.sum(pw_probs)

        # pw_probs = np.triu(pw_probs, k=1)
        # pw_probs.retain_grad()  # debug: view the backward pass result

        return sdp_obj_value, pw_probs

    def forward(self,
                edge_weights):
        # Calculate num of points given mlp output size
        self.num_points = round(math.sqrt(2*edge_weights.size(dim=0))) + 1
        n = self.num_points
        # formulate SDP
        logging.info('Constructing optimization problem')
        # Convert the 1D pairwise-similarities list to nxn upper triangular matrix
        ind = torch.triu_indices(n, n, offset=1)
        self.W_val = (torch.sparse_coo_tensor(ind, edge_weights, [n, n, 1])).to_dense()
        self.W_val = torch.reshape(self.W_val, (n, n))
        self.W_val.retain_grad()

        # Solve the SDP and return result
        sdp_obj_value, pw_probs = self.build_and_solve_sdp()
        return pw_probs
