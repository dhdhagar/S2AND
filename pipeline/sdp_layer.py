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
        #n = self.num_points
        n = 4
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
        # what all should be requires_grad = True?
        self.num_points = edge_weights.size(dim=0)
        # formulate SDP
        logging.info('Constructing optimization problem')
        # Conversion of scipy sparse matrix to tensor not needed as we r getting tensor as input
        #W = csr_matrix((edge_weights.data, (edge_weights.row, edge_weights.col)), shape=(self.n, self.n))
        #self.W_val = torch.tensor(W.todense(), requires_grad=True)
        # Convert the 1D matrix to upper triangular matrix
        edge_weights = edge_weights[:10, :]
        ind = torch.triu_indices(4, 4)
        edge_weights = torch.sparse_coo_tensor(ind, edge_weights, [4, 4])
        edge_weights = edge_weights.to_dense()
        self.W_val = torch.from_numpy(edge_weights)

        # Solve the SDP and return result
        sdp_obj_value, pw_probs = self.build_and_solve_sdp()
        return pw_probs
