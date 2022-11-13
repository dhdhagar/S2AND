import torch
import numpy as np
from scipy.sparse import csr_matrix
from ecc.ecc_layer import cluster_labels_to_matrix
import logging
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class SdpLayer(torch.nn.Module):
    # Taken from the ECC Clusterer class in ecc_layer.py
    def __init__(self,
                 max_num_ecc: int,
                 max_pos_feats: int,
                 max_sdp_iters: int):
        self.max_num_ecc = max_num_ecc
        self.max_pos_feats = max_pos_feats
        self.max_sdp_iters = max_sdp_iters
        self.num_ecc = 0
        self.ecc_constraints = []

        self.ecc_mx = None
        self.incompat_mx = None

        # Attributes that will be initialized in the forward pass
        self.edge_weights = csr_matrix(
            (10, 10)).tocoo()  # Initialize to something random, will be changed in forward pass or keep it as None
        self.gold_clustering = csr_matrix((10,))  # Initialize to something random, will be changed in forward pass
        self.gold_clustering = None
        self.gold_clustering_matrix = None
        self.num_points = None
        self.W_val = None
        self.X = None
        self.W = None
        self.prob = None
        self.sdp_layer = None


    def build_and_solve_sdp(self):
        logging.info('Solving optimization problem')

        # Forward pass through the SDP cvxpylayer
        pw_probs = self.sdp_layer(self.W_val, solver_args={
            "solve_method": "SCS",
            "verbose": True,
            # "warm_start": True,  # Enabled by default
            "max_iters": self.max_sdp_iters,
            "eps": 1e-3
        })
        pw_probs = torch.triu(pw_probs[0], diagonal=1)
        with torch.no_grad():
            sdp_obj_value = torch.sum(self.W_val * pw_probs).item()

        # number of active graph nodes we are clustering
        active_n = self.num_points + self.num_ecc

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

        if self.incompat_mx is not None:
            # discourage incompatible nodes from clustering together
            self.incompat_mx = np.concatenate(
                (np.zeros((active_n, self.num_points), dtype=bool),
                 self.incompat_mx), axis=1
            )
            pw_probs[self.incompat_mx] -= np.sum(pw_probs)

        # pw_probs = np.triu(pw_probs, k=1)
        # pw_probs.retain_grad()  # debug: view the backward pass result

        return sdp_obj_value, pw_probs

    def forward(self, edge_weights, gold_clustering):
        # what all should be requires_grad = True?
        self.edge_weights = edge_weights.tocoo()
        self.gold_clustering = gold_clustering
        self.gold_clustering_matrix = cluster_labels_to_matrix(self.gold_clustering)
        # Since point_features.shape[0] == edge_weights.shape[0]
        self.num_points = edge_weights.shape[0]
        n = self.num_points + self.max_num_ecc

        # formulate SDP
        logging.info('Constructing optimization problem')
        W = csr_matrix((self.edge_weights.data, (self.edge_weights.row, self.edge_weights.col)), shape=(n, n))
        self.W_val = torch.tensor(W.todense(), requires_grad=True)

        self.X = cp.Variable((n, n), PSD=True)

        self.W = cp.Parameter((n, n))

        # build out constraint set
        constraints = [
                cp.diag(self.X) == np.ones((n,)),
                self.X[:self.num_points, :] >= 0,
        ]

        # create problem
        self.prob = cp.Problem(cp.Maximize(cp.trace(self.W @ self.X)), constraints)
        # Note: maximizing the trace is equivalent to maximizing the sum_E (w_uv * X_uv) objective
        # because W is upper-triangular and X is symmetric

        # Build the SDP cvxpylayer
        self.sdp_layer = CvxpyLayer(self.prob, parameters=[self.W], variables=[self.X])

        # Solve the SDP and return result
        # TODO: Zero grad parameters reqd?
        # params = [self.W_val]
        # for param in params:
        #     param.grad = None

        sdp_obj_value, pw_probs = self.build_and_solve_sdp()
        return pw_probs
