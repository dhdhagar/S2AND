import torch
from ecc.ecc_layer import cluster_labels_to_matrix
from utils.trellis_helper_fns import build_trellis, cut_trellis


class TrellisCutLayer(torch.nn.Module):
    """
    Takes the SDP solution as input and executes the trellis-cut rounding algorithm in the forward pass
    Executes a straight-through estimator in the backward pass
    """
    def __init__(self):
        super().__init__()

    def get_rounded_solution(self, edge_weights, pw_probs, only_avg_hac=False):
        t = build_trellis(pw_probs.detach().numpy(), only_avg_hac=only_avg_hac)
        pred_clustering, cut_obj_value, num_ecc_satisfied = cut_trellis(t, edge_weights.detach().tocoo())

        self.cut_obj_value = cut_obj_value
        self.num_ecc_satisfied = num_ecc_satisfied
        self.pred_clustering = pred_clustering

        # Return an NxN matrix of 0s and 1s based on the predicted clustering
        round_mat = cluster_labels_to_matrix(self.pred_clustering)

        return round_mat

    def forward(self, X, only_avg_hac=False):
        return X + (self.get_rounded_solution(X, only_avg_hac) - X).detach()