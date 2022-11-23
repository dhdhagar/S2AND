from ecc.trellis import Trellis
import numpy as np
import numba as nb
from scipy.sparse import csr_matrix, coo_matrix

def build_trellis(pw_probs: np.ndarray, only_avg_hac: bool = False):
    t = Trellis(adj_mx=pw_probs)
    t.fit(only_avg_hac=only_avg_hac)
    return t


def cut_trellis(t: Trellis, edge_weights: coo_matrix):
    membership_indptr = t.leaves_indptr
    membership_indices = t.leaves_indices
    membership_data = get_membership_data(membership_indptr,
                                          membership_indices)
    obj_vals = np.zeros((t.num_nodes,))
    num_ecc_sat = np.zeros((t.num_nodes,))

    for node in t.internal_nodes_topo_ordered():
        node_start = membership_indptr[node]
        node_end = membership_indptr[node + 1]
        leaves = membership_indices[node_start:node_end]
        # if num_ecc > 0: # Condition is never true for 0 there-exists constraints
        #     num_ecc_sat[node] = get_num_ecc_sat(
        #         leaves, num_points)
        obj_vals[node] = get_intra_cluster_energy(edge_weights, leaves)
        for lchild, rchild in t.get_child_pairs_iter(node):
            cpair_num_ecc_sat = num_ecc_sat[lchild] + num_ecc_sat[rchild]
            cpair_obj_val = obj_vals[lchild] + obj_vals[rchild]
            if (num_ecc_sat[node] < cpair_num_ecc_sat
                    or (num_ecc_sat[node] == cpair_num_ecc_sat
                        and obj_vals[node] < cpair_obj_val)):
                num_ecc_sat[node] = cpair_num_ecc_sat
                obj_vals[node] = cpair_obj_val
                lchild_start = membership_indptr[lchild]
                lchild_end = membership_indptr[lchild + 1]
                rchild_start = membership_indptr[rchild]
                rchild_end = membership_indptr[rchild + 1]
                merge_memberships(
                    membership_indices[lchild_start:lchild_end],
                    membership_data[lchild_start:lchild_end],
                    membership_indices[rchild_start:rchild_end],
                    membership_data[rchild_start:rchild_end],
                    membership_indices[node_start:node_end],
                    membership_data[node_start:node_end],
                )

    # The value of `node` is the root since we iterate over the trellis
    # nodes in topological order bottom up. Moreover, the values of
    # `node_start` and `node_end` also correspond to the root of the
    # trellis.
    best_clustering = membership_data[node_start:node_end]
    # if num_ecc > 0: # Condition is never true for 0 there-exists constraints
    #     best_clustering = best_clustering[:-num_ecc]

    return best_clustering, obj_vals[node], num_ecc_sat[node]

@nb.njit(parallel=True)
def get_membership_data(indptr: np.ndarray,
                        indices: np.ndarray):
    data = np.empty(indices.shape, dtype=np.int64)
    for i in nb.prange(indptr.size-1):
        for j in nb.prange(indptr[i], indptr[i+1]):
            data[j] = i
    return data

def get_intra_cluster_energy(edge_weights: coo_matrix,
                             leaves: np.ndarray):
    row_mask = np.isin(edge_weights.row, leaves)
    col_mask = np.isin(edge_weights.col, leaves)
    data_mask = row_mask & col_mask
    return np.sum(edge_weights.data[data_mask])

@nb.njit
def merge_memberships(lchild_indices: np.ndarray,
                      lchild_data: np.ndarray,
                      rchild_indices: np.ndarray,
                      rchild_data: np.ndarray,
                      parent_indices: np.ndarray,
                      parent_data: np.ndarray):
        assert lchild_indices.size == lchild_data.size
        assert rchild_indices.size == rchild_data.size
        assert parent_indices.size == parent_data.size
        assert (lchild_data.size + rchild_data.size) == parent_data.size
        lchild_ptr = 0
        rchild_ptr = 0
        for i in range(parent_data.size):
            if (rchild_ptr == rchild_indices.size or
                    (lchild_ptr < lchild_indices.size and
                     lchild_indices[lchild_ptr] < rchild_indices[rchild_ptr])):
                assert parent_indices[i] == lchild_indices[lchild_ptr]
                parent_data[i] = lchild_data[lchild_ptr]
                lchild_ptr += 1
            else:
                assert parent_indices[i] == rchild_indices[rchild_ptr]
                assert (lchild_ptr == lchild_indices.size
                        or lchild_indices[lchild_ptr] != rchild_indices[rchild_ptr])
                parent_data[i] = rchild_data[rchild_ptr]
                rchild_ptr += 1