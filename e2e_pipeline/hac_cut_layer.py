import torch


class HACCutLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.round_matrix = None
        self.cluster_labels = None
        self.parents = None
        self.objective_value = None

    """
    Takes fractional SDP output as input, and simultaneously builds & cuts avg. HAC tree to get rounded solution.
    Executes straight-through estimator as the backward pass.
    """
    def get_rounded_solution(self, X, weights, _MAX_DIST=1000, use_similarities=True, max_similarity=1, verbose=False):
        """
        X is a symmetric NxN matrix of fractional, decision values with a 1-diagonal (output from the SDP layer)
        weights is an NxN upper-triangular (shift 1) matrix of edge weights
        Return a symmetric NxN matrix of 0-1 decision values with a 1-diagonal
        """

        # Initialization
        device = X.device
        D = X.size(1)
        parents = torch.arange(D + (D - 1))
        parent_to_idx = torch.arange(D + (D - 1))
        idx_to_parent = torch.arange(D)
        cluster_sizes = torch.ones(D + (D - 1))

        energy = torch.zeros(D + (D - 1), device=device)
        clustering = torch.zeros((D + (D - 1), D))
        clustering[torch.arange(D), torch.arange(D)] = torch.arange(1, D + 1, dtype=clustering.dtype)
        round_matrix = torch.eye(D, device=device)

        # Take the upper triangular and mask the other values with a large number
        _MAX_DIST = max(torch.abs(X)) * _MAX_DIST
        Y = _MAX_DIST * torch.ones(D, D, device=device).tril() + (max_similarity - X if use_similarities else X).triu(1)
        # Compute the dissimilarity minima per row
        values, indices = torch.min(Y, dim=1)

        if verbose:
            print('Initialization:')
            print('Y:', Y)
            print('\tparents:', parents)
            print('\tparent_to_idx:', parent_to_idx)
            print('\tidx_to_parent:', idx_to_parent)
            print('\tminima (values):', values)
            print('\tminima (indices):', indices)
            print('\tcluster_sizes:', cluster_sizes)
            print()

        ####################################

        max_node = D - 1
        if verbose:
            print('Starting algorithm:')
        for i in range(D - 1):
            max_node += 1
            min_minima_idx = torch.argmin(values).item()

            # Merge the index of the minimum value of minimums across rows with the index of the minimum value in its row
            merge_idx_1 = min_minima_idx
            merge_idx_2 = indices[merge_idx_1].item()

            # Find highest-altitude clusters corresponding to the merge indices
            parent_1 = idx_to_parent[parent_to_idx[idx_to_parent[merge_idx_1]]].item()
            parent_2 = idx_to_parent[parent_to_idx[idx_to_parent[merge_idx_2]]].item()

            if verbose:
                print(f'    #{i} Merging:', (merge_idx_1, merge_idx_2), 'i.e.', (parent_1, parent_2), '=>', max_node)

            # Add parent for the clusters being merged
            parents[parent_1] = max_node
            parents[parent_2] = max_node

            # Update mappings
            idx_to_parent[merge_idx_1] = max_node
            parent_to_idx[max_node] = merge_idx_1

            # Update the matrix with merged values for cluster similarities
            max_dist_mask = Y == _MAX_DIST
            new_cluster_size = cluster_sizes[parent_1] + cluster_sizes[parent_2]
            cluster_sizes[max_node] = new_cluster_size
            new_merge_idx_1_values = (torch.min(Y[merge_idx_1, :], Y[:, merge_idx_1]) * cluster_sizes[parent_1] + \
                                      torch.min(Y[:, merge_idx_2], Y[merge_idx_2, :]) * cluster_sizes[parent_2]) / \
                                     new_cluster_size
            Y[:, merge_idx_1] = new_merge_idx_1_values
            Y[merge_idx_1, :] = new_merge_idx_1_values
            Y[max_dist_mask] = _MAX_DIST
            Y[:, merge_idx_2] = _MAX_DIST
            Y[merge_idx_2, :] = _MAX_DIST

            # Update nearest neighbour trackers
            values[merge_idx_2] = _MAX_DIST

            max_dist_mask = values == _MAX_DIST
            values, indices = torch.min(Y, dim=1)
            values[max_dist_mask] = _MAX_DIST

            # Energy calculations
            clustering[max_node] = clustering[parent_1] + clustering[parent_2]
            leaf_indices = torch.where(clustering[max_node])[0]
            leaf_edges = torch.meshgrid(leaf_indices, leaf_indices, indexing='ij')
            energy[max_node] = energy[parent_1] + energy[parent_2]
            merge_energy = torch.sum(weights[leaf_edges])
            if merge_energy >= energy[max_node]:
                energy[max_node] = merge_energy
                clustering[max_node][clustering[max_node] > 0] = max_node
                round_matrix[leaf_edges] = 1
            if verbose:
                print('Y:', Y)
                print('\tminima (values):', values)
                print('\tminima (indices):', indices)
                print('\tparents:', parents)
                print('\tparent_to_idx:', parent_to_idx)
                print('\tidx_to_parent:', idx_to_parent)
                print('\tcluster_sizes:', cluster_sizes)
                print('\tclustering (current):', clustering[max_node])
                print('round_matrix:')
                print(round_matrix)
                print()

        self.round_matrix = round_matrix
        self.cluster_labels = clustering[-1]
        self.parents = parents
        with torch.no_grad():
            objective_matrix = weights * torch.triu(round_matrix, diagonal=1)
            self.objective_value = (energy[max_node] - torch.sum(objective_matrix[objective_matrix < 0])).item()  # MA
        return self.round_matrix

    def forward(self, X, W, use_similarities=True, return_triu=False):
        solution = X + (self.get_rounded_solution(X, W,
                                                  use_similarities=use_similarities,
                                                  max_similarity=max(X)) - X).detach()
        if return_triu:
            triu_indices = torch.triu_indices(len(solution), len(solution), offset=1)
            return solution[triu_indices[0], triu_indices[1]]
        return solution
