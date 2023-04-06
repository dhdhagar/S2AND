import torch

import higra as hg
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

import logging
from IPython import embed

from e2e_scripts.train_utils import compute_b3_f1

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class HACInference:
    """
    HAC inference-only.
    """

    def __init__(self, threshold=0.5, tune_metric="b3_f1"):
        super().__init__()
        self.cut_threshold = threshold
        self.tune_metric = tune_metric  # "b3_f1" or "vmeasure"  # TODO: Take as parameter from main script

    def set_threshold(self, threshold):
        self.cut_threshold = threshold

    def tune_threshold(self, model, dataloader, device, n_trials=1000):
        device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_gold = []
        blockwise_trees = []
        all_dists = []
        n_features = dataloader.dataset[0][0].shape[1]
        for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Tuning threshold on dev')):
            data, _, cluster_ids = batch
            data = data.reshape(-1, n_features).float()
            if data.shape[0] == 0:
                # Only one signature in block; skip for tuning
                continue
            all_gold += list(np.reshape(cluster_ids, (len(cluster_ids),)))
            block_size = len(cluster_ids)

            # Forward pass through the e2e model
            data = data.to(device)
            edge_weights = model(data, N=len(cluster_ids), warmstart=True)
            tree_and_alts, dists = self.cluster(edge_weights, block_size, return_tree=True)
            blockwise_trees.append(tree_and_alts)
            all_dists.append(dists)

        # Compute thresholds to try
        all_dists = np.concatenate((all_dists)).reshape(-1, 1)
        logger.info("Computing thresholds to try...")
        kmeans = KMeans(n_clusters=n_trials, random_state=17)
        thresholds = np.sort(np.unique(kmeans.fit(all_dists).cluster_centers_.flatten()))

        # Cluster on each threshold and compute metric to pick best threshold
        best_threshold = thresholds[0]
        best_dev_metric = -1
        for _thresh in tqdm(thresholds, desc="Finding best cut threshold"):
            all_pred = []
            max_pred_id = -1  # In each iter, add to all blockwise predicted IDs to distinguish from previous blocks
            for (_hac, _hac_alts) in blockwise_trees:
                _cut_labels = self.cut_tree(_hac, _hac_alts, _thresh)
                pred_cluster_ids = _cut_labels + (max_pred_id + 1)
                max_pred_id = max(pred_cluster_ids)
                all_pred += list(pred_cluster_ids)
            b3_f1 = compute_b3_f1(all_gold, all_pred)[2]
            if b3_f1 > best_dev_metric:
                best_dev_metric = b3_f1
                best_threshold = _thresh
        logger.info(f"Best dev b3_f1 of {best_dev_metric} found at {best_threshold}")

        self.cut_threshold = best_threshold

    def cut_tree(self, tree, alts, thresh):
        _cut_explorer = hg.HorizontalCutExplorer(tree, alts)
        _cut = _cut_explorer.horizontal_cut_from_altitude(thresh)
        _cut_labels = _cut.labelisation_leaves(tree)
        return _cut_labels

    def cluster(self, edge_weights, N, min_id=0, threshold=None, return_tree=False, verbose=False):
        """
        edge_weights: N(N-1)/2 length array of weights from the upper-triangular (shift 1) pairwise weight matrix
        """
        _thresh = threshold if threshold is not None else self.cut_threshold
        _data = []
        _g = hg.UndirectedGraph(N)
        r, c = np.triu_indices(N, k=1)
        _dists = 1. - torch.sigmoid(edge_weights).cpu().numpy().reshape(N*(N-1)//2)
        _g.add_edges(r, c)
        _hac, _hac_alts = hg.binary_partition_tree_average_linkage(_g, _dists)

        if return_tree:
            return (_hac, _hac_alts), _dists

        _cut_labels = self.cut_tree(_hac, _hac_alts, _thresh)

        return list(_cut_labels + min_id)

    def __call__(self, *args, **kwargs):
        return self.cluster(args[0], args[1], min_id=kwargs['min_id'])
