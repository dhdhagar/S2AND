"""
Example usage: python e2e_debug/solve.py --data_fpath=e2e_debug/run_errors/1wt1e4ui.json --scs_silent --max_scaling --inspect_before_solving
"""
import json
import argparse
import cvxpy as cp
import logging
import numpy as np
import torch

from IPython import embed

from e2e_pipeline.hac_cut_layer import HACCutLayer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--data_fpath", type=str
        )
        self.add_argument(
            "--data_idx", type=int, default=0
        )
        self.add_argument(
            "--scs_max_sdp_iters", type=int, default=50000
        )
        self.add_argument(
            "--scs_silent", action="store_true",
        )
        self.add_argument(
            "--scs_eps", type=float, default=1e-3
        )
        self.add_argument(
            "--scs_scale", type=float, default=1e-1,
        )
        self.add_argument(
            "--scs_dont_normalize", action="store_true",
        )
        self.add_argument(
            "--scs_use_indirect", action="store_true",
        )
        self.add_argument(
            "--scs_dont_use_quad_obj", action="store_true",
        )
        self.add_argument(
            "--scs_alpha", type=float, default=1.5
        )
        self.add_argument(
            "--scs_log_csv_filename", type=str,
        )
        self.add_argument(
            "--max_scaling", action="store_true",
        )
        self.add_argument(
            "--interactive", action="store_true",
        )
        self.add_argument(
            "--inspect_before_solving", action="store_true",
        )


if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    logger.info("Script arguments:")
    logger.info(args.__dict__)

    # Read error file
    logger.info("Reading input data")
    if args.data_fpath.endswith('.pt'):
        _W_val = torch.load(args.data_fpath, map_location='cpu').numpy()
    else:
        with open(args.data_fpath, 'r') as fh:
            data = json.load(fh)
        assert len(data['errors']) > 0
        # Pick specific error instance to process
        error_data = data['errors'][args.data_idx]

        # Extract input data from the error instance
        _raw = np.array(error_data['model_call_args']['data'])
        _W_val = np.array(error_data['cvxpy_layer_args']['W_val'])

    # Construct cvxpy problem
    logger.info('Constructing optimization problem')
    # edge_weights = _W_val.tocoo()
    n = _W_val.shape[0]
    W = _W_val
    # W = csr_matrix((edge_weights.data, (edge_weights.row, edge_weights.col)), shape=(n, n))
    X = cp.Variable((n, n), PSD=True)
    # Build constraint set
    constraints = [
        cp.diag(X) == np.ones((n,)),
        X[:n, :] >= 0,
        # X[:n, :] <= 1
    ]

    # Setup HAC Cut
    hac_cut = HACCutLayer()
    hac_cut.eval()

    sdp_obj_value = float('inf')
    result_idxs, results_X, results_clustering = [], [], []
    no_solution_scaling_factors = []
    for i in range(0, 10):  # n
        # Skipping 1; no scaling leads to non-convergence (infinite objective value)
        if i == 0:
            scaling_factor = np.max(np.abs(W))
        else:
            scaling_factor = i
            if args.max_scaling:
                continue
        logger.info(f'Scaling factor={scaling_factor}')
        # Create problem
        W_scaled = W / scaling_factor
        problem = cp.Problem(cp.Maximize(cp.trace(W_scaled @ X)), constraints)

        if args.inspect_before_solving:
            embed()

        # Solve problem
        sdp_obj_value = problem.solve(
            solver=cp.SCS,
            verbose=not args.scs_silent,
            max_iters=args.scs_max_sdp_iters,
            eps=args.scs_eps,
            normalize=not args.scs_dont_normalize,
            alpha=args.scs_alpha,
            scale=args.scs_scale,
            use_indirect=args.scs_use_indirect,
            # use_quad_obj=not args.scs_dont_use_quad_obj
        )
        logger.info(f"@scaling={scaling_factor}, objective value = {sdp_obj_value}, norm={np.linalg.norm(W_scaled)}")
        if sdp_obj_value != float('inf'):
            result_idxs.append(i)
            results_X.append(X.value)
            # Find clustering solution
            hac_cut.get_rounded_solution(torch.tensor(X.value), torch.tensor(W_scaled))
            results_clustering.append(hac_cut.cluster_labels.numpy())
        else:
            no_solution_scaling_factors.append(scaling_factor)
    logger.info(f"Solution not found = {len(no_solution_scaling_factors)}")
    logger.info(f"Solution found = {len(results_X)}")

    # logger.info("Same clustering:")
    # for i in range(len(results_clustering)-1):
    #     logger.info(np.array_equal(results_clustering[i], results_clustering[i + 1]))
    # logger.info(f"Solution found with scaling factor = {scaling_factor}")
    # if args.interactive and sdp_obj_value == float('inf'):
    #     embed()

    if args.interactive:
        embed()
