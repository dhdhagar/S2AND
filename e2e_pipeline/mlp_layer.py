import torch
import torch.nn as nn
from neumiss import NeuMissBlock, NeuMissDEQBlock


class MLPLayer(torch.nn.Module):
    def __init__(self, n_features, neumiss_depth=10, dropout_p=0.1, dropout_only_once=False,
                 add_neumiss=True, neumiss_deq=False, hidden_dim=1024, n_hidden_layers=1, add_batchnorm=True,
                 activation="leaky_relu", negative_slope=0.01, hidden_config=None):
        super().__init__()
        neumiss_layer = NeuMissDEQBlock if neumiss_deq else NeuMissBlock
        neumiss_args = {"n_features": n_features}
        if not neumiss_deq:
            neumiss_args.update({"depth": neumiss_depth})

        activation_fn = nn.ReLU if activation == "relu" else nn.LeakyReLU
        activation_args = {}
        if activation == "leaky_relu":
            activation_args.update({"negative_slope": negative_slope})

        if hidden_config is not None:
            network = [neumiss_layer(**neumiss_args)] if add_neumiss else []
            in_dim = n_features
            for out_dim in hidden_config:
                network += [nn.Linear(in_dim, out_dim)] + \
                           ([activation_fn(**activation_args)]) + \
                           ([nn.BatchNorm1d(out_dim)] if add_batchnorm else []) + \
                           ([nn.Dropout(p=dropout_p)] if not dropout_only_once else [])
                in_dim = out_dim
            network += [nn.Linear(in_dim, 1)]
            self.mlp_model = nn.Sequential(*network)
        else:
            if n_hidden_layers < 1:
                raise ValueError("NeuMissVanilla requires a minimum of one hidden layer.")
            self.mlp_model = nn.Sequential(
                *(([neumiss_layer(**neumiss_args)] if add_neumiss else []) +
                  [nn.Linear(n_features, hidden_dim)] +
                  ([activation_fn(**activation_args)] + ([nn.BatchNorm1d(hidden_dim)] if add_batchnorm else []) +
                   ([nn.Dropout(p=dropout_p)] if not dropout_only_once else []) +
                   [nn.Linear(hidden_dim, hidden_dim)]) * (n_hidden_layers - 1) +
                  ([activation_fn(**activation_args)] + [nn.BatchNorm1d(hidden_dim)] if add_batchnorm else []) +
                  [nn.Dropout(p=dropout_p), nn.Linear(hidden_dim, 1)])
            )

        self.last_output = None  # Stores the most recent result of the forward call

    def forward(self, x):
        output = self.mlp_model(x)
        self.last_output = output.detach()
        return output
