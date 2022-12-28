import torch
import torch.nn as nn
#from utils.convert_lgbm_to_torch import convert_pretrained_model


class MLPLayer(torch.nn.Module):
    def __init__(self, n_features, use_hb=False, dropout_p=0.1,
                 hidden_dim=1024, n_hidden_layers=1, add_batchnorm=True,
                 activation="leaky_relu", negative_slope=0.01, hidden_config=None):
        super().__init__()
        self.use_hb = use_hb
        if not self.use_hb:
            #self.mlp_model = convert_pretrained_model(dropout_p)
        #else:
            activation_fn = nn.ReLU if activation == "relu" else nn.LeakyReLU
            activation_args = {}
            if activation == "leaky_relu":
                activation_args.update({"negative_slope": negative_slope})

            if hidden_config is not None:
                network = []
                in_dim = n_features
                for out_dim in hidden_config:
                    network += [nn.Linear(in_dim, out_dim)] + \
                               ([activation_fn(**activation_args)]) + \
                               ([nn.BatchNorm1d(out_dim)] if add_batchnorm else []) + [nn.Dropout(p=dropout_p)]
                    in_dim = out_dim
                network += [nn.Linear(in_dim, 1)]
                self.mlp_model = nn.Sequential(*network)
            else:
                if n_hidden_layers < 1:
                    raise ValueError("NeuMissVanilla requires a minimum of one hidden layer.")
                self.mlp_model = nn.Sequential(
                    *(([]) +
                      [nn.Linear(n_features, hidden_dim)] +
                      ([activation_fn(**activation_args)] + ([nn.BatchNorm1d(hidden_dim)] if add_batchnorm else []) +
                       [nn.Dropout(p=dropout_p),
                        nn.Linear(hidden_dim, hidden_dim)]) * (n_hidden_layers - 1) +
                      ([activation_fn(**activation_args)] + [nn.BatchNorm1d(hidden_dim)] if add_batchnorm else []) +
                                                                                 [nn.Dropout(p=dropout_p),
                                                                                 nn.Linear(hidden_dim, 1)])
                )

    def forward(self, x):
        y = self.mlp_model(x)
        self.edge_weights = y
        if(self.use_hb):
            # Since output is from hummingbird model, need to extract only the probabilities of class label 1
            self.edge_weights = y[1][:, 1:]
        if self.training:
            self.edge_weights.retain_grad()
        return self.edge_weights