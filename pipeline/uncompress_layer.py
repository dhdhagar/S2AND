import torch
import math

class UncompressTransformLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.uncompressed_matrix = None

    def forward(self, compressed_matrix):
        device = compressed_matrix.get_device()
        # Calculate size of uncompressed matrix
        n = round(math.sqrt(2 * compressed_matrix.size(dim=0))) + 1
        # Convert the 1D pairwise-similarities list to nxn upper triangular matrix
        ind = torch.triu_indices(n, n, offset=1, device=device)
        self.uncompressed_matrix = (torch.sparse_coo_tensor(ind, compressed_matrix, [n, n])).to_dense()
        # Make symmetric
        self.uncompressed_matrix = self.uncompressed_matrix + torch.transpose(self.uncompressed_matrix, 0, 1) - torch.diag(self.uncompressed_matrix)
        self.uncompressed_matrix += torch.eye(n, device=device)
        if self.training:
            self.uncompressed_matrix.retain_grad()

        return self.uncompressed_matrix