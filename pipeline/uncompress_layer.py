import torch
import math

class UncompressTransformLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.uncompressed_matrix = torch.nn.Parameter()

    def forward(self, compressed_matrix):
        # Calculate size of uncompressed matrix
        n = round(math.sqrt(2 * compressed_matrix.size(dim=0))) + 1
        # Convert the 1D pairwise-similarities list to nxn upper triangular matrix
        ind = torch.triu_indices(n, n, offset=1)
        output = (torch.sparse_coo_tensor(ind, compressed_matrix, [n, n, 1])).to_dense()
        self.uncompressed_matrix = torch.reshape(output, (n, n))
        self.uncompressed_matrix.retain_grad()

        return self.uncompressed_matrix