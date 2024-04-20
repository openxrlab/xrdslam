import torch


class FeatureGrid():
    def __init__(self, xyz_len, grid_len, c_dim, std=0.01):
        super(FeatureGrid, self).__init__()

        val_shape = list(map(int, (xyz_len / grid_len).tolist()))
        val_shape[0], val_shape[2] = val_shape[2], val_shape[0]
        val_shape = [1, c_dim, *val_shape]
        val = torch.zeros(val_shape).normal_(mean=0, std=std)

        self.val = val
