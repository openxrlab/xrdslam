# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn


class FeatureGrid(nn.Module):
    def __init__(self, xyz_len, grid_len, c_dim, std=0.01):
        super(FeatureGrid, self).__init__()

        val_shape = list(map(int, (xyz_len / grid_len).tolist()))
        val_shape[0], val_shape[2] = val_shape[2], val_shape[0]
        val_shape = [1, c_dim, *val_shape]
        val = torch.zeros(val_shape).normal_(mean=0, std=std)

        mask = np.ones((val_shape[2:])[::-1]).astype(bool)
        mask = torch.from_numpy(mask).permute(
            2, 1, 0).unsqueeze(0).unsqueeze(0).repeat(1, val_shape[1], 1, 1, 1)

        self.val = nn.Parameter(val, requires_grad=True)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def set_mask(self, new_mask):
        self.mask.data.copy_(new_mask)

    def val_mask(self):
        masked_val = self.val * self.mask
        return masked_val
