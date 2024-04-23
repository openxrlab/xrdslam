# This file is from DPVO,
# licensed under the MIT License.

import altcorr
import fastba
import torch
import torch.nn as nn
import torch.nn.functional as F

from slam.model_components.blocks_dpvo import (GatedResidual, GradientClip,
                                               SoftAgg)
from slam.model_components.extractor_dpvo import BasicEncoder4
from slam.model_components.utils_dpvo import coords_grid_with_index, pyramidify

DIM = 384


class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True),
                                nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True),
                                nn.Linear(DIM, DIM))

        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2 * 49 * p * p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2),
                               GradientClip())

        self.w = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2),
                               GradientClip(), nn.Sigmoid())

    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """update operator."""
        net = net + inp + self.corr(corr)
        net = self.norm(net)
        ix, jx = fastba.neighbors(kk, jj)

        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)
        net = net + self.c1(mask_ix * net[:, ix])
        net = net + self.c2(mask_jx * net[:, jx])
        net = net + self.agg_kk(net, kk)

        net = net + self.agg_ij(net, ii * 12345 + jj)
        net = self.gru(net)
        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[..., :-1, 1:] - gray[..., :-1, :-1]
        dy = gray[..., 1:, :-1] - gray[..., :-1, :-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self,
                images,
                patches_per_image=80,
                disps=None,
                gradient_bias=False,
                return_color=False):
        """extract patches from input images."""
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        P = self.patch_size

        # bias patch selection towards regions with high gradient
        if gradient_bias:
            g = self.__image_gradient(images)
            x = torch.randint(1,
                              w - 1,
                              size=[n, 3 * patches_per_image],
                              device='cuda')
            y = torch.randint(1,
                              h - 1,
                              size=[n, 3 * patches_per_image],
                              device='cuda')

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0, :, None], coords,
                                 0).view(n, 3 * patches_per_image)

            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        else:
            x = torch.randint(1,
                              w - 1,
                              size=[n, patches_per_image],
                              device='cuda')
            y = torch.randint(1,
                              h - 1,
                              size=[n, patches_per_image],
                              device='cuda')

        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P // 2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4 * (coords + 0.5),
                                   0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device='cuda')

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords,
                                   P // 2).view(b, -1, 3, P, P)

        index = torch.arange(n, device='cuda').view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1, 4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4

    # @torch.cuda.amp.autocast(enabled=False)
    # TODO: add forward()
    # def forward(self,
    #             images,
    #             poses,
    #             disps,
    #             intrinsics,
    #             STEPS=12,
    #             P=1,
    #             structure_only=False):
