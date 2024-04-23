from dataclasses import dataclass, field
from typing import Type

import altcorr
import fastba
import lietorch
import numpy as np
import torch
import torch.nn.functional as F
from lietorch import SE3

from slam.algorithms.base_algorithm import Algorithm, AlgorithmConfig
from slam.common.camera import Camera
from slam.model_components import projective_ops_dpvo as pops
from slam.model_components.utils_dpvo import flatmeshgrid

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device='cuda')


@dataclass
class DPVOConfig(AlgorithmConfig):
    """DPVO  Config."""
    _target: Type = field(default_factory=lambda: DPVO)
    patch_per_frame: int = 96
    patch_lifetime: int = 13
    init_frame_num: int = 8
    gradient_bias: bool = False
    mixed_precision: bool = False
    optimization_window: int = 10
    keyframe_index: int = 4
    keyframe_thresh: float = 15.0
    removal_window: int = 22
    motion_model: str = 'DAMPED_LINEAR'
    motion_damping: float = 0.5
    buffer_size: int = 2048
    mem: int = 32


class DPVO(Algorithm):

    config: DPVOConfig

    def __init__(self, config: DPVOConfig, camera: Camera,
                 device: str) -> None:
        super().__init__(config, camera, device)
        self.model = self.config.model.setup(camera=camera, bounding_box=None)
        self.model.to(device)
        self.bundle_adjust = False

        self.n = 0
        self.m = 0
        self.M = self.config.patch_per_frame
        self.N = self.config.buffer_size

        self.intrinsics_ori = torch.from_numpy(
            np.array([camera.fx, camera.fy, camera.cx, camera.cy])).cuda()

        # state attributes #
        self.counter = 0
        self.tlist = []

        # steal network attributes
        self.DIM = self.model.network.DIM
        self.RES = self.model.network.RES
        self.P = self.model.network.P

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device='cuda')
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device='cuda')
        self.patches_ = torch.zeros(self.N,
                                    self.M,
                                    3,
                                    self.P,
                                    self.P,
                                    dtype=torch.float,
                                    device='cuda')
        self.intrinsics_ = torch.zeros(self.N,
                                       4,
                                       dtype=torch.float,
                                       device='cuda')

        self.points_ = torch.zeros(self.N * self.M,
                                   3,
                                   dtype=torch.float,
                                   device='cuda')
        self.colors_ = torch.zeros(self.N,
                                   self.M,
                                   3,
                                   dtype=torch.uint8,
                                   device='cuda')

        self.index_ = torch.zeros(self.N,
                                  self.M,
                                  dtype=torch.long,
                                  device='cuda')
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device='cuda')

        # network attributes #
        self.mem = self.config.mem
        ht = camera.height // self.RES
        wd = camera.width // self.RES

        if self.config.mixed_precision:
            self.kwargs = kwargs = {'device': 'cuda', 'dtype': torch.half}
        else:
            self.kwargs = kwargs = {'device': 'cuda', 'dtype': torch.float}

        self.imap_ = torch.zeros(self.mem, self.M, self.DIM, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P,
                                 **kwargs)

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, self.DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device='cuda')
        self.jj = torch.as_tensor([], dtype=torch.long, device='cuda')
        self.kk = torch.as_tensor([], dtype=torch.long, device='cuda')

        # initialize poses to identity matrix
        self.poses_[:, 6] = 1.0
        # store relative poses for removed frames
        self.delta = {}

    def render_img(self, c2w, gt_depth=None, idx=None):
        return None, None

    def add_keyframe(self, keyframe):
        pass

    def do_mapping(self, cur_frame):
        pass

    def get_cloud(self, c2w_np, gt_depth_np):
        with self.lock and torch.no_grad():
            cloud_pos = self.points_[:self.m]
            cloud_rgb = self.colors_.view(-1, 3)[:self.m].float() / 255.0
            cloud_rgb = torch.clamp(cloud_rgb, 0, 1)

            # filter the outlier depth
            median_depth = torch.median(cloud_pos[:, 2])
            mask = (cloud_pos[:, 2] <= median_depth * 10) & (cloud_pos[:, 2] >
                                                             0)

            return cloud_pos[mask].detach().cpu().numpy(
            ), cloud_rgb[mask].detach().cpu().numpy()

    def do_tracking(self, cur_frame):
        if (self.n + 1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase'
                            f'it using "--buffer {self.N*2}"')

        self.detect_patches(cur_frame)

        self.counter += 1
        if self.n > 0 and not self.is_initialized():
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M
        self.append_factors(*self.edges_forw())
        self.append_factors(*self.edges_back())

        if self.n == self.config.init_frame_num and not self.is_initialized():
            self.set_initialized()
            for itr in range(12):
                self.update()

            # update the first 7 frames' pose
            poses, fids = self.get_all_poses()
            for i in range(self.counter - 1):
                self.update_framepose(fids[i], torch.from_numpy(poses[i]))

        elif self.is_initialized():
            self.update()
            self.keyframe()

        torch.cuda.empty_cache()

        return SE3(
            self.poses_[self.n -
                        1]).inv().matrix().clone().detach().cpu().numpy()

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N * self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def corr(self, coords, indices=None):
        """This function is from DPVO, licensed under the MIT License."""
        """local correlation volume."""
        ii, jj = indices if indices is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1,
                             3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1,
                             3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indices=None):
        """This function is from DPVO, licensed under the MIT License."""
        """reproject patch k from i -> j."""
        (ii, jj, kk) = indices if indices is not None else (self.ii, self.jj,
                                                            self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics,
                                ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        """This function is from DPVO, licensed under the MIT License."""
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        """This function is from DPVO, licensed under the MIT License."""
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:, ~m]

    def motion_probe(self):
        """This function is from DPVO, licensed under the MIT License."""
        """kinda hacky way to ensure enough motion for initialization."""
        kk = torch.arange(self.m - self.M, self.m, device='cuda')
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indices=(ii, jj, kk))

        with autocast(enabled=self.config.mixed_precision) and torch.no_grad():
            corr = self.corr(coords, indices=(kk, jj))
            ctx = self.imap[:, kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.model.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        """This function is from DPVO, licensed under the MIT License."""
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses),
                             self.patches,
                             self.intrinsics,
                             ii,
                             jj,
                             kk,
                             beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        """This function is from DPVO, licensed under the MIT License."""
        i = self.n - self.config.keyframe_index - 1
        j = self.n - self.config.keyframe_index + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.config.keyframe_thresh:
            k = self.n - self.config.keyframe_index

            t0 = self.tstamps_[k - 1].item()
            t1 = self.tstamps_[k].item()
            dP = SE3(self.poses_[k]) * SE3(self.poses_[k - 1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)
            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n - 1):
                self.tstamps_[i] = self.tstamps_[i + 1]
                self.poses_[i] = self.poses_[i + 1]
                self.colors_[i] = self.colors_[i + 1]
                self.patches_[i] = self.patches_[i + 1]
                self.intrinsics_[i] = self.intrinsics_[i + 1]

                self.imap_[i % self.mem] = self.imap_[(i + 1) % self.mem]
                self.gmap_[i % self.mem] = self.gmap_[(i + 1) % self.mem]
                self.fmap1_[0, i % self.mem] = self.fmap1_[0,
                                                           (i + 1) % self.mem]
                self.fmap2_[0, i % self.mem] = self.fmap2_[0,
                                                           (i + 1) % self.mem]

            self.n -= 1
            self.m -= self.M

        to_remove = self.ix[self.kk] < self.n - self.config.removal_window
        self.remove_factors(to_remove)

    def update(self):
        """This function is from DPVO, licensed under the MIT License."""
        coords = self.reproject()
        with autocast(enabled=True) and torch.no_grad():
            corr = self.corr(coords)
            ctx = self.imap[:, self.kk % (self.M * self.mem)]
            self.net, (delta, weight,
                       _) = self.model.network.update(self.net, ctx, corr,
                                                      None, self.ii, self.jj,
                                                      self.kk)
        lmbda = torch.as_tensor([1e-4], device='cuda')
        weight = weight.float()
        target = coords[..., self.P // 2, self.P // 2] + delta.float()
        t0 = self.n - self.config.optimization_window if self.is_initialized(
        ) else 1
        t0 = max(t0, 1)

        try:
            fastba.bundle_adjust_dpvo(self.poses, self.patches,
                                      self.intrinsics, target, weight, lmbda,
                                      self.ii, self.jj, self.kk, t0, self.n, 2)

        except Exception as e:
            print('Warning: bundle_adjust_dpvo failed with exception:', e)
            import traceback
            traceback.print_exc()

        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m],
                                  self.intrinsics, self.ix[:self.m])
        points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
        with self.lock:
            self.points_[:len(points)] = points[:]

    def edges_forw(self):
        """This function is from DPVO, licensed under the MIT License."""
        r = self.config.patch_lifetime
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(torch.arange(t0, t1, device='cuda'),
                            torch.arange(self.n - 1, self.n, device='cuda'),
                            indexing='ij')

    def edges_back(self):
        """This function is from DPVO, licensed under the MIT License."""
        r = self.config.patch_lifetime
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device='cuda'),
                            torch.arange(max(self.n - r, 0),
                                         self.n,
                                         device='cuda'),
                            indexing='ij')

    def get_pose(self, t):
        """This function is from DPVO, licensed under the MIT License."""
        if t in self.traj:
            return SE3(self.traj[t])
        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def get_all_poses(self):
        """This function is from DPVO, licensed under the MIT License."""
        """interpolate missing poses."""
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]
        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().matrix().data.cpu().numpy()
        tstamps = np.array(self.tlist)
        return poses, tstamps

    def detect_patches(self, cur_frame):
        image = torch.from_numpy(cur_frame.rgb).permute(2, 0, 1).to(
            self.device)  # [C, H, W]
        image = 2 * (image[None, None]) - 0.5
        image = image[:self.camera.height -
                      self.camera.height % 16, :self.camera.width -
                      self.camera.width % 16]
        with autocast(enabled=self.config.mixed_precision) and torch.no_grad():
            fmap, gmap, imap, patches, _, clr = self.model.network.patchify(
                images=image,
                patches_per_image=self.config.patch_per_frame,
                gradient_bias=self.config.gradient_bias,
                return_color=True)

        # update state attributes #
        self.tlist.append(cur_frame.fid)
        self.intrinsics_[self.n] = self.intrinsics_ori / self.RES
        self.tstamps_[self.n] = self.counter

        # color info for visualization
        clr = (clr + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.config.motion_model == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n - 1])
                P2 = SE3(self.poses_[self.n - 2])

                xi = self.config.motion_damping * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n - 1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if self.is_initialized():
            s = torch.median(self.patches_[self.n - 3:self.n, :, 2])
            patches[:, :, 2] = s

        self.patches_[self.n] = patches

        # update network attributes #
        self.imap_[self.n % self.mem].fill_(0)
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem].fill_(0)
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem].fill_(0)
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem].fill_(0)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)
