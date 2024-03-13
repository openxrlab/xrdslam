from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch.nn import Parameter

from deepslam.common.camera import Camera
from deepslam.model_components.gaussian_cloud_splaTAM import GaussianCloud
from deepslam.model_components.slam_external_splatam import calc_ssim
from deepslam.model_components.slam_helpers_splatam import l1_loss_v1
from deepslam.models.base_model import Model, ModelConfig


@dataclass
class GaussianSplattingConfig(ModelConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: GaussianSplatting)

    # tracking
    tracking_use_sil_for_loss: bool = True
    tracking_sil_thres: float = 0.99
    tracking_use_l1: bool = True
    tracking_ignore_outlier_depth_loss: bool = False
    tracking_loss_weights = dict(
        rgb=0.5,
        depth=1.0,
    )

    # mapping
    mapping_use_sil_for_loss: bool = False
    mapping_sil_thres: float = 0.5
    mapping_use_l1: bool = True
    mapping_ignore_outlier_depth_loss: bool = False
    mapping_loss_weights = dict(
        rgb=0.5,
        depth=1.0,
    )
    mapping_do_ba: bool = False
    # Needs to be updated based on the number of mapping iterations
    mapping_pruning_dict = dict(
        start_after=0,
        remove_big_after=0,
        stop_after=20,
        prune_every=20,
        removal_opacity_threshold=0.005,
        final_removal_opacity_threshold=0.005,
        reset_opacities=False,
        reset_opacities_every=500,  # Doesn't consider iter 0
    )
    # Use Gaussian Splatting-based Densification during Mapping
    mapping_use_gaussian_splatting_densification: bool = False
    # Needs to be updated based on the number of mapping iterations
    mapping_densify_dict = dict(
        start_after=500,
        remove_big_after=3000,
        stop_after=5000,
        densify_every=100,
        grad_thresh=0.0002,
        num_to_split_into=2,
        removal_opacity_threshold=0.005,
        final_removal_opacity_threshold=0.005,
        reset_opacities_every=3000,  # Doesn't consider iter 0
    )
    mapping_mean_sq_dist_method: str = 'projective'


class GaussianSplatting(Model):
    """Model class."""

    config: GaussianSplattingConfig

    def __init__(
        self,
        config: GaussianSplattingConfig,
        camera: Camera,
        bounding_box,
        **kwargs,
    ) -> None:
        super().__init__(config=config,
                         camera=camera,
                         bounding_box=bounding_box,
                         **kwargs)

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        super().populate_modules()
        self.gaussian_cloud = None

    def model_update(self, cur_frame):
        # Add new Gaussians to the scene based on the Silhouette
        if self.gaussian_cloud is None:
            self.gaussian_cloud = GaussianCloud(
                init_rgb=torch.from_numpy(cur_frame.rgb).to(self.device),
                init_depth=torch.from_numpy(cur_frame.depth).to(self.device),
                w2c=torch.inverse(cur_frame.get_pose()).to(self.device),
                camera=self.camera,
                prune_dict=self.config.mapping_pruning_dict,
                densify_dict=self.config.mapping_densify_dict)
        else:
            self.gaussian_cloud.add_new_gaussians(
                gt_rgb=torch.from_numpy(cur_frame.rgb).to(self.device),
                gt_depth=torch.from_numpy(cur_frame.depth).to(self.device),
                curr_w2c=torch.inverse(cur_frame.get_pose()).to(self.device),
                sil_thres=self.config.mapping_sil_thres,
                mean_sq_dist_method=self.config.mapping_mean_sq_dist_method)

    def post_processing(self, iter, optimizer=None):
        if optimizer is None:
            return
        with torch.no_grad():
            # Prune Gaussians
            self.gaussian_cloud.prune_gaussians(iter, optimizer)
            # Gaussian-Splatting's Gradient-based Densification
            if self.config.mapping_use_gaussian_splatting_densification:
                self.gaussian_cloud.densify(iter, optimizer)

    def get_outputs(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        w2c = input['w2c']
        is_mapping = input['is_mapping']
        retain_grad = input['retain_grad']
        if not is_mapping:
            # only the camera pose gets gradient
            gaussians_grad = False
            camera_grad = True
        else:
            if self.config.mapping_do_ba:
                gaussians_grad = True
                camera_grad = True
            else:
                # only the Gaussians get gradient
                gaussians_grad = True
                camera_grad = False
        # for render image
        if not retain_grad:
            gaussians_grad = False
            camera_grad = False

        return self.gaussian_cloud.render(w2c, gaussians_grad, camera_grad,
                                          retain_grad)

    def get_loss_dict(self,
                      outputs,
                      inputs,
                      is_mapping,
                      stage=None) -> Dict[str, torch.Tensor]:
        loss_dict = {}

        if is_mapping:
            use_sil_for_loss = self.config.mapping_use_sil_for_loss
            sil_thres = self.config.mapping_sil_thres
            use_l1 = self.config.mapping_use_l1
            ignore_outlier_depth_loss = \
                self.config.mapping_ignore_outlier_depth_loss
            loss_weights = self.config.mapping_loss_weights
        else:
            use_sil_for_loss = self.config.tracking_use_sil_for_loss
            sil_thres = self.config.tracking_sil_thres
            use_l1 = self.config.tracking_use_l1
            ignore_outlier_depth_loss = \
                self.config.tracking_ignore_outlier_depth_loss
            loss_weights = self.config.tracking_loss_weights

        target_d = torch.from_numpy(inputs['target_d']).to(
            self.device)  # [H, W]
        target_rgb = torch.from_numpy(inputs['target_s']).to(
            self.device)  # [H, W, 3]
        target_rgb = torch.permute(target_rgb, (2, 0, 1)).float()  # [3, H, W]
        target_d = target_d.unsqueeze(0)  # [1, H, W]

        rgb = outputs['rgb']  # [3, H, W]
        depth_sil = outputs['depth_sil']
        depth = depth_sil[0, :, :].unsqueeze(0)  # [1, H, W]

        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)
        depth_sq = depth_sil[2, :, :].unsqueeze(0)
        uncertainty = depth_sq - depth**2
        uncertainty = uncertainty.detach()

        # Mask with valid depth values (accounts for outlier depth values)
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        if ignore_outlier_depth_loss:
            depth_error = torch.abs(target_d - depth) * (target_d > 0)
            mask = (depth_error < 10 * depth_error.median())
            mask = mask & (target_d > 0)
        else:
            mask = (target_d > 0)
        mask = mask & nan_mask

        # Mask with presence silhouette mask (accounts for empty space)
        if not is_mapping and use_sil_for_loss:
            mask = mask & presence_sil_mask

        # Depth loss
        if use_l1:
            mask = mask.detach()
            if not is_mapping:
                depth_loss = torch.abs(target_d - depth)[mask].sum()
            else:
                depth_loss = torch.abs(target_d - depth)[mask].mean()
            loss_dict['depth'] = depth_loss

        # RGB Loss
        if not is_mapping and (use_sil_for_loss or ignore_outlier_depth_loss):
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            color_loss = torch.abs(target_rgb - rgb)[color_mask].sum()
        elif not is_mapping:
            color_loss = torch.abs(target_rgb - rgb).sum()
        else:
            color_loss = 0.8 * l1_loss_v1(
                rgb, target_rgb) + 0.2 * (1.0 - calc_ssim(rgb, target_rgb))
        loss_dict['rgb'] = color_loss

        # weighted loss
        loss_dict = {k: v * loss_weights[k] for k, v in loss_dict.items()}

        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # gaussian_cloud params
        for key, val in self.gaussian_cloud.params.items():
            val = val.to(self.device)
            param_groups[key] = [val]
        return param_groups
