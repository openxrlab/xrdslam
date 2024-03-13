"""Put all the method implementations in one location."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import tyro

from slam.algorithms.coslam import CoSLAMConfig
from slam.algorithms.nice_slam import NiceSLAMConfig
from slam.algorithms.point_slam import PointSLAMConfig
from slam.algorithms.splatam import SplaTAMConfig
from slam.algorithms.voxfusion import VoxFusionConfig
from slam.common.mesher import MesherConfig
from slam.engine.optimizers import AdamOptimizerConfig
from slam.engine.schedulers import (LRconfig, NiceSLAMSchedulerConfig,
                                    PointSLAMSchedulerConfig)
from slam.engine.xrdslamer import XRDSLAMerConfig
from slam.models.conv_onet import ConvOnetConfig
from slam.models.conv_onet2 import ConvOnet2Config
from slam.models.gaussian_splatting import GaussianSplattingConfig
from slam.models.joint_encoding import JointEncodingConfig
from slam.models.sparse_voxel import SparseVoxelConfig
from slam.pipeline.mapper import MapperConfig
from slam.pipeline.tracker import TrackerConfig
from slam.pipeline.visualizer import VisualizerConfig
from slam.pipeline.xrdslam import XRDSLAMConfig

method_configs: Dict[str, XRDSLAMerConfig] = {}

descriptions = {
    'nice-slam': 'Inplementation of nice-slam.',
    'vox-fusion': 'Implementation of vox-fusion.',
    'co-slam': 'Implementation of co-slam.',
    'point-slam': 'Implementation of point-slam.',
    'splaTAM': 'Implementation of splaTAM.',
}

method_configs['nice-slam'] = XRDSLAMerConfig(
    method_name='nice-slam',
    xrdslam=XRDSLAMConfig(
        tracker=TrackerConfig(map_every=5,
                              render_freq=50,
                              use_relative_pose=False,
                              save_debug_result=False),
        mapper=MapperConfig(keyframe_every=50, ),
        method=NiceSLAMConfig(
            coarse=True,
            tracking_n_iters=10,
            mapping_n_iters=60,
            mapping_first_n_iters=1500,
            mapping_window_size=5,
            tracking_sample=200,
            mapping_sample=1000,
            min_sample_pixels=200,
            ray_batch_size=100000,
            tracking_Wedge=100,
            tracking_Hedge=100,
            # office0
            mapping_bound=[[-5.5, 5.9], [-6.7, 5.4], [-4.7, 5.3]],
            marching_cubes_bound=[[-5.5, 5.9], [-6.7, 5.4], [-4.7, 5.3]],
            # office1
            # mapping_bound=[[-5.3,6.5],[-5.1,6.0],[-4.5,5.2]],
            # marching_cubes_bound=[[-5.3,6.5],[-5.1,6.0],[-4.5,5.2]],
            # office2
            # mapping_bound=[[-5.0,4.6],[-4.4,6.9],[-2.8,3.1]],
            # marching_cubes_bound=[[-5.0,4.6],[-4.4,6.9],[-2.8,3.1]],
            # office3
            # mapping_bound=[[-6.7,5.1],[-7.5,4.9],[-2.8,3.5]],
            # marching_cubes_bound=[[-6.7,5.1],[-7.5,4.9],[-2.8,3.5]],
            # office4
            # mapping_bound=[[-3.7,7.8],[-4.8,6.7],[-3.7,4.1]],
            # marching_cubes_bound=[[-3.7,7.8],[-4.8,6.7],[-3.7,4.1]],
            # room0
            # mapping_bound=[[-2.9,8.9],[-3.2,5.5],[-3.5,3.3]],
            # marching_cubes_bound=[[-2.9,8.9],[-3.2,5.5],[-3.5,3.3]],
            # room1
            # mapping_bound=[[-7.0,2.8],[-4.6,4.3],[-3.0,2.9]],
            # marching_cubes_bound=[[-7.0,2.8],[-4.6,4.3],[-3.0,2.9]],
            # room2
            # mapping_bound=[[-4.3,9.5],[-6.7,5.2],[-6.4,4.2]],
            # marching_cubes_bound=[[-4.3,9.5],[-6.7,5.2],[-6.4,4.2]],
            mapping_middle_iter_ratio=0.4,
            mapping_fine_iter_ratio=0.6,
            mapping_lr_factor=1.0,
            mapping_lr_first_factor=5.0,
            mesher=MesherConfig(
                resolution=256,
                points_batch_size=30000,
            ),
            model=ConvOnetConfig(
                points_batch_size=100000,
                mapping_frustum_feature_selection=True,
                pretrained_decoders_coarse=Path(
                    'pretrained/nice_slam/coarse.pt'),
                pretrained_decoders_middle_fine=Path(
                    'pretrained/nice_slam/middle_fine.pt'),
            ),
            optimizers={
                'decoder': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    NiceSLAMSchedulerConfig(stage_lr=LRconfig(
                        coarse=0.0, middle=0.0, fine=0.0, color=0.005)),
                },
                'grid_coarse': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    NiceSLAMSchedulerConfig(stage_lr=LRconfig(
                        coarse=0.001, middle=0.0, fine=0.0, color=0.0)),
                },
                'grid_middle': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    NiceSLAMSchedulerConfig(stage_lr=LRconfig(
                        coarse=0.0, middle=0.1, fine=0.005, color=0.005)),
                },
                'grid_fine': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    NiceSLAMSchedulerConfig(stage_lr=LRconfig(
                        coarse=0.0, middle=0.0, fine=0.005, color=0.005)),
                },
                'grid_color': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    NiceSLAMSchedulerConfig(stage_lr=LRconfig(
                        coarse=0.0, middle=0.0, fine=0.0, color=0.005)),
                },
                'tracking_pose': {
                    'optimizer': AdamOptimizerConfig(lr=1e-3),
                    'scheduler': None,
                },
                'mapping_pose': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    NiceSLAMSchedulerConfig(stage_lr=LRconfig(
                        coarse=0.0, middle=0.0, fine=0.0, color=0.001)),
                },
            },
        ),
        visualizer=VisualizerConfig(),
        enable_vis=False,
        device='cuda:0',
    ))

method_configs['vox-fusion'] = XRDSLAMerConfig(
    method_name='vox-fusion',
    xrdslam=XRDSLAMConfig(
        tracker=TrackerConfig(map_every=1,
                              render_freq=50,
                              use_relative_pose=True,
                              save_debug_result=False,
                              init_pose_offset=10),
        mapper=MapperConfig(keyframe_every=10, ),
        method=VoxFusionConfig(
            # keyframe_selection_method='random',
            tracking_n_iters=30,
            mapping_n_iters=15,  # 30
            mapping_first_n_iters=100,
            mapping_window_size=5,
            mapping_sample=1024,
            tracking_sample=1024,
            ray_batch_size=3000,
            model=SparseVoxelConfig(),
            optimizers={
                'decoder': {
                    'optimizer': AdamOptimizerConfig(lr=5e-3),
                    'scheduler': None,
                },
                'embeddings': {
                    'optimizer': AdamOptimizerConfig(lr=5e-3),
                    'scheduler': None,
                },
                'tracking_pose': {
                    'optimizer': AdamOptimizerConfig(lr=1e-2),
                    'scheduler': None,
                },
                'mapping_pose': {
                    'optimizer': AdamOptimizerConfig(lr=1e-3),
                    'scheduler': None,
                },
            },
        ),
        visualizer=VisualizerConfig(),
        enable_vis=False,
        device='cuda:0',
    )  # TODO: only support cuda:0 now
)

method_configs['co-slam'] = XRDSLAMerConfig(
    method_name='co-slam',
    xrdslam=XRDSLAMConfig(
        tracker=TrackerConfig(map_every=5,
                              render_freq=50,
                              use_relative_pose=False,
                              save_debug_result=False),
        mapper=MapperConfig(keyframe_every=5, ),
        method=CoSLAMConfig(
            separate_LR=True,
            retain_graph=True,
            rot_rep='axis_angle',
            tracking_n_iters=10,
            mapping_n_iters=10,
            mapping_first_n_iters=200,
            keyframe_selection_method='all',
            mapping_sample=2048,
            tracking_sample=1024,
            min_sample_pixels=100,
            ray_batch_size=30000,
            tracking_Wedge=20,
            tracking_Hedge=20,
            # office0
            mapping_bound=[[-3, 3], [-4, 2.5], [-2, 2.5]],
            marching_cubes_bound=[[-2.2, 2.6], [-3.4, 2.1], [-1.4, 2.0]],
            # office1
            # mapping_bound=[[-2,3.2],[-1.7,2.7],[-1.2,2.0]],
            # marching_cubes_bound=[[-1.9,3.1],[-1.6,2.6],[-1.1,1.8]],
            # office2
            # mapping_bound=[[-3.6,3.2],[-3.0,5.5],[-1.4,1.7]],
            # marching_cubes_bound=[[-3.5,3.1],[-2.9,5.4],[-1.3,1.6]],
            # office3
            # mapping_bound=[[-5.3,3.7],[-6.1,3.4],[-1.4,2.0]],
            # marching_cubes_bound=[[-5.2,3.6],[-6.0,3.3],[-1.3,1.9]],
            # office4
            # mapping_bound=[[-1.4,5.5],[-2.5,4.4],[-1.4,1.8]],
            # marching_cubes_bound=[[-1.3,5.4],[-2.4,4.3],[-1.3,1.7]],
            # room0
            # mapping_bound=[[-1.0,7.0],[-1.3,3.7],[-1.7,1.4]],
            # marching_cubes_bound=[[-1.0,7.0],[-1.3,3.7],[-1.7,1.4]],
            # room1
            # mapping_bound=[[-5.6, 1.4], [-3.2, 2.8], [-1.6, 1.8]],
            # marching_cubes_bound=[[-5.6, 1.4], [-3.2, 2.8], [-1.6, 1.8]],
            # room2
            # mapping_bound=[[-1.0,6.1],[-3.4,1.9],[-3.1,0.8]],
            # marching_cubes_bound=[[-0.9,6.0],[-3.3,1.8],[-3.0,0.7]],
            mesher=MesherConfig(
                resolution=256,
                points_batch_size=30000,
            ),
            model=JointEncodingConfig(cam_depth_trunc=100.0,
                                      tcnn_encoding=True),
            optimizers={
                'decoder': {
                    'optimizer':
                    AdamOptimizerConfig(lr=1e-2,
                                        weight_decay=1e-6,
                                        betas=(0.9, 0.99)),
                    'scheduler':
                    None,
                },
                'embed_fn': {
                    'optimizer':
                    AdamOptimizerConfig(lr=1e-2, eps=1e-15, betas=(0.9, 0.99)),
                    'scheduler':
                    None,
                },
                'embed_fn_color': {
                    'optimizer':
                    AdamOptimizerConfig(lr=1e-2, eps=1e-15, betas=(0.9, 0.99)),
                    'scheduler':
                    None,
                },
                'tracking_pose_r': {
                    'optimizer': AdamOptimizerConfig(lr=1e-3),
                    'scheduler': None,
                },
                'tracking_pose_t': {
                    'optimizer': AdamOptimizerConfig(lr=1e-3),
                    'scheduler': None,
                },
                'mapping_pose_r': {
                    'optimizer': AdamOptimizerConfig(lr=1e-3, accum_step=5),
                    'scheduler': None,
                },
                'mapping_pose_t': {
                    'optimizer': AdamOptimizerConfig(lr=1e-3, accum_step=5),
                    'scheduler': None,
                },
            },
        ),
        visualizer=VisualizerConfig(),
        enable_vis=False,
        device='cuda:0'))

method_configs['point-slam'] = XRDSLAMerConfig(
    method_name='point-slam',
    xrdslam=XRDSLAMConfig(
        tracker=TrackerConfig(map_every=5,
                              lazy_start=20,
                              render_freq=50,
                              use_relative_pose=False,
                              save_debug_result=False),
        mapper=MapperConfig(keyframe_every=20, ),
        method=PointSLAMConfig(
            separate_LR=True,
            tracking_n_iters=40,
            mapping_n_iters=300,
            mapping_first_n_iters=1500,
            mapping_window_size=12,
            tracking_sample=1500,
            mapping_sample=5000,
            min_sample_pixels=40,
            ray_batch_size=3000,
            tracking_Wedge=100,
            tracking_Hedge=100,
            mapping_BA=False,
            mapping_frustum_feature_selection=True,
            mapping_pixels_based_on_color_grad=1000,
            model=ConvOnet2Config(
                cuda_id=0,  # should be same with 'device'
                points_batch_size=500000,
                pretrained_decoders_middle_fine=Path(
                    'pretrained/point_slam/middle_fine.pt'),
            ),
            optimizers={
                'decoder': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    PointSLAMSchedulerConfig(start_lr=0.001, end_lr=0.005),
                },
                'geometry': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    PointSLAMSchedulerConfig(start_lr=0.03, end_lr=0.005),
                },
                'color': {
                    'optimizer':
                    AdamOptimizerConfig(),
                    'scheduler':
                    PointSLAMSchedulerConfig(start_lr=0.0, end_lr=0.005),
                },
                # "tracking_pose": {
                #     "optimizer": AdamOptimizerConfig(lr=0.002),
                #     "scheduler": None,
                # },
                # "mapping_pose": {
                #     "optimizer": AdamOptimizerConfig(lr=0.0002),
                #     "scheduler": None,
                # },
                'tracking_pose_r': {
                    'optimizer': AdamOptimizerConfig(lr=0.002 * 0.2),
                    'scheduler': None,
                },
                'tracking_pose_t': {
                    'optimizer': AdamOptimizerConfig(lr=0.002),
                    'scheduler': None,
                },
                'mapping_pose_r': {
                    'optimizer': AdamOptimizerConfig(lr=0.0002),
                    'scheduler': None,
                },
                'mapping_pose_t': {
                    'optimizer': AdamOptimizerConfig(lr=0.0002),
                    'scheduler': None,
                },
            },
        ),
        visualizer=VisualizerConfig(),
        enable_vis=False,
        device='cuda:0'))

method_configs['splaTAM'] = XRDSLAMerConfig(
    method_name='splaTAM',
    xrdslam=XRDSLAMConfig(
        tracker=TrackerConfig(map_every=1,
                              render_freq=50,
                              use_relative_pose=True,
                              save_debug_result=False),
        mapper=MapperConfig(keyframe_every=5, ),
        method=SplaTAMConfig(
            retain_graph=False,
            separate_LR=True,
            keyframe_use_ray_sample=False,
            tracking_n_iters=40,
            mapping_n_iters=60,
            mapping_first_n_iters=60,
            mapping_window_size=24,
            model=GaussianSplattingConfig(),
            optimizers={
                'means3D': {
                    'optimizer': AdamOptimizerConfig(lr=0.0001, eps=1e-15),
                    'scheduler': None,
                },
                'rgb_colors': {
                    'optimizer': AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                    'scheduler': None,
                },
                'unnorm_rotations': {
                    'optimizer': AdamOptimizerConfig(lr=0.001, eps=1e-15),
                    'scheduler': None,
                },
                'logit_opacities': {
                    'optimizer': AdamOptimizerConfig(lr=0.05, eps=1e-15),
                    'scheduler': None,
                },
                'log_scales': {
                    'optimizer': AdamOptimizerConfig(lr=0.001, eps=1e-15),
                    'scheduler': None,
                },
                # "tracking_pose": {
                #     "optimizer": AdamOptimizerConfig(lr=0.002),
                #     "scheduler": None,
                # },
                'tracking_pose_r': {
                    'optimizer': AdamOptimizerConfig(lr=0.0004),
                    'scheduler': None,
                },
                'tracking_pose_t': {
                    'optimizer': AdamOptimizerConfig(lr=0.002),
                    'scheduler': None,
                },
            },
        ),
        visualizer=VisualizerConfig(),
        enable_vis=False,
        device='cuda:0')  # TODO: only support cuda:0 now
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
    # Don't show unparsable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[tyro.extras.subcommand_type_from_defaults(
        defaults=method_configs, descriptions=descriptions)]]
"""Union[] type over config types, annotated with default instances for use
with tyro.cli(). Allows the user to pick between one of several base
configurations, and then override values in it."""
