from dataclasses import dataclass, field
from typing import List, Type

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh

from slam.algorithms.base_algorithm import Algorithm, AlgorithmConfig
from slam.common.camera import Camera
# from slam.common.common import rgbd2pcd
from slam.model_components.neural_recon_components.mesh_renderer import \
    Renderer
from slam.model_components.utils import tsdf2mesh  # rotx
from slam.model_components.utils import (get_view_frustum,
                                         rotate_view_to_align_xyplane)


@dataclass
class NeuralReconConfig(AlgorithmConfig):
    """NeuralRecon  Config."""
    _target: Type = field(default_factory=lambda: NeuralRecon)

    # keyframe selection
    min_angle: float = 15.0
    min_distance: float = 0.1
    # model_input
    max_depth: float = 3.0
    img_size_w: int = 640
    img_size_h: int = 480
    stride: int = 4
    c2w_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # mesh
    mesh_use_double: bool = False


class NeuralRecon(Algorithm):
    def __init__(self, config: NeuralReconConfig, camera: Camera,
                 device: str) -> None:
        super().__init__(config, camera, device)

        self.model = self.config.model.setup(camera=camera, bounding_box=None)
        self.model.to(device)

        self.frag_frames = []
        self.last_mesh = None
        self.cloud = None

        self.voxel_dim = self.model.cfg.MODEL.N_VOX
        self.voxel_size = self.model.cfg.MODEL.VOXEL_SIZE
        self.fragment_id = 0
        # adjust camera intrinsic (640,480)
        img_h = int(self.camera.height /
                    self.config.img_size_h) * self.config.img_size_h
        img_w = int(self.camera.width /
                    self.config.img_size_w) * self.config.img_size_w
        self.h_crop = int((self.camera.height - img_h) / 2)
        self.w_crop = int((self.camera.width - img_w) / 2)
        self.down_sample_h = img_h / self.config.img_size_h
        self.down_sample_w = img_w / self.config.img_size_w
        self.cam_intr = torch.tensor(
            [[
                self.camera.fx / self.down_sample_w, 0,
                (self.camera.cx - self.w_crop) / self.down_sample_w
            ],
             [
                 0, self.camera.fy / self.down_sample_h,
                 (self.camera.cy - self.h_crop) / self.down_sample_h
             ], [0, 0, 1]])

    def render_img(self, c2w, gt_depth=None, idx=None):
        return None, None

    def add_keyframe(self, keyframe):
        pass

    def get_mesh(self):
        with self.lock:
            if self.config.mesh_use_double:
                return self.last_mesh
            else:
                double_mesh = self.last_mesh
                if double_mesh is None:
                    return double_mesh
                renderer = Renderer()
                mesh_opengl = renderer.mesh_opengl(double_mesh)
                voxel_size = 4
                volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=float(voxel_size) / 100,
                    sdf_trunc=3 * float(voxel_size) / 100,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.
                    RGB8)
                for i in range(len(self.estimate_c2w_list)):
                    cam_pose = self.estimate_c2w_list[i]
                    _, depth_pred = renderer(self.config.img_size_h,
                                             self.config.img_size_w,
                                             self.cam_intr, cam_pose,
                                             mesh_opengl)
                    color_im = np.repeat(depth_pred[:, :, np.newaxis] * 255,
                                         3,
                                         axis=2).astype(np.uint8)
                    depth_pred = o3d.geometry.Image(depth_pred)
                    color_im = o3d.geometry.Image(color_im)
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_im,
                        depth_pred,
                        depth_scale=1.0,
                        depth_trunc=5.0,
                        convert_rgb_to_intensity=False)

                    volume.integrate(
                        rgbd,
                        o3d.camera.PinholeCameraIntrinsic(
                            width=self.config.img_size_w,
                            height=self.config.img_size_h,
                            fx=self.cam_intr[0, 0],
                            fy=self.cam_intr[1, 1],
                            cx=self.cam_intr[0, 2],
                            cy=self.cam_intr[1, 2]), np.linalg.inv(cam_pose))
                o3d_mesh = volume.extract_triangle_mesh()
                vertices = o3d_mesh.vertices
                faces = o3d_mesh.triangles
                single_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                return single_mesh

    def get_cloud(self, c2w_np, gt_depth_np):
        return self.cloud

    def do_tracking(self, cur_frame):
        c2w = cur_frame.gt_pose
        # the coordinate system should be same with ScanNet
        # X left to right, Y up to down and Z in the positive viewing direction
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        # ------- For 7-Scenes only -------
        # rot_mat = c2w[:3, :3]
        # trans = c2w[:3, 3]
        # c2w[:3, :3] = rotx(-np.pi / 2) @ rot_mat
        # c2w[:3, 3] = rotx(-np.pi / 2) @ trans
        # should make sure the pointcloud is in the positive direction of XYZ.
        c2w[:3, 3] += np.array(self.config.c2w_offset)
        # Save the point cloud to observe how the c2w is adjusted such
        # that the world coordinate system's z-axis is facing upwards,
        # with the xy-plane as the horizontal plane.
        # init_pts, init_cols = rgbd2pcd(cur_frame.rgb,
        #                                cur_frame.depth,
        #                                c2w,
        #                                self.camera,
        #                                'color',
        #                                device=self.device)
        # self.cloud = init_pts, init_cols
        return c2w

    def get_model_input(self, optimize_frames, is_mapping=True):
        vol_origin = torch.tensor(np.array([0.0, 0.0, 0.0]))
        bnds = torch.zeros((3, 2))
        bnds[:, 0] = np.inf
        bnds[:, 1] = -np.inf

        n_imgs = len(optimize_frames)
        proj_matrices = []
        middle_pose = optimize_frames[n_imgs //
                                      2].get_pose().detach().cpu().numpy()
        rotation_matrix = rotate_view_to_align_xyplane(middle_pose)
        rotation_matrix4x4 = np.eye(4)
        rotation_matrix4x4[:3, :3] = rotation_matrix
        world_to_aligned_camera = torch.from_numpy(
            rotation_matrix4x4).float() @ np.linalg.inv(middle_pose)

        imgs = []
        for idx in range(n_imgs):
            frame = optimize_frames[idx]
            rgb_np = frame.rgb
            # crop and resize to (640, 480)
            if self.h_crop > 0:
                rgb_np = rgb_np[self.h_crop:-self.h_crop, :]  # (H, W, C)
            if self.w_crop > 0:
                rgb_np = rgb_np[:, self.w_crop:-self.w_crop]  # (H, W, C)
            rgb_np = cv2.resize(
                rgb_np, (self.config.img_size_w, self.config.img_size_h),
                interpolation=cv2.INTER_LINEAR)
            resized_image_np = rgb_np * 255.
            resized_image = torch.from_numpy(resized_image_np).permute(
                2, 0, 1)  # [C, H, W]
            imgs.append(resized_image)
            # computing visual frustum hull
            size = resized_image.shape[1:]
            cam_pose = frame.get_pose().detach()  # c2w
            view_frust_pts = get_view_frustum(self.config.max_depth, size,
                                              self.cam_intr, cam_pose)
            bnds[:, 0] = torch.min(bnds[:, 0],
                                   torch.min(view_frust_pts, dim=1)[0])
            bnds[:, 1] = torch.max(bnds[:, 1],
                                   torch.max(view_frust_pts, dim=1)[0])
            view_proj_matrics = []
            for i in range(3):
                proj_mat = torch.inverse(cam_pose)  # w2c
                scale_intrinsics = self.cam_intr / self.config.stride / 2**i
                scale_intrinsics[-1, -1] = 1
                proj_mat[:3, :4] = scale_intrinsics @ proj_mat[:3, :4]
                view_proj_matrics.append(proj_mat)
            view_proj_matrics = torch.stack(view_proj_matrics)
            proj_matrices.append(view_proj_matrics)
        # adjust volume bounds
        num_layers = 3
        center = (torch.tensor([(bnds[0, 1] + bnds[0, 0]) / 2,
                                (bnds[1, 1] + bnds[1, 0]) / 2,
                                (bnds[2, 1] + bnds[2, 0]) / 2]) -
                  vol_origin) / self.voxel_size
        center[:3] = torch.round(center[:3] / 2**num_layers) * 2**num_layers
        origin = torch.zeros_like(center)
        origin[:3] = center[:3] - torch.tensor(self.voxel_dim[:3]) // 2
        # center = (torch.tensor(((bnds[0, 1] + bnds[0, 0]) / 2,
        #                         (bnds[1, 1] + bnds[1, 0]) / 2, -0.2)) -
        #           vol_origin) / self.voxel_size
        # center[:2] = torch.round(center[:2] / 2**num_layers) * 2**num_layers
        # center[2] = torch.floor(center[2] / 2**num_layers) * 2**num_layers
        # origin = torch.zeros_like(center)
        # origin[:2] = center[:2] - torch.tensor(self.voxel_dim[:2]) // 2
        # origin[2] = center[2]
        vol_origin_partial = origin * self.voxel_size + vol_origin
        # batch_size = 1
        inputs = {
            'imgs':
            torch.stack(imgs,
                        dim=0).unsqueeze(0),  # (batch_size, n_views, C, H, W)
            'scene': ['neucon_demodata_b5f1'],
            'fragment': [('neucon_demodata_b5f1_' + str(self.fragment_id))],
            'vol_origin': vol_origin.unsqueeze(0),
            'vol_origin_partial': vol_origin_partial.unsqueeze(0),
            'world_to_aligned_camera': world_to_aligned_camera.unsqueeze(0),
            'proj_matrices': torch.stack(proj_matrices).unsqueeze(0),
        }
        self.fragment_id += 1
        return inputs

    # only 3D reconstruction, not optimize poses
    def do_mapping(self, cur_frame):
        if not self.is_initialized():
            self.set_initialized()

        self.check_keyframe(cur_frame)
        if len(self.frag_frames) <= self.config.mapping_window_size:
            return

        inputs = self.get_model_input(self.frag_frames)
        outputs = self.model(inputs)

        with self.lock:
            if 'scene_tsdf' in outputs:
                tsdf_volume = outputs['scene_tsdf'][0].data.cpu().numpy()
                origin = outputs['origin'][0].data.cpu().numpy()
                if (tsdf_volume == 1).all() or len(tsdf_volume) < 2:
                    print('No valid data for mesh generation.')
                else:
                    self.last_mesh = tsdf2mesh(self.model.cfg.MODEL.VOXEL_SIZE,
                                               origin, tsdf_volume)

        self.frag_frames.clear()
        torch.cuda.empty_cache()

    def check_keyframe(self, cur_frame):
        if len(self.frag_frames) == 0:
            self.frag_frames.append(cur_frame)
        else:
            last_pose = self.frag_frames[-1].get_pose().detach().cpu().numpy()
            cur_pose = cur_frame.get_pose().detach().cpu().numpy()
            temp = ((np.linalg.inv(cur_pose[:3, :3]) @ last_pose[:3, :3]
                     @ np.array([0, 0, 1]).T) * np.array([0, 0, 1])).sum()
            temp = np.clip(temp, -1, 1)
            angle = np.arccos(temp)
            dis = np.linalg.norm(cur_pose[:3, 3] - last_pose[:3, 3])
            if angle > (self.config.min_angle /
                        180) * np.pi or dis > self.config.min_distance:
                self.frag_frames.append(cur_frame)
