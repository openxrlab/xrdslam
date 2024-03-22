# This file is based on code from Nice-slam,
# (https://github.com/cvg/nice-slam/blob/master/src/utils/Mesher.py)
# licensed under the Apache License, Version 2.0.

from dataclasses import dataclass, field
from typing import Type

import numpy as np
import open3d as o3d
import skimage
import torch
import trimesh
from packaging import version

from slam.common.camera import Camera
from slam.configs.base_config import InstantiateConfig


@dataclass
class MesherConfig(InstantiateConfig):
    """Mesher  Config."""
    _target: Type = field(default_factory=lambda: Mesher)

    points_batch_size: int = 500000
    resolution: int = 130
    level_set: int = 0
    remove_small_geometry_threshold: float = 0.2
    clean_mesh_bound_scale: float = 1.02
    get_largest_components: bool = False


class Mesher():
    def __init__(self, config: MesherConfig, camera: Camera, bounding_box,
                 marching_cubes_bound) -> None:
        self.config = config
        self.bounding_box = bounding_box
        self.marching_cubes_bound = marching_cubes_bound
        self.camera = camera
        self.scale = 1.0

    def get_grid_uniform(self, resolution):
        """Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.marching_cubes_bound

        padding = 0.0
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding,
                        resolution)
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding,
                        resolution)
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding,
                        resolution)

        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        grid_points = torch.tensor(np.vstack(
            [xx.ravel(), yy.ravel(), zz.ravel()]).T,
                                   dtype=torch.float)

        return {'grid_points': grid_points, 'xyz': [x, y, z]}

    def get_bound_from_frames(self, keyframe_graph, scale=1):
        """Get the scene bound (convex hull), using sparse estimated camera
        poses and corresponding depth images.

        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """

        H, W, fx, fy, cx, cy = (self.camera.height, self.camera.width,
                                self.camera.fx, self.camera.fy, self.camera.cx,
                                self.camera.cy)

        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for keyframe in keyframe_graph:
            c2w = keyframe.get_pose().cpu().numpy()
            # convert to open3d camera pose
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])
            depth = keyframe.depth
            color = keyframe.rgb

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)

            volume.integrate(rgbd, intrinsic, w2c)

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.config.clean_mesh_bound_scale,
                              mesh.get_center())
        else:
            mesh = mesh.scale(self.config.clean_mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return return_mesh

    def eval_points(self, p, query_fn, boundingbox, device='cuda:0'):
        """Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): point coordinates.
            query_fn: .
            device (str, optional): device name to compute on.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """
        p_split = torch.split(p, self.config.points_batch_size)
        bound = boundingbox

        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            # render p
            ret = query_fn(pi)

            ret[~mask, :] = 100
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    def get_mesh(self,
                 keyframe_graph,
                 query_fn,
                 color_func=None,
                 device='cuda:0',
                 use_mask=False):

        with torch.no_grad():
            grid = self.get_grid_uniform(self.config.resolution)
            points = grid['grid_points']
            points = points.to(device)

            z = []
            for i, pnts in enumerate(
                    torch.split(points, self.config.points_batch_size, dim=0)):
                z.append(
                    self.eval_points(pnts,
                                     query_fn=query_fn,
                                     boundingbox=self.bounding_box,
                                     device=device).cpu().numpy()[:, -1])
            z = np.concatenate(z, axis=0)
            if use_mask:
                mesh_bound = self.get_bound_from_frames(keyframe_graph,
                                                        scale=1.0)
                mask = []
                for i, pnts in enumerate(
                        torch.split(points,
                                    self.config.points_batch_size,
                                    dim=0)):
                    mask.append(mesh_bound.contains(pnts.cpu().numpy()))
                mask = np.concatenate(mask, axis=0)
                z[~mask] = 100
            z = z.astype(np.float32)

            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, _, _ = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.config.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, _, _ = \
                        skimage.measure.marching_cubes_lewiner(
                            volume=z.reshape(
                                grid['xyz'][1].shape[0],
                                grid['xyz'][0].shape[0],
                                grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                            level=self.config.level_set,
                            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                     grid['xyz'][1][2] - grid['xyz'][1][1],
                                     grid['xyz'][2][2] - grid['xyz'][2][1]))
            except Exception as e:
                print('marching_cubes error:', e)
                print('Possibly no surface extracted from the level set.')
                return
            # convert back to world coordinates
            vertices = verts + np.array(
                [grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            if color_func is not None:
                # color is extracted by passing the coordinates of mesh
                # vertices through the network
                points = torch.from_numpy(vertices)
                z = []

                for i, pnts in enumerate(
                        torch.split(points,
                                    self.config.points_batch_size,
                                    dim=0)):
                    z_color = self.eval_points(pnts.to(device).float(),
                                               query_fn=color_func,
                                               boundingbox=self.bounding_box,
                                               device=device).cpu()[..., :3]
                    z.append(z_color)
                z = torch.cat(z, axis=0)
                vertex_colors = z.cpu().numpy()
                vertex_colors = np.clip(vertex_colors, 0, 1) * 255
                vertex_colors = vertex_colors.astype(np.uint8)
            else:
                vertex_colors = None

            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices,
                                   faces,
                                   vertex_colors=vertex_colors)
            # mesh.fix_normals()

            return mesh
