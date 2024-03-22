import functools
from dataclasses import dataclass, field
from queue import Empty
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from scripts.utils.viz_utils import create_camera_actor
from slam.common.camera import Camera
from slam.configs.base_config import InstantiateConfig


@dataclass
class VisualizerConfig(InstantiateConfig):
    """Visualizer  Config."""
    _target: Type = field(default_factory=lambda: Visualizer)
    save_rendering: bool = True
    eval_img: bool = False
    win_w: int = 800
    win_h: int = 600
    img_show_w: int = 640
    img_show_h: int = 480

    cam_scale: float = 0.03


class Visualizer():
    def __init__(self,
                 config: VisualizerConfig,
                 camera: Camera,
                 out_dir='.') -> None:
        self.config = config
        self.config.img_show_h = camera.height
        self.config.img_show_w = camera.width

        self.out_dir = out_dir
        self.estimate_c2w_list = []
        self.gt_c2w_list = []
        # show objects
        self.cam_actor = None
        self.show_mesh = None
        self.show_cloud = None
        self.show_traj_actor = None
        self.show_traj_actor_gt = None

        self.pose_prev = None
        # for eval img
        self.frame_cnt, self.psnr_sum, self.ssim_sum, self.lpips_sum, \
            self.depth_l1_render_sum = 0, 0, 0, 0, 0

    def animation_callback(self, vis, viz_queue):
        # cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        while True:
            try:
                data = viz_queue.get_nowait()
                if data[0] == 'pose':
                    idx, est_pose, gt_pose = data[1:]
                    if isinstance(est_pose, torch.Tensor):
                        est_pose = est_pose.cpu().numpy()
                    if isinstance(gt_pose, torch.Tensor):
                        gt_pose = gt_pose.cpu().numpy()

                    self.estimate_c2w_list.append(est_pose.copy())
                    self.gt_c2w_list.append(gt_pose)

                    # show camera
                    est_pose[:3, 2] *= -1
                    if self.cam_actor is None:
                        self.cam_actor = create_camera_actor(
                            is_gt=False, scale=self.config.cam_scale)
                        self.cam_actor.transform(est_pose)
                        vis.add_geometry(self.cam_actor)
                    else:
                        pose_change = est_pose @ np.linalg.inv(self.pose_prev)
                        self.cam_actor.transform(pose_change)
                        vis.update_geometry(self.cam_actor)
                    self.pose_prev = est_pose

                    # show traj
                    color = (0.0, 0.0, 0.0)
                    color_gt = (1.0, .0, .0)
                    traj_actor = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(
                            [pose[:3, 3] for pose in self.estimate_c2w_list]))
                    traj_actor.paint_uniform_color(color)
                    traj_actor_gt = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(
                            [pose[:3, 3] for pose in self.gt_c2w_list]))
                    traj_actor_gt.paint_uniform_color(color_gt)

                    if self.show_traj_actor is not None:
                        vis.remove_geometry(self.show_traj_actor)
                        del self.show_traj_actor
                    self.show_traj_actor = traj_actor
                    vis.add_geometry(self.show_traj_actor)
                    if self.show_traj_actor_gt is not None:
                        vis.remove_geometry(self.show_traj_actor_gt)
                        del self.show_traj_actor_gt
                    self.show_traj_actor_gt = traj_actor_gt
                    vis.add_geometry(self.show_traj_actor_gt)

                elif data[0] == 'img':
                    idx, gt_color, gt_depth, rcolor, rdepth = data[1:]
                    if rcolor is None or rdepth is None:
                        rcolor = np.zeros_like(gt_color)
                        rdepth = np.zeros_like(gt_depth)
                    gt_color = np.clip(gt_color, 0, 1)
                    rcolor = np.clip(rcolor, 0, 1)
                    depth_residual = np.abs(gt_depth - rdepth)
                    depth_residual[gt_depth == 0.0] = 0.0
                    color_residual = np.abs(gt_color - rcolor)
                    color_residual[gt_depth == 0.0] = 0.0
                    color_residual = np.clip(color_residual, 0, 1)
                    max_depth = np.max(gt_depth)
                    self.depth_plot = self.ax_depth.imshow(gt_depth,
                                                           cmap='plasma',
                                                           vmin=0,
                                                           vmax=max_depth)
                    self.rdepth_plot = self.ax_rdepth.imshow(rdepth,
                                                             cmap='plasma',
                                                             vmin=0,
                                                             vmax=max_depth)
                    self.depth_diff_plot = self.ax_depth_diff.imshow(
                        depth_residual, cmap='plasma', vmin=0, vmax=max_depth)
                    self.color_plot.set_array(gt_color)
                    self.rcolor_plot.set_array(rcolor)
                    self.color_diff_plot.set_array(color_residual)
                    # 2d metrics
                    # rgb
                    depth_mask = (torch.from_numpy(
                        gt_depth > 0).unsqueeze(-1)).float()
                    gt_color = torch.tensor(gt_color) * depth_mask
                    rcolor = torch.tensor(rcolor) * depth_mask
                    mse_loss = torch.nn.functional.mse_loss(gt_color, rcolor)
                    psnr = -10. * torch.log10(mse_loss)
                    ssim = ms_ssim(gt_color.transpose(0,
                                                      2).unsqueeze(0).float(),
                                   rcolor.transpose(0, 2).unsqueeze(0).float(),
                                   data_range=1.0,
                                   size_average=True)
                    cal_lpips = LearnedPerceptualImagePatchSimilarity(
                        net_type='alex', normalize=True)
                    lpips = cal_lpips(
                        (gt_color).unsqueeze(0).permute(0, 3, 1, 2).float(),
                        (rcolor).unsqueeze(0).permute(0, 3, 1,
                                                      2).float()).item()

                    # depth
                    gt_depth = torch.tensor(gt_depth)
                    rdepth = torch.tensor(rdepth)
                    depth_l1_render = torch.abs(
                        gt_depth[gt_depth > 0] -
                        rdepth[gt_depth > 0]).mean().item() * 100
                    text = (f'PSNR[dB]^: {psnr.item():.2f}, SSIM^: {ssim:.2f},'
                            f'LPIPS: {lpips:.2f}, Depth_L1[cm]:'
                            f'{depth_l1_render:.2f}')

                    if hasattr(self, 'metrics_text'):
                        self.metrics_text.remove()
                    self.metrics_text = self.fig.text(0.02,
                                                      0.02,
                                                      text,
                                                      ha='left',
                                                      va='bottom',
                                                      fontsize=12,
                                                      color='red')

                    # 2d rendering metric average
                    if self.config.eval_img:
                        self.psnr_sum += psnr
                        self.ssim_sum += ssim
                        self.lpips_sum += lpips
                        self.depth_l1_render_sum += depth_l1_render
                        self.frame_cnt += 1
                        avg_psnr = self.psnr_sum / self.frame_cnt
                        avg_ssim = self.ssim_sum / self.frame_cnt
                        avg_lpips = self.lpips_sum / self.frame_cnt
                        avg_depth_l1 = \
                            self.depth_l1_render_sum / self.frame_cnt
                        print(f'avg_psnr[dB]: {avg_psnr}, avg_ms_ssim:'
                              f'{avg_ssim}, avg_lpips: {avg_lpips},'
                              f'avg_depth_l1[cm]: {avg_depth_l1}')

                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.pause(0.1)
                    if self.config.save_rendering:
                        plt.savefig(f'{self.out_dir}/imgs/{idx:05d}.jpg',
                                    bbox_inches='tight',
                                    pad_inches=0.2)

                elif data[0] == 'mesh':
                    idx, input_mesh = data[1:]
                    if input_mesh is None:
                        continue
                    if self.show_mesh is not None:
                        vis.remove_geometry(self.show_mesh)
                    else:
                        self.show_mesh = o3d.geometry.TriangleMesh()
                    # trimesh.Trimesh -> o3d.geometry.TriangleMesh
                    self.show_mesh.vertices = o3d.utility.Vector3dVector(
                        input_mesh.vertices)
                    self.show_mesh.triangles = o3d.utility.Vector3iVector(
                        input_mesh.faces)
                    if hasattr(input_mesh.visual, 'vertex_colors'):
                        vertex_colors_rgb = np.asarray(
                            input_mesh.visual.vertex_colors[:, :3])
                        self.show_mesh.vertex_colors = \
                            o3d.utility.Vector3dVector(
                                vertex_colors_rgb / 255.0)
                    vis.add_geometry(self.show_mesh)
                    if self.config.save_rendering:
                        mesh_savepath = f'{self.out_dir}/mesh/{idx:05d}.ply'
                        input_mesh.export(mesh_savepath)

                elif data[0] == 'cloud':
                    idx, input_cloud = data[1:]
                    if input_cloud is None:
                        continue
                    pts, colors = input_cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    vis.add_geometry(pcd)
                    if self.config.save_rendering:
                        cloud_savepath = f'{self.out_dir}/cloud/{idx:05d}.ply'
                        o3d.io.write_point_cloud(cloud_savepath, pcd)
                else:
                    print('Unknown show type: ', data[0])

            except Empty:
                break

            if len(self.estimate_c2w_list) % 2 == 0:
                vis.poll_events()
                vis.update_renderer()

    def spin(self, viz_queue):
        # create image visualizer
        self.fig, axes = plt.subplots(2, 3)
        self.fig.tight_layout()
        self.ax_depth, self.ax_rdepth, self.ax_depth_diff, \
            self.ax_color, self.ax_rcolor, self.ax_color_diff = axes.flatten()
        self.depth_plot = self.ax_depth.imshow(np.zeros(
            (self.config.img_show_h, self.config.img_show_w)),
                                               cmap='plasma')
        self.rdepth_plot = self.ax_rdepth.imshow(np.zeros(
            (self.config.img_show_h, self.config.img_show_w)),
                                                 cmap='plasma')
        self.depth_diff_plot = self.ax_depth_diff.imshow(np.zeros(
            (self.config.img_show_h, self.config.img_show_w)),
                                                         cmap='plasma')
        self.color_plot = self.ax_color.imshow(np.zeros(
            (self.config.img_show_h, self.config.img_show_w, 3)),
                                               cmap='plasma')
        self.rcolor_plot = self.ax_rcolor.imshow(np.zeros(
            (self.config.img_show_h, self.config.img_show_w, 3)),
                                                 cmap='plasma')
        self.color_diff_plot = self.ax_color_diff.imshow(np.zeros(
            (self.config.img_show_h, self.config.img_show_w, 3)),
                                                         cmap='plasma')
        self.ax_depth.set_title('Input Depth')
        self.ax_depth.set_xticks([])
        self.ax_depth.set_yticks([])
        self.ax_rdepth.set_title('Generated Depth')
        self.ax_rdepth.set_xticks([])
        self.ax_rdepth.set_yticks([])
        self.ax_depth_diff.set_title('Depth Residual')
        self.ax_depth_diff.set_xticks([])
        self.ax_depth_diff.set_yticks([])
        self.ax_color.set_title('Input RGB')
        self.ax_color.set_xticks([])
        self.ax_color.set_yticks([])
        self.ax_rcolor.set_title('Generated RGB')
        self.ax_rcolor.set_xticks([])
        self.ax_rcolor.set_yticks([])
        self.ax_color_diff.set_title('RGB Residual')
        self.ax_color_diff.set_xticks([])
        self.ax_color_diff.set_yticks([])
        plt.show(block=False)

        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        partial_callback = functools.partial(self.animation_callback,
                                             viz_queue=viz_queue)
        vis.register_animation_callback(partial_callback)
        vis.create_window('XRslam',
                          width=self.config.win_w,
                          height=self.config.win_h,
                          visible=True)
        vis.get_render_option().point_size = 4
        vis.get_render_option().mesh_show_back_face = True

        # run open3d visualizer
        vis.run()
        vis.destroy_window()
        plt.close(self.fig)
