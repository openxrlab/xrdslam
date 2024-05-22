import os
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import open3d as o3d
import torch
import trimesh
from packaging import version
from torch.utils.data import DataLoader
from tqdm import tqdm

from slam.common.common import clean_mesh, cull_mesh, save_render_imgs
from slam.common.datasets import BaseDataset
from slam.common.frame import Frame
from slam.configs.base_config import InstantiateConfig


@dataclass
class TrackerConfig(InstantiateConfig):
    """Tracker  Config."""
    _target: Type = field(default_factory=lambda: Tracker)
    render_freq: int = 1
    map_every: int = 1
    lazy_start: int = -1  # if to map every frame at beginning, default not
    use_relative_pose: bool = False
    save_debug_result: bool = False
    save_gt_mesh: bool = False
    save_re_render_result: bool = True  # for evaluation
    init_pose_offset: int = 0


class Tracker():
    def __init__(self,
                 config: TrackerConfig,
                 dataset: BaseDataset,
                 enable_vis: bool,
                 out_dir='.') -> None:
        self.config = config
        self.out_dir = out_dir
        self.enable_vis = enable_vis
        self.dataset = dataset

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(f'{self.out_dir}/mesh', exist_ok=True)
        os.makedirs(f'{self.out_dir}/cloud', exist_ok=True)
        os.makedirs(f'{self.out_dir}/imgs', exist_ok=True)
        # for eval img
        self.frame_cnt, self.psnr_sum, self.ssim_sum, self.lpips_sum, \
            self.depth_l1_render_sum = 0, 0, 0, 0, 0

    def spin(self, map_buffer, algorithm, viz_buffer, event_ready,
             event_processed):
        pbar = tqdm(
            DataLoader(self.dataset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=1))
        gt_depth, imu_datas = None, None
        for cur_data in pbar:
            if self.dataset.data_format == 'RGBD':
                idx, gt_color, gt_depth, gt_c2w = cur_data
            else:
                idx, gt_color, imu_datas, gt_c2w = cur_data

            idx_np = idx[0].cpu().numpy()
            if gt_depth is not None:
                gt_depth_np = gt_depth[0].cpu().numpy()
            else:
                gt_depth_np = None
            gt_color_np = gt_color[0].cpu().numpy()
            gt_c2w = gt_c2w[0]
            gt_c2w_np = gt_c2w.cpu().numpy()
            gt_c2w_ori = torch.from_numpy(gt_c2w_np)

            # use relative pose
            if self.config.use_relative_pose:
                if idx == 0:
                    first_pose_old = gt_c2w_np
                    gt_c2w_np = np.eye(4)
                    # NOTE init_pose_offset for octree.
                    gt_c2w_np[:3, 3] += self.config.init_pose_offset
                    first_pose_new = gt_c2w_np
                else:
                    delta = np.linalg.inv(first_pose_old) @ gt_c2w_np
                    gt_c2w_np = first_pose_new @ delta
                gt_c2w = torch.from_numpy(gt_c2w_np)
            else:
                gt_c2w = torch.from_numpy(gt_c2w_np)

            init_pose = self.predict_current_pose(
                idx_np,
                gt_c2w_np,
                estimate_c2w_list=algorithm.get_estimate_c2w_list())

            current_frame = Frame(fid=idx_np,
                                  rgb=gt_color_np,
                                  depth=gt_depth_np,
                                  gt_pose=gt_c2w_np,
                                  init_pose=init_pose,
                                  separate_LR=algorithm.is_separate_LR(),
                                  rot_rep=algorithm.get_rot_rep())

            candidate_c2w = None
            # optimize curframe pose
            candidate_c2w = algorithm.do_tracking(current_frame)
            if algorithm.is_initialized():
                current_frame.set_pose(candidate_c2w,
                                       separate_LR=algorithm.is_separate_LR(),
                                       rot_rep=algorithm.get_rot_rep())

            # for visualize and  extract mesh
            cur_c2w = current_frame.get_pose()
            cur_c2w_np = cur_c2w.clone().detach().cpu().numpy()
            algorithm.add_framepose(cur_c2w.detach(), gt_c2w, gt_c2w_ori)

            if self.enable_vis and algorithm.is_initialized(
            ) and self.config.render_freq > 0 and (
                    idx_np) % self.config.render_freq == 0:
                # send to visualizer
                self.visualize_results(viz_buffer, idx_np, cur_c2w_np,
                                       gt_c2w_np, gt_color_np, gt_depth_np,
                                       algorithm)
            if not self.enable_vis and self.config.save_debug_result and \
                algorithm.is_initialized() and self.config.render_freq > 0 \
                    and ((idx_np) % self.config.render_freq == 0
                         or idx_np == len(self.dataset) - 1):
                # save debug results
                result_2d = self.save_debug_results(algorithm, idx_np,
                                                    gt_color_np, gt_depth_np,
                                                    cur_c2w_np)
                if result_2d is not None:
                    psnr, ssim, lpips, depth_l1_render = result_2d
                    self.psnr_sum += psnr
                    self.ssim_sum += ssim
                    self.lpips_sum += lpips
                    self.depth_l1_render_sum += depth_l1_render
                    self.frame_cnt += 1
                if idx_np == len(self.dataset) - 1 and self.frame_cnt > 0:
                    # print 2d debug render metric
                    avg_psnr = self.psnr_sum / self.frame_cnt
                    avg_ssim = self.ssim_sum / self.frame_cnt
                    avg_lpips = self.lpips_sum / self.frame_cnt
                    avg_depth_l1 = self.depth_l1_render_sum / self.frame_cnt
                    print(f'<debug> avg_psnr[dB]: {avg_psnr}, avg_ms_ssim:'
                          f' {avg_ssim}, avg_lpips: {avg_lpips},'
                          f' avg_depth_l1[cm]: {avg_depth_l1}')

            # check mapframe and send the mapframe to mapper
            is_mapframe = self.check_mapframe(current_frame, map_buffer)
            # wait for mapping
            if is_mapframe:
                event_ready.set()
                event_processed.wait()
                event_processed.clear()

            torch.cuda.empty_cache()

        # re-rendering for evaluation
        if not self.enable_vis and not self.config.save_debug_result \
           and self.config.save_re_render_result:
            print('Starting re-rendering frames...')
            self.save_re_render_frames(algorithm)

        if self.config.save_gt_mesh:
            self.save_gt_mesh()

        # set finished
        algorithm.set_finished()

    def check_mapframe(self, check_frame, map_buffer):
        if check_frame.fid <= self.config.lazy_start:
            map_every = 1
        else:
            map_every = self.config.map_every
        # send to mapper
        if map_every != -1 and (check_frame.fid % map_every == 0
                                or check_frame.fid == len(self.dataset) - 1):
            check_frame.is_final_frame = (
                check_frame.fid == len(self.dataset) - 1)
            map_buffer.put(check_frame, block=True)
            return True
        return False

    def predict_current_pose(self, frame_id, gt_c2w_np, estimate_c2w_list):
        """Predict current pose from previous pose using camera motion
        model."""
        if frame_id < 1:
            return gt_c2w_np
        elif frame_id == 1:
            return estimate_c2w_list[frame_id - 1].detach().cpu().numpy()
        else:
            c2w_est_prev_prev = estimate_c2w_list[frame_id -
                                                  2].detach().cpu().numpy()
            c2w_est_prev = estimate_c2w_list[frame_id -
                                             1].detach().cpu().numpy()
            delta = c2w_est_prev @ np.linalg.inv(c2w_est_prev_prev)
            init_c2w_np = delta @ c2w_est_prev
            return init_c2w_np

    def visualize_results(self, viz_buffer, idx_np, cur_c2w_np, gt_c2w_np,
                          gt_color_np, gt_depth_np, algorithm):
        # pose
        viz_buffer.put_nowait(('pose', idx_np, cur_c2w_np, gt_c2w_np))
        # render img
        rcolor_np, rdepth_np = algorithm.render_img(
            c2w=cur_c2w_np, gt_depth=gt_depth_np,
            idx=idx_np)  # color: [H, W, C], depth: [H, W]
        viz_buffer.put_nowait(
            ('img', idx_np, gt_color_np, gt_depth_np, rcolor_np, rdepth_np))
        # extract mesh
        mesh = algorithm.get_mesh()
        if mesh is not None:
            culled_mesh = cull_mesh(
                dataset=self.dataset,
                mesh=mesh,
                estimate_c2w_list=algorithm.get_estimate_c2w_list(),
                eval_rec=True)
            viz_buffer.put_nowait(('mesh', idx_np, culled_mesh))
        # get pointcloud
        cloud = algorithm.get_cloud(cur_c2w_np, gt_depth_np=gt_depth_np)
        if cloud is not None:
            viz_buffer.put_nowait(('cloud', idx_np, cloud))

    def save_debug_results(self, algorithm, idx_np, gt_color_np, gt_depth_np,
                           cur_c2w_np):
        result_2d = None
        # save render imgs
        imgs_save_path = f'{self.out_dir}/imgs'
        rcolor_np, rdepth_np = algorithm.render_img(c2w=cur_c2w_np,
                                                    gt_depth=gt_depth_np,
                                                    idx=idx_np)
        result_2d = save_render_imgs(idx_np,
                                     gt_color_np=gt_color_np,
                                     gt_depth_np=gt_depth_np,
                                     color_np=rcolor_np,
                                     depth_np=rdepth_np,
                                     img_save_dir=imgs_save_path)
        if result_2d is not None:
            psnr, ssim, lpips, depth_l1_render = result_2d
            print(f'idx: {idx_np}, psnr[dB]: {psnr}, ssim: {ssim},'
                  f'lpips: {lpips}, depth_l1_render[cm]: {depth_l1_render}')
        # save mesh
        mesh_savepath = f'{self.out_dir}/mesh/{idx_np:05d}.ply'
        mesh = algorithm.get_mesh()
        if mesh is not None:
            mesh.export(mesh_savepath)
            if idx_np == len(self.dataset) - 1:
                mesh_savepath = f'{self.out_dir}/final_mesh.ply'
                mesh.export(mesh_savepath)
                mesh_savepath = f'{self.out_dir}/final_mesh_rec.ply'
                culled_mesh = cull_mesh(
                    dataset=self.dataset,
                    mesh=mesh,
                    estimate_c2w_list=algorithm.get_estimate_c2w_list(),
                    eval_rec=True)
                culled_mesh.export(mesh_savepath)
        # save cloud
        cloud_savepath = f'{self.out_dir}/cloud/{idx_np:05d}.ply'
        cloud = algorithm.get_cloud(
            cur_c2w_np, gt_depth_np=gt_depth_np)  # o3d.geometry.PointCloud
        if cloud is not None:
            pts, colors = cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(cloud_savepath, pcd)
        # save traj
        traj_savepath = os.path.join(self.out_dir, 'eval.tar')
        torch.save(
            {
                'gt_c2w_list_ori': algorithm.get_gt_c2w_list_ori(),
                'gt_c2w_list': algorithm.get_gt_c2w_list(),
                'estimate_c2w_list': algorithm.get_estimate_c2w_list(),
                'idx': torch.tensor(idx_np),
            },
            traj_savepath,
            _use_new_zipfile_serialization=False)

        return result_2d

    def save_gt_mesh(self):
        H, W, fx, fy, cx, cy = (self.dataset.camera.height,
                                self.dataset.camera.width,
                                self.dataset.camera.fx, self.dataset.camera.fy,
                                self.dataset.camera.cx, self.dataset.camera.cy)
        voxel_size = 4
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=float(voxel_size) / 100.0,
                sdf_trunc=3 * float(voxel_size) / 100,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=float(voxel_size) / 100.0,
                sdf_trunc=3 * float(voxel_size) / 100,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        for cur_data in DataLoader(self.dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=1):
            if self.dataset.data_format == 'RGBD':
                idx, gt_color, gt_depth, gt_c2w = cur_data
            else:
                print(
                    'There are no depth image, Not support generate gt mesh!')
                return
            gt_depth_np = gt_depth[0].cpu().numpy()
            gt_color_np = gt_color[0].cpu().numpy()
            gt_c2w = gt_c2w[0]
            gt_c2w_np = gt_c2w.cpu().numpy()

            c2w = gt_c2w_np
            # convert to open3d camera pose
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            depth = gt_depth_np
            color = gt_color_np
            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)
        o3d_mesh = volume.extract_triangle_mesh()
        vertices = o3d_mesh.vertices
        faces = o3d_mesh.triangles
        gt_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_savepath = f'{self.out_dir}/gt_mesh.ply'
        gt_mesh.export(mesh_savepath)
        gt_mesh_clean = clean_mesh(o3d_mesh)
        mesh_savepath = f'{self.out_dir}/gt_mesh_clean.ply'
        gt_mesh_clean.export(mesh_savepath)

    def save_re_render_frames(self, algorithm):
        self.frame_cnt, self.psnr_sum, self.ssim_sum, self.lpips_sum, \
            self.depth_l1_render_sum = 0, 0, 0, 0, 0
        pbar = tqdm(
            DataLoader(self.dataset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=1))
        estimate_c2w_list = algorithm.get_estimate_c2w_list()

        gt_depth, imu_datas = None, None
        for cur_data in pbar:
            if self.dataset.data_format == 'RGBD':
                idx, gt_color, gt_depth, gt_c2w = cur_data
            else:
                idx, gt_color, imu_datas, gt_c2w = cur_data

            idx_np = idx[0].cpu().numpy()
            if gt_depth is not None:
                gt_depth_np = gt_depth[0].cpu().numpy()
            else:
                gt_depth_np = None
            gt_color_np = gt_color[0].cpu().numpy()
            cur_c2w_np = estimate_c2w_list[idx_np].detach().cpu().numpy()
            if idx_np % self.config.render_freq == 0:
                # save imgs
                imgs_save_path = f'{self.out_dir}/imgs'
                rcolor_np, rdepth_np = algorithm.render_img(
                    c2w=cur_c2w_np, gt_depth=gt_depth_np, idx=idx_np)
                result_2d = save_render_imgs(idx_np,
                                             gt_color_np=gt_color_np,
                                             gt_depth_np=gt_depth_np,
                                             color_np=rcolor_np,
                                             depth_np=rdepth_np,
                                             img_save_dir=imgs_save_path)
                if result_2d is not None:
                    psnr, ssim, lpips, depth_l1_render = result_2d
                    print(f'idx: {idx_np}, psnr[dB]: {psnr}, ssim: {ssim},'
                          f'lpips: {lpips}, depth_l1_render[cm]:'
                          f'{depth_l1_render}')
                    self.psnr_sum += psnr
                    self.ssim_sum += ssim
                    self.lpips_sum += lpips
                    self.depth_l1_render_sum += depth_l1_render
                    self.frame_cnt += 1
            # save final mesh
            if idx_np == len(self.dataset) - 1:
                mesh = algorithm.get_mesh()
                if mesh is not None:
                    mesh_savepath = f'{self.out_dir}/final_mesh.ply'
                    mesh.export(mesh_savepath)
                    mesh_savepath = f'{self.out_dir}/final_mesh_rec.ply'
                    culled_mesh = cull_mesh(
                        dataset=self.dataset,
                        mesh=mesh,
                        estimate_c2w_list=algorithm.get_estimate_c2w_list(),
                        eval_rec=True)
                    culled_mesh.export(mesh_savepath)

        # print 2d render metric
        if self.frame_cnt > 0:
            avg_psnr = self.psnr_sum / self.frame_cnt
            avg_ssim = self.ssim_sum / self.frame_cnt
            avg_lpips = self.lpips_sum / self.frame_cnt
            avg_depth_l1 = self.depth_l1_render_sum / self.frame_cnt
            print(f'avg_psnr[dB]: {avg_psnr}, avg_ms_ssim: {avg_ssim},'
                  f'avg_lpips: {avg_lpips}, avg_depth_l1[cm]: {avg_depth_l1}')
        # save final traj
        traj_savepath = os.path.join(self.out_dir, 'eval.tar')
        torch.save(
            {
                'gt_c2w_list_ori': algorithm.get_gt_c2w_list_ori(),
                'gt_c2w_list': algorithm.get_gt_c2w_list(),
                'estimate_c2w_list': algorithm.get_estimate_c2w_list(),
                'idx': idx,
            },
            traj_savepath,
            _use_new_zipfile_serialization=False)
