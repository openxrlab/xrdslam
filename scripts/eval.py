from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

from scripts.utils.eval_ate import convert_poses, evaluate
from scripts.utils.eval_recon import (calc_2d_metric, calc_3d_metric,
                                      calc_3d_metric_New)


@dataclass
class EvalMatrics:
    """Evaluate trajectory accuracy and 3D reconstruction quality."""

    # Path to xrdslam running result.
    output_dir: Path
    # Path to groundtruth mesh file.
    gt_mesh: Optional[str] = 'None'

    eval_traj: bool = True
    eval_recon: bool = True
    correct_scale: bool = False

    distance_thresh: float = 0.01
    eval_2d_metric: bool = True

    def main(self) -> None:
        """Main function."""
        output = self.output_dir
        eval_pose_tar = f'{output}/eval.tar'
        eval_mesh = f'{output}/final_mesh_rec.ply'
        traj_result = None
        if self.eval_traj:
            if not os.path.exists(eval_pose_tar):
                print('traj file: ', eval_pose_tar, ' is not exist!')
                return
            ckpt = torch.load(eval_pose_tar, map_location=torch.device('cpu'))
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list_ori']
            N = ckpt['idx']
            poses_gt, mask = convert_poses(gt_c2w_list, N, 1.0)
            poses_est, _ = convert_poses(estimate_c2w_list, N, 1.0)
            poses_est = poses_est[mask]
            traj_result = evaluate(poses_gt,
                                   poses_est,
                                   plot=f'{output}/eval_ate_plot.png',
                                   correct_scale=self.correct_scale)
            print(traj_result)
        if self.eval_recon:
            if self.gt_mesh is None or not os.path.exists(self.gt_mesh):
                print('gt_mesh file: ', self.gt_mesh, ' is not exist!')
                return
            # use transform_matrix from traj_align_result
            transform_matrix = None
            if traj_result is not None:
                rot = traj_result['rot']
                trans = traj_result['trans']
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot
                transform_matrix[:3, 3] = trans.reshape(-1)
            result_3d = calc_3d_metric(eval_mesh,
                                       self.gt_mesh,
                                       transform_matrix=transform_matrix)
            result_3d_more = calc_3d_metric_New(
                eval_mesh,
                self.gt_mesh,
                distance_thresh=self.distance_thresh,
                transform_matrix=transform_matrix)

            result = result_3d | result_3d_more

            if self.eval_2d_metric:
                result_2d = calc_2d_metric(eval_mesh,
                                           self.gt_mesh,
                                           transform_matrix=transform_matrix,
                                           n_imgs=1000)
                result = result_2d | result
            print(result)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color('bright_yellow')
    tyro.cli(EvalMatrics).main()


if __name__ == '__main__':
    entrypoint()
