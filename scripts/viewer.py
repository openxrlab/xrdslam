from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import torch
import tyro
from tqdm import tqdm

from deepslam.configs.base_config import PrintableConfig
from scripts.utils.viz_utils import SLAMFrontend


@dataclass
class OfflineViewerConfig(PrintableConfig):
    vis_dir: Path = Path('outputs')
    """Offline data path from xrdslam."""
    save_rendering: bool = True
    """Save the rendering result or not."""
    method_name: Optional[str] = None


@dataclass
class OfflineViewer:
    """Start the offline viewer."""

    config: OfflineViewerConfig

    def main(self) -> None:
        """Main function."""
        vis_dir = self.config.vis_dir
        method_name = self.config.method_name
        # read trajs
        if os.path.exists(vis_dir):
            eval_file = [
                os.path.join(vis_dir, f) for f in sorted(os.listdir(vis_dir))
                if 'tar' in f
            ]
            if len(eval_file) > 0:
                traj_path = eval_file[-1]
                traj = torch.load(traj_path, map_location=torch.device('cpu'))
                estimate_c2w_list = traj['estimate_c2w_list']
                gt_c2w_list = traj['gt_c2w_list']
                N = traj['idx']

        frontend = SLAMFrontend(vis_dir,
                                init_pose=estimate_c2w_list[0],
                                cam_scale=0.1,
                                save_rendering=self.config.save_rendering,
                                near=0,
                                estimate_c2w_list=estimate_c2w_list,
                                gt_c2w_list=gt_c2w_list,
                                method_name=method_name).start()

        for i in tqdm(range(0, N + 1)):
            img_file = f'{vis_dir}/imgs/{i:05d}.jpg'
            if os.path.exists(img_file):
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.axis('off')  # Turn off axis labels
                plt.show(block=False)
            plt.pause(0.005)
            meshfile = f'{vis_dir}/mesh/{i:05d}.ply'
            if os.path.isfile(meshfile):
                frontend.update_mesh(meshfile)
            else:
                cloudfile = f'{vis_dir}/cloud/{i:05d}.ply'
                if os.path.isfile(cloudfile):
                    frontend.update_cloud(cloudfile)
            frontend.update_pose(1, estimate_c2w_list[i], gt=False)
            # Note: not show gt_traj for splaTAM
            if method_name != 'splaTAM':
                frontend.update_pose(1, gt_c2w_list[i], gt=True)
            # the visualizer might get stuck if update every frame
            # with a long sequence (10000+ frames)
            if i % 10 == 0:
                frontend.update_cam_trajectory(i, gt=False)
                # Note: not show gt_traj for splaTAM
                if method_name != 'splaTAM':
                    frontend.update_cam_trajectory(i, gt=True)

        if self.config.save_rendering:
            time.sleep(1)
            os.system(f"/usr/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i \
                    '{vis_dir}/tmp_rendering/*.jpg' -y {vis_dir}/vis.mp4")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color('bright_yellow')
    tyro.cli(tyro.conf.FlagConversionOff[OfflineViewer]).main()


if __name__ == '__main__':
    entrypoint()
