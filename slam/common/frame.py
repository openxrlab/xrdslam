from typing import List

import torch
import torch.nn as nn
from torch.nn import Parameter

from slam.utils.opt_pose import OptimizablePose


class Frame(nn.Module):
    def __init__(self,
                 fid,
                 rgb,
                 depth,
                 init_pose=None,
                 gt_pose=None,
                 separate_LR=False,
                 rot_rep='axis_angle') -> None:
        super().__init__()
        self.fid = fid
        self.h, self.w = depth.shape
        self.rgb = rgb
        self.depth = depth
        self.gt_pose = gt_pose
        self.separate_LR = separate_LR
        self.rot_rep = rot_rep
        self.is_final_frame = False

        if init_pose is not None:
            pose = torch.tensor(init_pose,
                                requires_grad=True,
                                dtype=torch.float32)
            self.pose = OptimizablePose.from_matrix(Rt=pose,
                                                    separate_LR=separate_LR,
                                                    rot_rep=rot_rep)
            # check_consistency
            if not torch.allclose(
                    pose.detach(), self.pose.matrix().detach(), atol=1e-3):
                error_message = 'Transformation inconsistency detected!'
                raise ValueError(error_message, pose, self.pose.matrix())
        else:
            self.pose = None

    def set_pose(self, pose_np, separate_LR=False, rot_rep='axis_angle'):
        pose = torch.tensor(pose_np, requires_grad=True, dtype=torch.float32)
        self.pose = OptimizablePose.from_matrix(pose,
                                                separate_LR=separate_LR,
                                                rot_rep=rot_rep)

    def get_pose(self):
        return self.pose.matrix()

    def get_translation(self):
        return self.pose.translation()

    def get_rotation(self):
        return self.pose.rotation()

    def get_params(self) -> List[Parameter]:
        pose_params = []
        if self.pose is not None:
            if self.separate_LR:
                if self.rot_rep == 'quat':
                    pose_params.append(self.pose.data_q)
                    pose_params.append(self.pose.data_t)
                elif self.rot_rep == 'axis_angle':
                    pose_params.append(self.pose.data_r)
                    pose_params.append(self.pose.data_t)
            else:
                pose_params = list(self.pose.parameters())
        return pose_params
